"""
run_experiment2.py — Experiment 2: Label-Efficiency Sweep
==========================================================
Compare CLKD-FT (distilled + fine-tuned) vs B5 (plain fine-tuning)
across varying amounts of Indonesian adaptation data.

N per class: 10, 25, 50, 100, 200, (max available)
Seeds: 42, 123, 999, 2026, 777
Test set: same 154 samples throughout

Key requirements from Prof. Huang:
  1. Same test set for every N
  2. Nested subsets (N=10 ⊂ N=25 ⊂ N=50 ⊂ ...)
  3. Reuse teacher + distilled student checkpoints

Total: 6 N × 5 seeds × 2 models = 60 fine-tuning runs (~4-5 hours)

Output: results/experiment2_sweep_results.json
"""

import json, os, random, time, copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    XLMRobertaForSequenceClassification,
    AutoTokenizer,
)
from sklearn.model_selection import train_test_split
import pandas as pd

import config
from utils import (
    FakeNewsDataset, evaluate_model, load_data, clean_gpu,
    print_gpu_report, device,
    wilcoxon_test, cohens_d,
)
from train_clkd import (
    train_teacher, distill_student, finetune_student,
    StudentWithSentimentHead,
)

# ── Configuration ─────────────────────────────────────────────
N_VALUES = [10, 25, 50, 100, 200, 400]  # per class
SEEDS    = config.RANDOM_SEEDS           # [42, 123, 999, 2026, 777]
METRIC_KEYS = ["accuracy", "auc_roc", "sensitivity", "specificity", "f1"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Optimizer / AMP helpers (same as pipeline) ────────────────
def _get_optimizer(params):
    if config.USE_8BIT_ADAM and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            return bnb.optim.Adam8bit(params, lr=config.LEARNING_RATE)
        except ImportError:
            pass
    from torch.optim import AdamW
    return AdamW(params, lr=config.LEARNING_RATE)


def _autocast():
    if not config.USE_AMP or not torch.cuda.is_available():
        return torch.cuda.amp.autocast(enabled=False)
    return torch.cuda.amp.autocast(enabled=True, dtype=config.AMP_DTYPE)


def _get_scaler():
    needs = (config.USE_AMP and torch.cuda.is_available()
             and config.AMP_DTYPE == torch.float16)
    return torch.cuda.amp.GradScaler(enabled=needs)


# ── Data preparation with nested subsets ──────────────────────
def prepare_sweep_data(seed):
    """
    Prepare data with nested subsets following Prof. Huang's instruction:
      pool = train_id_class_k.sample(frac=1, random_state=seed)
      subset_N = pool.iloc[:N]

    Returns:
      test_id: fixed test set (154 samples)
      nested_subsets: dict {N: DataFrame} with nested training subsets
      train_en: English training data (for teacher/distillation)
      pool_sizes: dict with available samples per class
    """
    df_en_raw = load_data(config.EN_DATA_PATH, verbose=False)
    df_id_raw = load_data(config.ID_DATA_PATH, verbose=False)

    # Balance English data (same as prepare_datasets)
    df_en_fake = df_en_raw[df_en_raw.label == 1]
    df_en_real = df_en_raw[df_en_raw.label == 0].sample(
        n=len(df_en_fake), random_state=seed
    )
    df_en_balanced = pd.concat([df_en_fake, df_en_real]).sample(
        frac=1, random_state=seed
    )
    train_en, _ = train_test_split(
        df_en_balanced, test_size=config.TEST_SIZE_EN,
        stratify=df_en_balanced["label"], random_state=seed,
    )

    # Indonesian train/test split (same split as main pipeline)
    train_id_full, test_id = train_test_split(
        df_id_raw, test_size=config.TEST_SIZE_ID,
        stratify=df_id_raw["label"], random_state=seed,
    )

    # Separate pools per class and shuffle once per seed
    pool_real = (train_id_full[train_id_full.label == 0]
                 .sample(frac=1, random_state=seed)
                 .reset_index(drop=True))
    pool_fake = (train_id_full[train_id_full.label == 1]
                 .sample(frac=1, random_state=seed)
                 .reset_index(drop=True))

    pool_sizes = {"real": len(pool_real), "fake": len(pool_fake)}
    max_per_class = min(len(pool_real), len(pool_fake))

    print(f"  Pool sizes: Real={len(pool_real)}, Fake={len(pool_fake)}")
    print(f"  Max N per class: {max_per_class}")
    print(f"  Test set: {len(test_id)} samples")

    # Build nested subsets using prefix (N=10 ⊂ N=25 ⊂ N=50 ⊂ ...)
    nested_subsets = {}
    for N in N_VALUES:
        n_actual = min(N, max_per_class)
        subset = pd.concat([
            pool_real.iloc[:n_actual],
            pool_fake.iloc[:n_actual],
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)
        nested_subsets[N] = subset

        if n_actual < N:
            print(f"  N={N}: capped at {n_actual} per class "
                  f"({n_actual * 2} total) — pool exhausted")
        else:
            print(f"  N={N}: {n_actual} per class ({n_actual * 2} total)")

    return {
        "train_en": train_en,
        "test_id": test_id,
        "nested_subsets": nested_subsets,
        "pool_sizes": pool_sizes,
    }


# ── B5: Plain fine-tuning (no distillation) ───────────────────
def run_b5_finetune(train_subset, test_id, seed, N):
    """Fine-tune fresh XLM-R Base on N Indonesian samples."""
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        config.STUDENT_MODEL, num_labels=config.NUM_LABELS
    ).to(device)

    train_loader = DataLoader(
        FakeNewsDataset(train_subset.content, train_subset.label, tokenizer),
        batch_size=config.BATCH_SIZE, shuffle=True,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    optimizer = _get_optimizer(model.parameters())
    scaler = _get_scaler()

    model.train()
    for epoch in range(config.EPOCHS_FINETUNE):
        total_loss, n_batches = 0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            with _autocast():
                out = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(
                        device, non_blocking=True),
                    labels=batch["labels"].to(device, non_blocking=True),
                )
                loss = out.loss / config.ACC_STEPS
            scaler.scale(loss).backward()
            total_loss += out.loss.item()
            n_batches += 1
            if (i + 1) % config.ACC_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    test_loader = DataLoader(
        FakeNewsDataset(test_id.content, test_id.label, tokenizer),
        batch_size=16,
    )
    metrics = evaluate_model(model, test_loader,
                             f"B5 N={N} seed={seed}", verbose=False)
    del model
    clean_gpu()
    return metrics


# ── CLKD-FT: Distilled student + fine-tuning ─────────────────
def run_clkd_ft_finetune(distilled_state_dict, train_subset, test_id,
                         seed, N):
    """
    Load a COPY of the distilled student checkpoint,
    fine-tune on N Indonesian samples, evaluate.
    """
    set_seed(seed)

    tokenizer_s = AutoTokenizer.from_pretrained(config.STUDENT_MODEL)
    base_model = XLMRobertaForSequenceClassification.from_pretrained(
        config.STUDENT_MODEL, num_labels=config.NUM_LABELS
    ).to(device)
    student = StudentWithSentimentHead(base_model).to(device)

    # Load distilled checkpoint (deep copy to avoid mutation)
    student.load_state_dict(copy.deepcopy(distilled_state_dict))

    # Fine-tune on Indonesian subset
    student.train()
    if hasattr(student, 'base'):
        student.base.gradient_checkpointing_enable()

    backbone_params = list(student.base.parameters())
    head_params = list(student.sentiment_head.parameters())

    from train_clkd import _get_optimizer as clkd_get_optimizer
    optimizer = clkd_get_optimizer(backbone_params, head_params)
    scaler = _get_scaler()

    train_loader = DataLoader(
        FakeNewsDataset(train_subset.content, train_subset.label, tokenizer_s),
        batch_size=config.BATCH_SIZE, shuffle=True,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    for epoch in range(config.EPOCHS_FINETUNE):
        total_loss, n_batches = 0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            with _autocast():
                out = student(
                    batch["input_ids"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(
                        device, non_blocking=True),
                    labels=batch["labels"].to(device, non_blocking=True),
                )
                loss = out.loss / config.ACC_STEPS
            scaler.scale(loss).backward()
            total_loss += out.loss.item()
            n_batches += 1
            if (i + 1) % config.ACC_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Evaluate using base model (compatible with evaluate_model)
    eval_model = student.base if hasattr(student, 'base') else student
    test_loader = DataLoader(
        FakeNewsDataset(test_id.content, test_id.label, tokenizer_s),
        batch_size=16,
    )
    metrics = evaluate_model(eval_model, test_loader,
                             f"CLKD-FT N={N} seed={seed}", verbose=False)
    del student, eval_model
    clean_gpu()
    return metrics


# ── Main ──────────────────────────────────────────────────────
def main():
    clean_gpu()
    start_time = time.time()

    print("=" * 70)
    print("  EXPERIMENT 2: Label-Efficiency Sweep")
    print(f"  CLKD-FT vs B5 | N per class: {N_VALUES}")
    print(f"  Seeds: {SEEDS} | Total runs: "
          f"{len(N_VALUES)} × {len(SEEDS)} × 2 = "
          f"{len(N_VALUES) * len(SEEDS) * 2}")
    print("=" * 70)

    all_results = []   # list of per-seed results
    run_counter = 0
    total_runs = len(N_VALUES) * len(SEEDS) * 2

    for si, seed in enumerate(SEEDS):
        print(f"\n{'#' * 70}")
        print(f"  SEED {si+1}/{len(SEEDS)}: {seed}")
        print(f"{'#' * 70}")

        # ── 1. Prepare data with nested subsets ───────────────
        set_seed(seed)
        data = prepare_sweep_data(seed)
        train_en = data["train_en"]
        test_id = data["test_id"]
        nested = data["nested_subsets"]

        # ── 2. Train teacher (reuse per seed) ────────────────
        print(f"\n  Training teacher for seed {seed}...")
        teacher, tok_t = train_teacher(train_en, seed=seed)

        # ── 3. Distill student on English (reuse per seed) ───
        print(f"\n  Distilling student for seed {seed}...")
        distilled_student, _, _ = distill_student(
            teacher, tok_t, train_en, mode="full", seed=seed
        )
        # Save state dict for reuse across N values
        distilled_state = copy.deepcopy(distilled_student.state_dict())
        del distilled_student, teacher
        clean_gpu()

        # ── 4. Sweep across N values ─────────────────────────
        seed_results = {"seed": seed}

        for N in N_VALUES:
            subset = nested[N]
            n_actual = len(subset) // 2  # per class

            # ── B5: plain fine-tuning ─────────────────────────
            run_counter += 1
            print(f"\n  [{run_counter}/{total_runs}] B5 | N={N} "
                  f"(actual={n_actual}/class) | seed={seed}")
            set_seed(seed)
            b5_metrics = run_b5_finetune(subset, test_id, seed, N)
            print(f"    B5  F1={b5_metrics['f1']:.4f}")

            # ── CLKD-FT: distilled + fine-tuning ─────────────
            run_counter += 1
            print(f"  [{run_counter}/{total_runs}] CLKD-FT | N={N} "
                  f"(actual={n_actual}/class) | seed={seed}")
            set_seed(seed)
            ft_metrics = run_clkd_ft_finetune(
                distilled_state, subset, test_id, seed, N
            )
            print(f"    FT  F1={ft_metrics['f1']:.4f}  "
                  f"ΔF1={ft_metrics['f1'] - b5_metrics['f1']:+.4f}")

            seed_results[f"N{N}"] = {
                "N_per_class": n_actual,
                "N_total": len(subset),
                "b5": b5_metrics,
                "clkd_ft": ft_metrics,
            }

        all_results.append(seed_results)

    # ══════════════════════════════════════════════════════════
    # Aggregate and statistical tests
    # ══════════════════════════════════════════════════════════
    W = 80
    print(f"\n{'=' * W}")
    print(f"  EXPERIMENT 2 RESULTS — Label-Efficiency Sweep")
    print(f"{'=' * W}")

    summary = {}
    print(f"\n{'N':>5} {'B5 F1':>12} {'CLKD-FT F1':>12} {'ΔF1':>10} "
          f"{'p':>8} {'dz':>8} {'Sig':>5}")
    print("-" * W)

    for N in N_VALUES:
        key = f"N{N}"
        b5_f1s = [r[key]["b5"]["f1"] for r in all_results]
        ft_f1s = [r[key]["clkd_ft"]["f1"] for r in all_results]

        b5_mean, b5_std = np.mean(b5_f1s), np.std(b5_f1s)
        ft_mean, ft_std = np.mean(ft_f1s), np.std(ft_f1s)
        delta = ft_mean - b5_mean

        w = wilcoxon_test(ft_f1s, b5_f1s, "CLKD-FT", "B5")
        d = cohens_d(ft_f1s, b5_f1s)

        sig = "YES" if w["significant"] else "No"
        print(f"{N:>5} {b5_mean:>6.4f}±{b5_std:.4f} "
              f"{ft_mean:>6.4f}±{ft_std:.4f} "
              f"{delta:>+10.4f} {w['p_value']:>8.4f} "
              f"{d['d']:>+8.2f} {sig:>5}")

        # Per-seed detail
        n_actual = all_results[0][key]["N_per_class"]
        summary[key] = {
            "N_per_class": n_actual,
            "N_total": n_actual * 2,
            "b5": {"mean": b5_mean, "std": b5_std,
                   "per_seed": b5_f1s},
            "clkd_ft": {"mean": ft_mean, "std": ft_std,
                        "per_seed": ft_f1s},
            "delta_f1": delta,
            "wilcoxon": w,
            "cohens_d": d,
        }

        # Also aggregate all metrics
        for model_key in ["b5", "clkd_ft"]:
            for mk in METRIC_KEYS:
                vals = [r[key][model_key][mk] for r in all_results]
                summary[key][f"{model_key}_{mk}_mean"] = float(np.mean(vals))
                summary[key][f"{model_key}_{mk}_std"] = float(np.std(vals))

    print(f"{'=' * W}")

    # ── Find crossover point ──────────────────────────────────
    print(f"\n  CROSSOVER ANALYSIS:")
    last_sig_N = None
    first_nonsig_N = None
    for N in N_VALUES:
        key = f"N{N}"
        if summary[key]["wilcoxon"]["significant"]:
            last_sig_N = N
        elif last_sig_N is not None and first_nonsig_N is None:
            first_nonsig_N = N

    if last_sig_N and first_nonsig_N:
        print(f"  Distillation significant at N ≤ {last_sig_N} per class")
        print(f"  Convergence begins at N = {first_nonsig_N} per class")
        print(f"  Crossover region: N ∈ [{last_sig_N}, {first_nonsig_N}]")
    elif last_sig_N:
        print(f"  Distillation significant up to N = {last_sig_N} per class")
        print(f"  No convergence observed in tested range")
    else:
        print(f"  No significant difference at any N")
        print(f"  Distillation shows no benefit across the full range")

    # ── Save results ──────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    output = {
        "experiment": "Experiment 2: Label-Efficiency Sweep",
        "description": (
            "CLKD-FT vs B5 (plain fine-tuning) across varying amounts "
            "of Indonesian adaptation data. Nested subsets ensure "
            "variation comes from sample size, not article selection."
        ),
        "N_values_per_class": N_VALUES,
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "total_runs": total_runs,
        "summary": summary,
        "per_seed_runs": all_results,
    }

    path = os.path.join(config.RESULTS_DIR, "experiment2_sweep_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else
                  (x.tolist() if hasattr(x, "tolist") else x))

    elapsed = time.time() - start_time
    print(f"\n  Results saved → {path}")
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print_gpu_report()

    # ── Summary for advisor ───────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  SUMMARY FOR ADVISOR")
    print(f"{'=' * W}")
    for N in N_VALUES:
        key = f"N{N}"
        s = summary[key]
        sig = "***" if s["wilcoxon"]["significant"] else "n.s."
        print(f"  N={s['N_per_class']:>3}/class: "
              f"B5={s['b5']['mean']:.4f}  "
              f"CLKD-FT={s['clkd_ft']['mean']:.4f}  "
              f"Δ={s['delta_f1']:+.4f}  "
              f"p={s['wilcoxon']['p_value']:.4f} {sig}")
    print(f"{'=' * W}")


if __name__ == "__main__":
    main()
