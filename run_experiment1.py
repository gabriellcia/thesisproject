"""
run_experiment1.py — Experiment 1: No-Distillation Fine-Tuning Control (B5)
============================================================================
XLM-R Base fine-tuned on the same 200 Indonesian samples used by CLKD-FT,
but WITHOUT any teacher or distillation. Pure supervised cross-entropy only.

This answers: does distillation actually contribute, or is the 200-sample
fine-tuning doing all the work?

Five seeds (42, 123, 999, 2026, 777). Uses the same training infrastructure
as the rest of the pipeline (BF16, 8-bit Adam, gradient clipping).

Output: results/experiment1_b5_results.json
"""

import json, os, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaForSequenceClassification, AutoTokenizer

import config
from utils import (
    prepare_datasets, FakeNewsDataset, evaluate_model, clean_gpu,
    print_gpu_report, device,
    wilcoxon_test, cohens_d, recall_confidence_interval,
)

METRIC_KEYS = ["accuracy", "auc_roc", "sensitivity", "specificity", "f1"]
SEEDS = config.RANDOM_SEEDS  # [42, 123, 999, 2026, 777]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_optimizer(params):
    """Same optimizer setup as train_baselines.py."""
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


def train_b5_single_seed(seed):
    """
    Fine-tune XLM-R Base on 200 Indonesian samples (no teacher).
    Returns metrics dict.
    """
    set_seed(seed)
    clean_gpu()

    # ── Load data (same split as main pipeline) ──────────────
    data = prepare_datasets(random_state=seed, verbose=(seed == SEEDS[0]))
    train_id = data["train_id"]  # 200 samples (100 per class)
    test_id  = data["test_id"]   # 154 samples

    print(f"\n{'='*60}")
    print(f"  B5: XLM-R + ID Fine-Tune (No Distillation)")
    print(f"  Seed: {seed}")
    print(f"  Train: {len(train_id)} ID samples | Test: {len(test_id)} ID samples")
    print(f"  Epochs: {config.EPOCHS_FINETUNE} | LR: {config.LEARNING_RATE}")
    print(f"  Device: {device} | BF16: {config.USE_AMP}")
    print(f"{'='*60}")

    # ── Load model ───────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        config.STUDENT_MODEL, num_labels=config.NUM_LABELS
    ).to(device)

    # ── Training setup (matches existing pipeline) ───────────
    train_loader = DataLoader(
        FakeNewsDataset(train_id.content, train_id.label, tokenizer),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    optimizer = _get_optimizer(model.parameters())
    scaler = _get_scaler()

    # ── Training loop (pure cross-entropy, no teacher) ───────
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

        # Flush remaining gradients
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        avg_loss = total_loss / max(n_batches, 1)
        print(f"   Epoch {epoch+1}/{config.EPOCHS_FINETUNE} | Loss: {avg_loss:.4f}")

    # ── Evaluate ─────────────────────────────────────────────
    test_loader = DataLoader(
        FakeNewsDataset(test_id.content, test_id.label, tokenizer),
        batch_size=16,
    )
    metrics = evaluate_model(
        model, test_loader,
        f"B5: XLM-R + ID-FT [seed={seed}]"
    )

    # ── Recall CI (same as main pipeline) ────────────────────
    cm = np.array(metrics.get("cm", [[0, 0], [0, 0]]))
    if cm.shape == (2, 2):
        n_correct = cm[1, 1]
        n_total = cm[1, :].sum()
        ci_low, ci_high = recall_confidence_interval(n_correct, n_total)
        metrics["recall_fake_ci_95"] = [ci_low, ci_high]
        metrics["n_fake_test"] = int(n_total)

    del model
    clean_gpu()
    return metrics


def main():
    clean_gpu()
    start_time = time.time()

    print("=" * 70)
    print("  EXPERIMENT 1: No-Distillation Fine-Tuning Control (B5)")
    print(f"  XLM-R Base + {config.ID_TRAIN_PER_CLASS * 2} ID samples, "
          f"no teacher, {len(SEEDS)} seeds")
    print("=" * 70)

    # ── Run all seeds ────────────────────────────────────────
    all_runs = []
    for i, seed in enumerate(SEEDS):
        print(f"\n{'#'*70}")
        print(f"  RUN {i+1}/{len(SEEDS)} — Seed {seed}")
        print(f"{'#'*70}")
        metrics = train_b5_single_seed(seed)
        all_runs.append({"seed": seed, **metrics})

    # ── Aggregate mean ± std ─────────────────────────────────
    agg = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in all_runs]
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    # ── Print summary ────────────────────────────────────────
    W = 80
    print(f"\n{'='*W}")
    print(f"  EXPERIMENT 1 RESULTS — B5: XLM-R + ID-FT (No Distillation)")
    print(f"  {len(SEEDS)} seeds: {SEEDS}")
    print(f"{'='*W}")
    print(f"{'Seed':>6} {'Accuracy':>10} {'AUC-ROC':>10} {'Sensitiv':>10} "
          f"{'Specif':>10} {'F1':>10}")
    print(f"{'-'*W}")
    for r in all_runs:
        print(f"{r['seed']:>6} {r['accuracy']:>10.4f} {r['auc_roc']:>10.4f} "
              f"{r['sensitivity']:>10.4f} {r['specificity']:>10.4f} "
              f"{r['f1']:>10.4f}")
    print(f"{'-'*W}")
    print(f"{'Mean':>6} {agg['accuracy']['mean']:>10.4f} "
          f"{agg['auc_roc']['mean']:>10.4f} "
          f"{agg['sensitivity']['mean']:>10.4f} "
          f"{agg['specificity']['mean']:>10.4f} "
          f"{agg['f1']['mean']:>10.4f}")
    print(f"{'Std':>6} {agg['accuracy']['std']:>10.4f} "
          f"{agg['auc_roc']['std']:>10.4f} "
          f"{agg['sensitivity']['std']:>10.4f} "
          f"{agg['specificity']['std']:>10.4f} "
          f"{agg['f1']['std']:>10.4f}")
    print(f"{'='*W}")

    # ── Load existing results for comparison ─────────────────
    existing_path = os.path.join(config.RESULTS_DIR, "all_results.json")
    comparisons = {}
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)

        print(f"\n{'='*W}")
        print(f"  STATISTICAL COMPARISONS (paired sign test, one-sided)")
        print(f"{'='*W}")

        asr = existing.get("all_seed_runs", [])
        if len(asr) == len(SEEDS):
            # B5 vs CLKD-FT: does distillation add anything?
            ft_f1s = [asr[i]["ft"]["f1"] for i in range(len(SEEDS))]
            b5_f1s = [all_runs[i]["f1"] for i in range(len(SEEDS))]
            w = wilcoxon_test(ft_f1s, b5_f1s, "CLKD-FT", "B5")
            d = cohens_d(ft_f1s, b5_f1s)
            comparisons["clkd_ft_vs_b5"] = {
                "f1": {"wilcoxon": w, "cohens_d": d,
                       "mean_ft": float(np.mean(ft_f1s)),
                       "mean_b5": float(np.mean(b5_f1s))}
            }
            print(f"\n  CLKD-FT vs B5 [F1]:")
            print(f"    CLKD-FT mean = {np.mean(ft_f1s):.4f} ± {np.std(ft_f1s):.4f}")
            print(f"    B5 mean      = {np.mean(b5_f1s):.4f} ± {np.std(b5_f1s):.4f}")
            print(f"    ΔF1 = {np.mean(ft_f1s) - np.mean(b5_f1s):+.4f}")
            print(f"    p = {w['p_value']:.4f} | dz = {d['d']:+.2f} ({d['magnitude']})")
            if w['significant']:
                print(f"    → Distillation shows measurable contribution")
            else:
                print(f"    → No significant difference — distillation may not add "
                      f"beyond fine-tuning alone")

            # B5 vs IndoBERT
            b2_f1s = [asr[i]["b2"]["f1"] for i in range(len(SEEDS))]
            w2 = wilcoxon_test(b2_f1s, b5_f1s, "IndoBERT", "B5")
            d2 = cohens_d(b2_f1s, b5_f1s)
            comparisons["indobert_vs_b5"] = {
                "f1": {"wilcoxon": w2, "cohens_d": d2,
                       "mean_b2": float(np.mean(b2_f1s)),
                       "mean_b5": float(np.mean(b5_f1s))}
            }
            print(f"\n  IndoBERT vs B5 [F1]:")
            print(f"    IndoBERT mean = {np.mean(b2_f1s):.4f} ± {np.std(b2_f1s):.4f}")
            print(f"    B5 mean       = {np.mean(b5_f1s):.4f} ± {np.std(b5_f1s):.4f}")
            print(f"    ΔF1 = {np.mean(b2_f1s) - np.mean(b5_f1s):+.4f}")
            print(f"    p = {w2['p_value']:.4f} | dz = {d2['d']:+.2f} ({d2['magnitude']})")

            # B5 vs B1 (XLM-R zero-shot): does ID fine-tuning help at all?
            b1_f1s = [asr[i]["b1"]["f1"] for i in range(len(SEEDS))]
            w3 = wilcoxon_test(b5_f1s, b1_f1s, "B5", "B1")
            d3 = cohens_d(b5_f1s, b1_f1s)
            comparisons["b5_vs_b1"] = {
                "f1": {"wilcoxon": w3, "cohens_d": d3,
                       "mean_b5": float(np.mean(b5_f1s)),
                       "mean_b1": float(np.mean(b1_f1s))}
            }
            print(f"\n  B5 vs B1 (XLM-R Zero-Shot) [F1]:")
            print(f"    B5 mean = {np.mean(b5_f1s):.4f} ± {np.std(b5_f1s):.4f}")
            print(f"    B1 mean = {np.mean(b1_f1s):.4f} ± {np.std(b1_f1s):.4f}")
            print(f"    ΔF1 = {np.mean(b5_f1s) - np.mean(b1_f1s):+.4f}")
            print(f"    p = {w3['p_value']:.4f} | dz = {d3['d']:+.2f} ({d3['magnitude']})")

        print(f"{'='*W}")
    else:
        print(f"\n  Note: {existing_path} not found — skipping comparisons.")
        print(f"  Place all_results.json in {config.RESULTS_DIR}/ to enable.")

    # ── Save results ─────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    output = {
        "experiment": "Experiment 1: No-Distillation Fine-Tuning Control (B5)",
        "description": (
            "XLM-R Base fine-tuned on 200 Indonesian samples "
            "(100 per class) with pure cross-entropy loss. "
            "No teacher model, no distillation components."
        ),
        "model": config.STUDENT_MODEL,
        "train_samples": config.ID_TRAIN_PER_CLASS * 2,
        "epochs": config.EPOCHS_FINETUNE,
        "learning_rate": config.LEARNING_RATE,
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "per_seed_runs": all_runs,
        "aggregated": agg,
        "statistical_comparisons": comparisons,
    }

    path = os.path.join(config.RESULTS_DIR, "experiment1_b5_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: x if not hasattr(x, "tolist") else x.tolist())

    elapsed = time.time() - start_time
    print(f"\n  Results saved → {path}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print_gpu_report()

    # ── Final verdict ────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  SUMMARY FOR ADVISOR")
    print(f"{'='*W}")
    print(f"  B5 (XLM-R + ID-FT, no distill): F1 = "
          f"{agg['f1']['mean']:.4f} ± {agg['f1']['std']:.4f}")
    if comparisons.get("clkd_ft_vs_b5"):
        c = comparisons["clkd_ft_vs_b5"]["f1"]
        delta = c["mean_ft"] - c["mean_b5"]
        p = c["wilcoxon"]["p_value"]
        print(f"  CLKD-FT: F1 = {c['mean_ft']:.4f}")
        print(f"  ΔF1 (CLKD-FT − B5) = {delta:+.4f}, p = {p:.4f}")
        if p < 0.05:
            print(f"\n  → POSITIVE: Distillation contributes beyond fine-tuning.")
            print(f"    Paper framing: method paper (distillation works).")
        else:
            print(f"\n  → NEUTRAL/NEGATIVE: No significant difference.")
            print(f"    Paper framing: empirical assessment (fine-tuning drives "
                  f"performance; distillation components show no measurable "
                  f"benefit at this sample size).")
    print(f"{'='*W}")


if __name__ == "__main__":
    main()
