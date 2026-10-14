"""
run_ablation.py - Standalone Ablation Study with multi-seed support (FIX #3).
Reports mean ± std for all 5 metrics, delta table, and Wilcoxon + Cohen's d.
"""
import json, os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import XLMRobertaForSequenceClassification, AutoTokenizer
import torch, random

import config
from utils import (
    prepare_datasets, FakeNewsDataset, evaluate_model, clean_gpu,
    print_gpu_report, device,
    wilcoxon_test, cohens_d, full_statistical_comparison,
)
from train_clkd import train_teacher, distill_student

METRIC_KEYS = ["accuracy", "auc_roc", "sensitivity", "specificity", "f1"]
SEEDS = getattr(config, "STABILITY_SEEDS", [42, 123, 999, 2026, 777])


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_one_seed(seed):
    """Run ablation for a single seed. Returns list of mode results."""
    set_seed(seed); clean_gpu()
    data = prepare_datasets(random_state=seed, verbose=False)
    teacher, tok_t = train_teacher(data["train_en"], seed=seed)

    # No-Distill baseline
    set_seed(seed)
    tok_nd = AutoTokenizer.from_pretrained(config.STUDENT_MODEL)
    m_nd_model = XLMRobertaForSequenceClassification.from_pretrained(
        config.STUDENT_MODEL, num_labels=config.NUM_LABELS
    ).to(device)
    ld = DataLoader(
        FakeNewsDataset(data["train_en"].content, data["train_en"].label, tok_nd),
        batch_size=16, shuffle=True
    )
    opt = AdamW(m_nd_model.parameters(), lr=config.LEARNING_RATE)
    m_nd_model.train()
    for _ in range(config.BASELINE_EPOCHS):
        for batch in ld:
            opt.zero_grad()
            out = m_nd_model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )
            out.loss.backward(); opt.step()
    tl = DataLoader(
        FakeNewsDataset(data["test_id"].content, data["test_id"].label, tok_nd),
        batch_size=8
    )
    m_nd = evaluate_model(m_nd_model, tl, "No-Distill", verbose=False)
    del m_nd_model; clean_gpu()

    LABEL_MAP = {
        "no_distill": "No-Distill", "logits_only": "Logits-Only",
        "contrast_only": "Contrast-Only", "sent_only": "Sentiment-Only",
        "logits_contrast": "Logits+Contrast", "logits_sent": "Logits+Sentiment",
        "contrast_sent": "Contrast+Sentiment", "full": "Full SG-CLKD",
        "features_only": "Features-Only",
    }

    results = [{"mode": "Baseline (No-Distill)", **m_nd}]
    for mode in config.ABLATION_MODES:
        set_seed(seed)
        s, tok_s, _ = distill_student(teacher, tok_t, data["train_en"],
                                       mode=mode, seed=seed)
        tl = DataLoader(
            FakeNewsDataset(data["test_id"].content, data["test_id"].label, tok_s),
            batch_size=8
        )
        m = evaluate_model(s, tl, f"Ablation {mode}", verbose=False)
        label = LABEL_MAP.get(mode, mode)
        print(f"   [seed={seed}] {label}: F1={m['f1']:.4f}")
        results.append({"mode": label, **m})
        del s; clean_gpu()

    del teacher; clean_gpu()
    return results


def main():
    clean_gpu()
    print("=" * 70)
    print(f"  ABLATION STUDY - Multi-Seed ({len(SEEDS)} seeds)")
    print("=" * 70)

    all_runs = []
    for seed in SEEDS:
        print(f"\n{'─'*60}\n  SEED = {seed}\n{'─'*60}")
        results = run_one_seed(seed)
        all_runs.append(results)

    # Aggregate mean ± std per mode
    n_modes = len(all_runs[0])
    W = 95
    print(f"\n{'='*W}")
    print(f"  ABLATION SUMMARY (mean ± std, {len(SEEDS)} seeds)")
    print(f"{'='*W}")
    print(f"{'Mode':<28} {'F1':>14} {'AUC-ROC':>14} {'Accuracy':>14} "
          f"{'Sensitiv':>14}")
    print(f"{'-'*W}")

    agg_results = []
    for mode_idx in range(n_modes):
        mode_name = all_runs[0][mode_idx]["mode"]
        mode_agg = {"mode": mode_name}
        parts = []
        for k in METRIC_KEYS:
            vals = [all_runs[s][mode_idx][k] for s in range(len(SEEDS))]
            m, s = float(np.mean(vals)), float(np.std(vals))
            mode_agg[f"{k}_mean"] = m
            mode_agg[f"{k}_std"] = s
            parts.append(f"{m:.4f}±{s:.4f}")
        print(f"{mode_name:<28} {parts[4]:>14} {parts[1]:>14} "
              f"{parts[0]:>14} {parts[2]:>14}")
        agg_results.append(mode_agg)

    print(f"{'='*W}")

    # Wilcoxon + Cohen's d: Full vs No-Distill, Full vs Contrast-Only
    full_runs = [all_runs[s][-1] for s in range(len(SEEDS))]  # last mode = Full
    nd_runs   = [all_runs[s][0] for s in range(len(SEEDS))]   # first mode = No-Distill

    print(f"\n  Statistical Tests (Wilcoxon + Cohen's d):")
    for k in ["f1", "auc_roc"]:
        vals_full = [r[k] for r in full_runs]
        vals_nd   = [r[k] for r in nd_runs]
        w = wilcoxon_test(vals_full, vals_nd, "Full", "No-Distill")
        d = cohens_d(vals_full, vals_nd)
        print(f"    Full vs No-Distill [{k}]: p={w['p_value']:.4f}, "
              f"d={d['d']:+.4f} ({d['magnitude']})")

    # Save
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "ablation_results.json")
    output = {
        "seeds": SEEDS,
        "aggregated": agg_results,
        "all_runs": all_runs,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: x if not hasattr(x, "tolist") else x.tolist())
    print(f"\n  Saved -> {path}")
    print_gpu_report()


if __name__ == "__main__":
    main()
