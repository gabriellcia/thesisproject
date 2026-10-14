"""run_stability.py - Standalone Stability Test, 5 metrik per seed + mean±std."""
import json, os
import numpy as np
from torch.utils.data import DataLoader
import config
from utils import (
    prepare_datasets, FakeNewsDataset, evaluate_model, clean_gpu,
    print_gpu_report, wilcoxon_test, cohens_d,
)
from train_clkd import train_teacher, distill_student

METRIC_KEYS   = ["accuracy", "auc_roc", "sensitivity", "specificity", "f1"]
METRIC_LABELS = {
    "accuracy":    "Accuracy      ",
    "auc_roc":     "AUC-ROC       ",
    "sensitivity": "Sensitivity   ",
    "specificity": "Specificity   ",
    "f1":          "F1-Score (Mac)",
}


def main():
    clean_gpu()
    seeds   = config.STABILITY_SEEDS
    results = []

    print("=" * 70)
    print("  STABILITY TEST - Multiple Random Seeds (Zero-Shot)")
    print("=" * 70)

    for i, seed in enumerate(seeds):
        print(f"\n{'='*65}\n  RUN {i+1}/{len(seeds)} - Seed {seed}\n{'='*65}")
        data = prepare_datasets(random_state=seed); clean_gpu()
        teacher, tok_t = train_teacher(data["train_en"], seed=seed)
        student, tok_s, _ = distill_student(teacher, tok_t, data["train_en"],
                                            seed=seed, zs_mode=True)
        tl = DataLoader(FakeNewsDataset(data["test_id"].content, data["test_id"].label, tok_s),
                        batch_size=8)
        m = evaluate_model(student, tl, f"Seed {seed} (Zero-Shot ID)")
        results.append({"seed": seed, **m})
        del teacher, student; clean_gpu()

    # ── Per-seed table ─────────────────────────────────────────────
    W = 75
    print(f"\n{'='*W}")
    print(f" STABILITY RESULTS - Per Seed ({len(seeds)} seeds, Zero-Shot)")
    print(f"{'='*W}")
    print(f"{'Seed':>6} {'Accuracy':>9} {'AUC-ROC':>8} {'Sensitiv':>9} "
          f"{'Specif':>8} {'F1-Score':>9}")
    print(f"{'-'*W}")
    for r in results:
        print(f"{r['seed']:>6} {r['accuracy']:>9.4f} {r['auc_roc']:>8.4f} "
              f"{r['sensitivity']:>9.4f} {r['specificity']:>8.4f} {r['f1']:>9.4f}")
    print(f"{'='*W}")

    # ── Mean ± Std table ───────────────────────────────────────────
    agg_stats = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in results]
        agg_stats[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    print(f"\n STABILITY SUMMARY - Mean ± Std")
    print(f"{'-'*W}")
    print(f"{'Metrik':<16} {'Mean':>10} {'Std':>8} {'Min':>10} {'Max':>10}")
    print(f"{'-'*W}")
    for k in METRIC_KEYS:
        vals = [r[k] for r in results]
        print(f"{METRIC_LABELS[k]:<16} {agg_stats[k]['mean']:>10.4f} "
              f"{agg_stats[k]['std']:>8.4f} {min(vals):>10.4f} {max(vals):>10.4f}")
    print(f"{'='*W}")
    print(f" CLKD Zero-Shot stable: F1 = {agg_stats['f1']['mean']:.4f} ± {agg_stats['f1']['std']:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "stability_results.json")
    with open(path, "w") as f:
        json.dump({"runs": results, "stats": agg_stats}, f, indent=2)
    print(f" Saved -> {path}")
    print_gpu_report()


if __name__ == "__main__":
    main()
