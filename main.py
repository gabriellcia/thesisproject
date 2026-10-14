"""
main.py - Full SG-CLKD experiment (all fixes applied).
=========================================================
FIX #1: AUC sanity check (auto-flip if P(real) passed instead of P(fake))
FIX #2: Translate-Test baseline added
FIX #3: Multi-seed (5 seeds) for main table + ablation, Wilcoxon + Cohen's d
FIX #4: BoW diagnostic for source-blind leakage check
FIX #6: Confidence interval for recall (Wilson score)
FIX #7: Label smoothing option for teacher
FIX #8: Shared teacher between main run and ablation (no F1 mismatch)
"""

import json, os, random
import numpy as np, torch
from scipy import stats
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import XLMRobertaForSequenceClassification, AutoTokenizer

import config
from utils import (
    prepare_datasets, FakeNewsDataset, evaluate_model, clean_gpu,
    print_gpu_report, device,
    wilcoxon_test, cohens_d, recall_confidence_interval,
    full_statistical_comparison, bow_diagnostic,
)
from train_baselines import (
    train_baseline1, train_baseline2, train_baseline_translate,
    train_baseline_translate_test,
)
from train_clkd import train_teacher, distill_student, finetune_student

METRIC_KEYS   = ["accuracy", "auc_roc", "sensitivity", "specificity", "f1"]
METRIC_LABELS = {
    "accuracy":    "Accuracy      ",
    "auc_roc":     "AUC-ROC       ",
    "sensitivity": "Sensitivity   ",
    "specificity": "Specificity   ",
    "f1":          "F1-Score (Mac)",
}
SEEDS = getattr(config, "STABILITY_SEEDS", [42, 123, 999, 2026, 777])


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════
# FIX #3: Multi-Seed Run (main table + ablation)
# ══════════════════════════════════════════════════════════════

def run_single_seed(seed, run_ablation_flag=True, label_smoothing=0.0):
    """
    Run one complete experiment for a given seed.
    Returns all metrics for all models + ablation results.
    FIX #8: teacher is shared between main run and ablation.
    """
    print(f"\n{'#'*70}")
    print(f"  SEED = {seed}")
    print(f"{'#'*70}")
    set_seed(seed); clean_gpu()
    data = prepare_datasets(random_state=seed, verbose=(seed == SEEDS[0]))

    # ── Baselines ─────────────────────────────────────────────
    set_seed(seed)
    m_b1 = train_baseline1(data["train_en"], data["test_id"])
    clean_gpu()

    set_seed(seed)
    m_b2 = train_baseline2(data["train_id"], data["test_id"])
    clean_gpu()

    set_seed(seed)
    m_b3 = train_baseline_translate(data["train_en"], data["test_id"])
    clean_gpu()

    # ── Teacher (shared for CLKD + ablation + translate-test) ──
    set_seed(seed)
    teacher, tokenizer_t = train_teacher(
        data["train_en"], seed=seed, label_smoothing=label_smoothing
    )

    # ── FIX #2: Translate-Test baseline ───────────────────────
    set_seed(seed)
    m_b4 = train_baseline_translate_test(teacher, tokenizer_t, data["test_id"])

    # ── CLKD Zero-Shot ────────────────────────────────────────
    set_seed(seed)
    s_zs, tok_zs, pct = distill_student(
        teacher, tokenizer_t, data["train_en"], seed=seed, zs_mode=True
    )
    tl_zs = DataLoader(
        FakeNewsDataset(data["test_id"].content, data["test_id"].label, tok_zs),
        batch_size=8
    )
    m_zs = evaluate_model(s_zs, tl_zs, f"CLKD Zero-Shot (seed={seed})", verbose=False)
    del s_zs; clean_gpu()

    # ── CLKD Fine-Tuned ──────────────────────────────────────
    set_seed(seed)
    s_ft, tok_ft, _ = distill_student(
        teacher, tokenizer_t, data["train_en"], seed=seed, zs_mode=False
    )
    m_ft = finetune_student(s_ft, tok_ft, data["train_id"], data["test_id"])
    del s_ft; clean_gpu()

    # ── FIX #6: Confidence interval for recall ────────────────
    cm_ft = np.array(m_ft.get("cm", [[0,0],[0,0]]))
    if cm_ft.shape == (2, 2):
        n_fake_correct = cm_ft[1, 1]   # TP (fake correctly classified)
        n_fake_total = cm_ft[1, :].sum()  # total fake samples
        ci_low, ci_high = recall_confidence_interval(n_fake_correct, n_fake_total)
        m_ft["recall_fake_ci_95"] = [ci_low, ci_high]
        m_ft["n_fake_test"] = int(n_fake_total)
        print(f"   Recall(fake) CI 95%: [{ci_low:.4f}, {ci_high:.4f}] (n={n_fake_total})")

    # ── FIX #8: Ablation using SAME teacher ───────────────────
    ablation_results = []
    if run_ablation_flag:
        ablation_results = run_ablation_with_teacher(
            teacher, tokenizer_t, data, seed
        )

    del teacher; clean_gpu()

    result = {
        "seed": seed,
        "b1": m_b1, "b2": m_b2, "b3": m_b3, "b4": m_b4,
        "zs": m_zs, "ft": m_ft,
        "participation_pct": pct,
        "ablation": ablation_results,
    }
    return result, data


def run_ablation_with_teacher(teacher, tokenizer_t, data, seed):
    """
    FIX #8: Run ablation using the SAME teacher as main experiment.
    This eliminates the F1 mismatch between main table and ablation.
    """
    print(f"\n{'='*65}")
    print(f"  ABLATION (seed={seed}, shared teacher)")
    print(f"{'='*65}")

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

    results = [{"mode": "Baseline (No-Distill)", **m_nd}]

    for mode in config.ABLATION_MODES:
        set_seed(seed)
        s, tok_s, _ = distill_student(
            teacher, tokenizer_t, data["train_en"], mode=mode, seed=seed
        )
        tl = DataLoader(
            FakeNewsDataset(data["test_id"].content, data["test_id"].label, tok_s),
            batch_size=8
        )
        m = evaluate_model(s, tl, f"Ablation {mode}", verbose=False)
        LABEL_MAP = {
            "no_distill": "No-Distill", "logits_only": "Logits-Only",
            "contrast_only": "Contrast-Only", "sent_only": "Sentiment-Only",
            "logits_contrast": "Logits+Contrast", "logits_sent": "Logits+Sentiment",
            "contrast_sent": "Contrast+Sentiment", "full": "Full SG-CLKD",
            "features_only": "Features-Only",
        }
        label = LABEL_MAP.get(mode, mode)
        print(f"   -> {label}: F1={m['f1']:.4f} Acc={m['accuracy']:.4f}")
        results.append({"mode": label, **m})
        del s; clean_gpu()

    return results


# ══════════════════════════════════════════════════════════════
# SENSITIVITY (unchanged, but uses shared teacher from first seed)
# ══════════════════════════════════════════════════════════════

def run_sensitivity(teacher, tokenizer_t, data):
    print(f"\n{'='*68}")
    print(f"  SENSITIVITY ANALYSIS - Temperature T Sweep")
    print(f"{'='*68}")

    results = []
    for T in config.TEMPERATURE_CANDIDATES:
        print(f"\n   Testing T = {T}...")
        s, tok_s, _ = distill_student(
            teacher, tokenizer_t, data["train_en"],
            mode="full", temperature=T
        )
        tl = DataLoader(
            FakeNewsDataset(data["test_id"].content, data["test_id"].label, tok_s),
            batch_size=8,
        )
        m = evaluate_model(s, tl, f"T={T}", verbose=False)
        print(f"   -> T={T}: F1={m['f1']:.4f} Acc={m['accuracy']:.4f} "
              f"AUC={m['auc_roc']:.4f}")
        results.append({"temperature": T, **m})
        del s; clean_gpu()

    best = max(results, key=lambda x: x["f1"])
    print(f"\n   Optimal T* = {best['temperature']} "
          f"(F1={best['f1']:.4f}, Acc={best['accuracy']:.4f})")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "sensitivity_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: x if not hasattr(x, "tolist") else x.tolist())
    print(f"  Sensitivity saved -> {path}")
    return results


# ══════════════════════════════════════════════════════════════
# FIX #3: Statistical Tests — Wilcoxon + Cohen's d + t-test
# ══════════════════════════════════════════════════════════════

def run_statistical_tests(all_seed_results):
    """
    Run full statistical comparison across seeds:
    - Paired t-test
    - Wilcoxon signed-rank
    - Cohen's d effect size
    Key comparisons: Full vs No-Distill, Full vs Contrast-only,
    CLKD-FT vs B1, CLKD-ZS vs B1.
    """
    print(f"\n{'='*80}")
    print(f"  STATISTICAL SIGNIFICANCE TESTS ({len(all_seed_results)} seeds)")
    print(f"{'='*80}")

    # Extract per-seed metrics
    results_b1 = [r["b1"] for r in all_seed_results]
    results_zs = [r["zs"] for r in all_seed_results]
    results_ft = [r["ft"] for r in all_seed_results]

    comparisons = [
        ("CLKD-ZS vs B1",  results_zs, results_b1),
        ("CLKD-FT vs B1",  results_ft, results_b1),
        ("CLKD-FT vs ZS",  results_ft, results_zs),
    ]

    # Also compare ablation modes if available
    abl_comparisons = []
    # Gather ablation across seeds
    if all_seed_results[0].get("ablation"):
        n_modes = len(all_seed_results[0]["ablation"])
        for mode_idx in range(n_modes):
            mode_name = all_seed_results[0]["ablation"][mode_idx]["mode"]
            mode_results = []
            for r in all_seed_results:
                if r.get("ablation") and len(r["ablation"]) > mode_idx:
                    mode_results.append(r["ablation"][mode_idx])
            if mode_results and len(mode_results) == len(all_seed_results):
                abl_comparisons.append((mode_name, mode_results))

    # Find Full SG-CLKD and No-Distill for ablation comparisons
    full_results = None
    nodistill_results = None
    contrast_results = None
    for name, results in abl_comparisons:
        if "Full" in name:
            full_results = results
        if "No-Distill" in name:
            nodistill_results = results
        if "Contrast-Only" in name:
            contrast_results = results

    if full_results and nodistill_results:
        comparisons.append(("Full vs No-Distill (abl)", full_results, nodistill_results))
    if full_results and contrast_results:
        comparisons.append(("Full vs Contrast-Only (abl)", full_results, contrast_results))

    all_stats = {}
    W = 90
    # One-sided test: H1 = "first model outperforms second model"
    # Justified because we have directional hypotheses (FT > B1, etc.)
    alt = "greater"
    print(f"\n  Test: one-sided (H1: model_a > model_b), n={len(all_seed_results)} seeds")
    print(f"  Note: with n=5, one-sided sign test min p=0.03125 (significant)")
    print(f"        two-sided min p=0.0625 (never significant with n=5)\n")
    print(f"{'Comparison':<35} {'Metric':<12} {'Test':<18} {'p-value':>9} "
          f"{'Cohen d':>9} {'Effect':>10}")
    print(f"{'-'*W}")

    for label, res_a, res_b in comparisons:
        report = full_statistical_comparison(
            res_a, res_b, label.split(" vs ")[0],
            label.split(" vs ")[-1], alternative=alt
        )
        all_stats[label] = report
        for k in ["f1", "auc_roc"]:
            r = report[k]
            sig = "*" if r["wilcoxon"]["significant"] else ""
            test_name = r["wilcoxon"]["test"]
            print(f"{label:<35} {k:<12} "
                  f"{test_name:<18} "
                  f"{r['wilcoxon']['p_value']:>8.4f}{sig} "
                  f"{r['cohens_d']['d']:>+9.4f} "
                  f"{r['cohens_d']['magnitude']:>10}")
        print(f"{'-'*W}")

    print(f"{'='*W}")
    print(f"  * = p < 0.05 (statistically significant)")
    return all_stats


# ══════════════════════════════════════════════════════════════
# AGGREGATION: Mean ± Std across seeds
# ══════════════════════════════════════════════════════════════

def aggregate_seeds(all_seed_results, model_key):
    """Compute mean ± std for a model across seeds."""
    agg = {}
    for k in METRIC_KEYS:
        vals = [r[model_key][k] for r in all_seed_results
                if r.get(model_key) is not None]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        else:
            agg[k] = {"mean": 0.0, "std": 0.0}
    return agg


def aggregate_ablation(all_seed_results):
    """Compute mean ± std for each ablation mode across seeds."""
    if not all_seed_results[0].get("ablation"):
        return []

    n_modes = len(all_seed_results[0]["ablation"])
    agg_abl = []
    for mode_idx in range(n_modes):
        mode_name = all_seed_results[0]["ablation"][mode_idx]["mode"]
        mode_agg = {"mode": mode_name}
        for k in METRIC_KEYS:
            vals = []
            for r in all_seed_results:
                if r.get("ablation") and len(r["ablation"]) > mode_idx:
                    vals.append(r["ablation"][mode_idx][k])
            mode_agg[f"{k}_mean"] = float(np.mean(vals)) if vals else 0.0
            mode_agg[f"{k}_std"]  = float(np.std(vals)) if vals else 0.0
        agg_abl.append(mode_agg)
    return agg_abl


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    set_seed(SEEDS[0]); clean_gpu()

    # Label smoothing: set to 0.1 to address FIX #7, or 0.0 to leave as-is
    ls = getattr(config, "LABEL_SMOOTHING", 0.0)

    # ── Multi-seed runs (FIX #3) ─────────────────────────────
    all_seed_results = []
    first_data = None
    first_teacher = None
    first_tokenizer_t = None

    for i, seed in enumerate(SEEDS):
        # FIX #8: run ablation for ALL seeds inside run_single_seed
        # so teacher is always shared between main results and ablation
        result, data = run_single_seed(
            seed, run_ablation_flag=True, label_smoothing=ls
        )
        all_seed_results.append(result)
        if i == 0:
            first_data = data

    # ── Sensitivity (first seed only) ─────────────────────────
    set_seed(SEEDS[0]); clean_gpu()
    data_s = prepare_datasets(random_state=SEEDS[0], verbose=False)
    teacher_s, tok_t_s = train_teacher(
        data_s["train_en"], seed=SEEDS[0], label_smoothing=ls
    )
    sensitivity_results = run_sensitivity(teacher_s, tok_t_s, data_s)
    del teacher_s; clean_gpu()

    # ── Translate baselines (single-run, first seed, cached) ──
    # B3 and B4 already in all_seed_results

    # ── Statistical tests (FIX #3) ───────────────────────────
    stat_tests = run_statistical_tests(all_seed_results)

    # ── Aggregate results ─────────────────────────────────────
    agg_b1 = aggregate_seeds(all_seed_results, "b1")
    agg_b2 = aggregate_seeds(all_seed_results, "b2")
    agg_b3 = aggregate_seeds(all_seed_results, "b3")
    agg_b4 = aggregate_seeds(all_seed_results, "b4")
    agg_zs = aggregate_seeds(all_seed_results, "zs")
    agg_ft = aggregate_seeds(all_seed_results, "ft")
    agg_abl = aggregate_ablation(all_seed_results)

    # ── FIX (data-consistency): pick a representative seed for
    # ROC curve / confusion matrix, instead of always seed[0].
    # Otherwise the ROC curve can show a "lucky" single run while
    # the metrics table shows a very different multi-seed mean
    # (this happened with B3, which has high seed-to-seed variance).
    # We pick the seed whose FT-F1 is closest to the aggregated
    # mean FT-F1, since CLKD-FT is the primary reported result.
    ft_f1_mean = agg_ft["f1"]["mean"]
    rep_idx = min(
        range(len(all_seed_results)),
        key=lambda i: abs(all_seed_results[i]["ft"]["f1"] - ft_f1_mean),
    )
    rep_seed = SEEDS[rep_idx]
    print(f"\n  Representative seed for ROC/CM figures: seed={rep_seed} "
          f"(FT F1={all_seed_results[rep_idx]['ft']['f1']:.4f}, "
          f"closest to mean={ft_f1_mean:.4f})")
    print(f"  Note: main table uses {len(SEEDS)}-seed mean; "
          f"ROC curve/confusion matrix use this single representative run "
          f"since per-seed FPR/TPR curves cannot be simply averaged.")

    # ── Print main results table (mean ± std) ─────────────────
    W = 100
    print(f"\n{'='*W}")
    print(f"{'MAIN RESULTS TABLE (mean +/- std, n=' + str(len(SEEDS)) + ' seeds)':^{W}}")
    print(f"{'='*W}")
    print(f"{'Model':<42} | {'Accuracy':>14} | {'AUC-ROC':>14} | "
          f"{'Sensitiv':>14} | {'Specif':>14} | {'F1':>14}")
    print(f"{'-'*W}")

    def row(name, agg):
        parts = [f"{agg[k]['mean']:.4f}±{agg[k]['std']:.4f}" for k in METRIC_KEYS]
        print(f"{name:<42} | {parts[0]:>14} | {parts[1]:>14} | "
              f"{parts[2]:>14} | {parts[3]:>14} | {parts[4]:>14}")

    row("B1: XLM-R (Zero-Shot ID)",    agg_b1)
    row("B2: IndoBERT (Supervised ID)", agg_b2)
    row("B3: Translate-Train",          agg_b3)
    row("B4: Translate-Test (FIX #2)",  agg_b4)
    row("CLKD Zero-Shot",              agg_zs)
    row("CLKD Fine-Tuned",             agg_ft)
    print(f"{'='*W}")

    # ── Print ablation table (mean ± std) ─────────────────────
    if agg_abl:
        print(f"\n{'='*W}")
        print(f"{'ABLATION TABLE (mean +/- std, n=' + str(len(SEEDS)) + ' seeds)':^{W}}")
        print(f"{'='*W}")
        print(f"{'Mode':<28} | {'F1':>14} | {'AUC-ROC':>14} | "
              f"{'Accuracy':>14} | {'Sensitiv':>14} | {'Specif':>14}")
        print(f"{'-'*W}")
        for r in agg_abl:
            parts = [f"{r[f'{k}_mean']:.4f}±{r[f'{k}_std']:.4f}" for k in METRIC_KEYS]
            print(f"{r['mode']:<28} | {parts[4]:>14} | {parts[1]:>14} | "
                  f"{parts[0]:>14} | {parts[2]:>14} | {parts[3]:>14}")
        print(f"{'='*W}")

    # ── FIX #6: Recall CI for Fine-Tuned (representative seed) ─
    ft_first = all_seed_results[rep_idx]["ft"]
    if "recall_fake_ci_95" in ft_first:
        ci = ft_first["recall_fake_ci_95"]
        n = ft_first.get("n_fake_test", "?")
        print(f"\n  FIX #6: Recall(Fake) 95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]"
              f" (n_fake={n})")
        print(f"  -> Soften 'perfect recall' claim; report interval.")

    # ── Save JSON ─────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    all_results = {
        "seeds_used": SEEDS,
        "n_seeds": len(SEEDS),
        "representative_seed": rep_seed,
        "baselines": {
            "xlmr_zeroshot":       all_seed_results[rep_idx]["b1"],
            "indobert_supervised": all_seed_results[rep_idx]["b2"],
            "translate_train":     all_seed_results[rep_idx]["b3"],
            "translate_test":      all_seed_results[rep_idx]["b4"],
        },
        "clkd": {
            "zero_shot":  all_seed_results[rep_idx]["zs"],
            "fine_tuned": all_seed_results[rep_idx]["ft"],
        },
        "aggregated": {
            "b1": agg_b1, "b2": agg_b2, "b3": agg_b3, "b4": agg_b4,
            "zs": agg_zs, "ft": agg_ft,
        },
        # Backward-compat: visualize_metrics.py reads stability.zero_shot.stats
        "stability": {
            "zero_shot": {
                "stats": agg_zs,
                "runs":  [r["zs"] for r in all_seed_results],
            },
            "fine_tuned": {
                "stats": agg_ft,
                "runs":  [r["ft"] for r in all_seed_results],
            },
        },
        "ablation":           all_seed_results[rep_idx].get("ablation", []),
        "ablation_aggregated": agg_abl,
        "sensitivity_results": sensitivity_results,
        "statistical_tests":  stat_tests,
        "all_seed_runs":      all_seed_results,
        "participation_pct":  all_seed_results[0].get("participation_pct", 0),
        "label_smoothing":    ls,
    }

    path = os.path.join(config.RESULTS_DIR, "all_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: x if not hasattr(x, "tolist") else x.tolist())
    print(f"\n  All results saved -> {path}")
    print_gpu_report()


if __name__ == "__main__":
    main()
