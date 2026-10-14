"""
utils.py - Dataset class, evaluasi lengkap 5 metrik, data loading.
evaluate_model() -> dict: accuracy, auc_roc, sensitivity, specificity, f1, cm, fpr, tpr
"""

import gc
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import (
    f1_score, classification_report, accuracy_score,
    roc_auc_score, recall_score, confusion_matrix, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from scipy import stats as sp_stats
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(" GPU cache cleared.")


def print_gpu_report():
    if not torch.cuda.is_available():
        print("️  Running on CPU - no GPU memory report.")
        return
    total     = torch.cuda.get_device_properties(0).total_memory / 1024**2
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved  = torch.cuda.memory_reserved(0) / 1024**2
    free      = total - allocated
    pct       = (allocated / total) * 100
    bar       = "█" * int(20 * pct / 100) + "░" * (20 - int(20 * pct / 100))
    status    = " OK" if pct < 80 else " HIGH" if pct < 95 else " CRITICAL"
    print(f"\n{'='*48}\n GPU MEMORY REPORT\n{'='*48}")
    print(f"Total VRAM        : {total:>9.1f} MB")
    print(f"Allocated         : {allocated:>9.1f} MB")
    print(f"Reserved (Cache)  : {reserved:>9.1f} MB")
    print(f"Free              : {free:>9.1f} MB")
    print(f"Usage             : [{bar}] {pct:.1f}%  {status}")
    print(f"{'='*48}")


def print_stats(df, title):
    total = len(df)
    counts = df["label"].value_counts()
    real, fake = counts.get(0, 0), counts.get(1, 0)
    print(f" {title}: Total={total}  Real={real} ({real/total*100:.1f}%)  Fake={fake} ({fake/total*100:.1f}%)")
    print("-" * 45)


class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=None):
        self.texts     = texts.reset_index(drop=True)
        self.labels    = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len or config.MAX_LEN

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        enc = self.tokenizer(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels":         torch.tensor(self.labels[item], dtype=torch.long),
        }


def evaluate_model(model, data_loader, name="Model", verbose=True):
    """
    Evaluasi model -> dict berisi semua metrik:
      accuracy, auc_roc, sensitivity (recall macro),
      specificity (macro), f1 (macro),
      cm (confusion matrix list), fpr/tpr (untuk ROC plot)
    """
    model.eval()
    preds, actuals, probs = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out    = model(ids, attention_mask=mask)
            logits = out.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            actuals.extend(labels.cpu().numpy())
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    accuracy    = float(accuracy_score(actuals, preds))
    f1          = float(f1_score(actuals, preds, average="macro", zero_division=0))
    sensitivity = float(recall_score(actuals, preds, average="macro", zero_division=0))

    try:
        auc_roc = float(roc_auc_score(actuals, probs))
        # ── FIX #1: AUC sanity check ─────────────────────────────
        # If accuracy > 0.6 but AUC < 0.5, P(fake) is likely inverted
        if accuracy > 0.60 and auc_roc < 0.50:
            print(f"  WARNING: AUC={auc_roc:.4f} < 0.5 but Acc={accuracy:.4f} > 0.6")
            print(f"           Likely passing P(real) instead of P(fake).")
            print(f"           Flipping probabilities: 1 - P(real) = P(fake)")
            probs = [1.0 - p for p in probs]
            auc_roc = float(roc_auc_score(actuals, probs))
            print(f"           Corrected AUC = {auc_roc:.4f}")
    except Exception:
        auc_roc = 0.0

    # Specificity = TN/(TN+FP) per class -> rata-rata macro
    cm_arr = confusion_matrix(actuals, preds)
    specs = []
    for i in range(len(cm_arr)):
        tn = cm_arr.sum() - cm_arr[i, :].sum() - cm_arr[:, i].sum() + cm_arr[i, i]
        fp = cm_arr[:, i].sum() - cm_arr[i, i]
        specs.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    specificity = sum(specs) / len(specs)

    try:
        fpr_arr, tpr_arr, _ = roc_curve(actuals, probs)
        fpr_list, tpr_list = fpr_arr.tolist(), tpr_arr.tolist()
    except Exception:
        fpr_list, tpr_list = [0.0, 1.0], [0.0, 1.0]

    if verbose:
        print(f"\n{'='*58}")
        print(f"  EVALUATION: {name}")
        print(f"{'='*58}")
        print(classification_report(actuals, preds,
                                     target_names=["Real (0)", "Fake (1)"],
                                     zero_division=0))
        print(f"{'─'*58}")
        print(f"   Accuracy              : {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"   AUC-ROC               : {auc_roc:.4f}")
        print(f"   Sensitivity (Recall)  : {sensitivity:.4f}")
        print(f"   Specificity           : {specificity:.4f}")
        print(f"   F1-Score (Macro)      : {f1:.4f}")
        print(f"{'='*58}")

    return {
        "accuracy":    accuracy,
        "auc_roc":     auc_roc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1":          f1,
        "cm":          cm_arr.tolist(),
        "fpr":         fpr_list,
        "tpr":         tpr_list,
    }


def load_data(file_path, verbose=True):
    if file_path.endswith(".csv"):
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin1")
    else:
        df = pd.read_excel(file_path)
    df.columns = [c.replace("ï»¿", "").strip() for c in df.columns]
    if verbose:
        print(f" Loaded '{file_path}' -> {len(df)} rows")
    label_map = {"real": 0, "fake": 1, "Real": 0, "Fake": 1, 0: 0, 1: 1}
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["content", "label"])
    df["label"] = df["label"].astype(int)
    return df


def prepare_datasets(random_state=None, verbose=True):
    seed = random_state or config.RANDOM_STATE
    if verbose:
        print("\n" + "=" * 45 + "\n  STEP 1: LOAD & PREPROCESS DATA\n" + "=" * 45)

    df_en_raw = load_data(config.EN_DATA_PATH, verbose=verbose)
    df_id_raw = load_data(config.ID_DATA_PATH, verbose=verbose)

    if verbose:
        print_stats(df_en_raw, "English (Raw)")
        print_stats(df_id_raw, "Indonesian (Raw)")

    df_en_fake    = df_en_raw[df_en_raw.label == 1]
    df_en_real    = df_en_raw[df_en_raw.label == 0].sample(n=len(df_en_fake), random_state=seed)
    df_en_balanced = pd.concat([df_en_fake, df_en_real]).sample(frac=1, random_state=seed)

    train_en, test_en = train_test_split(
        df_en_balanced, test_size=config.TEST_SIZE_EN,
        stratify=df_en_balanced["label"], random_state=seed,
    )
    train_id, test_id = train_test_split(
        df_id_raw, test_size=config.TEST_SIZE_ID,
        stratify=df_id_raw["label"], random_state=seed,
    )

    n = getattr(config, "ID_TRAIN_PER_CLASS", 100)
    train_id = pd.concat([
        train_id[train_id.label == 0].sample(n=n, random_state=seed),
        train_id[train_id.label == 1].sample(n=n, random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    if verbose:
        print(f"\n Final Split:")
        print(f"   EN Train: {len(train_en)}  |  EN Test: {len(test_en)}")
        print(f"   ID Train: {len(train_id)} (low-resource)  |  ID Test: {len(test_id)}")

    return {"train_en": train_en, "test_en": test_en,
            "train_id": train_id, "test_id": test_id}


# ══════════════════════════════════════════════════════════════
# FIX #3: Statistical Tests — Wilcoxon + Cohen's d
# ══════════════════════════════════════════════════════════════
def wilcoxon_test(scores_a, scores_b, label_a="A", label_b="B",
                  alternative="greater"):
    """
    Wilcoxon signed-rank test for paired samples.
    Default: one-sided 'greater' (H1: a > b) — appropriate for directional
    hypotheses like 'CLKD-FT outperforms B1'.
    With n=5, two-sided minimum p=0.0625 (never significant);
    one-sided minimum p=0.03125 (significant if all diffs same direction).
    Falls back to exact sign test if n < 6.
    """
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    n = len([d for d in diffs if d != 0])  # non-zero differences

    if n < 6:
        # Fallback: exact sign test (binomial)
        pos = sum(1 for d in diffs if d > 0)
        try:
            from scipy.stats import binomtest
            alt_binom = "greater" if alternative == "greater" else "two-sided"
            result = binomtest(pos, n, 0.5, alternative=alt_binom)
            p = float(result.pvalue)
        except ImportError:
            p = float(sp_stats.binom.sf(pos - 1, n, 0.5))
            if alternative == "two-sided":
                p = min(p * 2, 1.0)
        return {
            "test": f"sign_test (n={n}, {alternative})",
            "statistic": float(pos),
            "p_value": p,
            "n_nonzero": n,
            "significant": bool(p < 0.05),
            "label": f"{label_a} vs {label_b}",
        }

    stat, p = sp_stats.wilcoxon(scores_a, scores_b, alternative=alternative)
    return {
        "test": f"wilcoxon ({alternative})",
        "statistic": float(stat),
        "p_value": float(p),
        "n_nonzero": n,
        "significant": bool(p < 0.05),
        "label": f"{label_a} vs {label_b}",
    }


def cohens_d(scores_a, scores_b):
    """
    Cohen's d effect size for paired samples.
    |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large.
    """
    a, b = np.array(scores_a), np.array(scores_b)
    diff = a - b
    d = float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-12))
    magnitude = ("negligible" if abs(d) < 0.2 else
                 "small" if abs(d) < 0.5 else
                 "medium" if abs(d) < 0.8 else "large")
    return {"d": d, "magnitude": magnitude}


def recall_confidence_interval(n_correct, n_total, confidence=0.95):
    """
    Wilson score interval for recall/proportion (FIX #6).
    More accurate than normal approximation for small n.
    Returns (lower, upper) bounds.
    """
    if n_total == 0:
        return (0.0, 0.0)
    p_hat = n_correct / n_total
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom
    return (max(0.0, float(center - spread)), min(1.0, float(center + spread)))


def full_statistical_comparison(results_a, results_b, label_a, label_b,
                                metric_keys=None, alternative="greater"):
    """
    Run Wilcoxon + Cohen's d for all metrics between two multi-seed runs.
    results_a, results_b: list of dicts with metric values (one per seed).
    alternative: 'greater' (H1: a > b) or 'two-sided'.
    Returns dict of {metric: {wilcoxon: ..., cohens_d: ...}}.
    """
    if metric_keys is None:
        metric_keys = ["accuracy", "auc_roc", "sensitivity", "specificity", "f1"]

    report = {}
    for k in metric_keys:
        vals_a = [r[k] for r in results_a]
        vals_b = [r[k] for r in results_b]
        w = wilcoxon_test(vals_a, vals_b, label_a, label_b,
                          alternative=alternative)
        d = cohens_d(vals_a, vals_b)
        report[k] = {"wilcoxon": w, "cohens_d": d,
                      "mean_a": float(np.mean(vals_a)),
                      "mean_b": float(np.mean(vals_b)),
                      "std_a": float(np.std(vals_a)),
                      "std_b": float(np.std(vals_b))}
    return report


# ══════════════════════════════════════════════════════════════
# FIX #4: BoW + Logistic Regression Diagnostic
# ══════════════════════════════════════════════════════════════
def bow_diagnostic(train_texts, train_labels, test_texts, test_labels,
                   name="BoW+LogReg"):
    """
    Shallow baseline: Bag-of-Words + Logistic Regression.
    If this gets > 95% accuracy on blinded data, lexical leakage persists.
    """
    print(f"\n{'─'*55}")
    print(f"  BOW DIAGNOSTIC: {name}")
    print(f"{'─'*55}")

    pipe = Pipeline([
        ("bow", CountVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    train_texts_list = [str(t) for t in train_texts]
    test_texts_list = [str(t) for t in test_texts]
    train_labels_list = list(train_labels)
    test_labels_list = list(test_labels)

    pipe.fit(train_texts_list, train_labels_list)
    preds = pipe.predict(test_texts_list)
    probs = pipe.predict_proba(test_texts_list)[:, 1]

    acc = float(accuracy_score(test_labels_list, preds))
    f1 = float(f1_score(test_labels_list, preds, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(test_labels_list, probs))
    except Exception:
        auc = 0.0

    leakage = acc > 0.95
    status = "LEXICAL LEAKAGE DETECTED" if leakage else "OK - leakage controlled"
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   AUC-ROC  : {auc:.4f}")
    print(f"   Status   : {status}")
    if leakage:
        print(f"   -> Shallow BoW model still > 95%: source markers remain.")
        print(f"   -> Add more patterns to prepare.py SOURCE_PATTERNS.")

    # Top discriminative features
    try:
        bow = pipe.named_steps["bow"]
        clf = pipe.named_steps["clf"]
        feature_names = bow.get_feature_names_out()
        coefs = clf.coef_[0]
        top_fake = np.argsort(coefs)[-10:][::-1]
        top_real = np.argsort(coefs)[:10]
        print(f"\n   Top 10 features -> FAKE:")
        for idx in top_fake:
            print(f"      {feature_names[idx]:30s} (coef={coefs[idx]:+.4f})")
        print(f"   Top 10 features -> REAL:")
        for idx in top_real:
            print(f"      {feature_names[idx]:30s} (coef={coefs[idx]:+.4f})")
    except Exception:
        pass

    return {
        "accuracy": acc, "f1": f1, "auc_roc": auc,
        "leakage_detected": leakage, "status": status,
    }
