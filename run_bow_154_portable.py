"""
run_bow_154.py — BoW probe on the 154-instance Indonesian test set
Exactly what Prof. Huang asked for in the email of 21 Sep 2026.
"""
import os, json, sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_ID_RAW = os.path.join(DATA_DIR, "test_indonesia_clean.xlsx")
TEST_ID_BLINDED = os.path.join(DATA_DIR, "test_indonesia_blinded.csv")


def load_indonesian_split(seed=42, use_blinded=True):
    """Reproduce utils.py prepare_datasets() 85/15 stratified split."""
    if use_blinded:
        df = pd.read_csv(TEST_ID_BLINDED, encoding="utf-8-sig")
        if "text_blinded" in df.columns:
            df["content"] = df["text_blinded"].fillna("")
    else:
        df = pd.read_excel(TEST_ID_RAW)

    label_map = {"real": 0, "fake": 1, 0: 0, 1: 1, "Real": 0, "Fake": 1}
    df["label"] = df["label"].map(label_map).astype(int)
    df = df.dropna(subset=["content", "label"]).reset_index(drop=True)

    # 85/15 split (matches utils.prepare_datasets test_size=0.15)
    train_id, test_id_154 = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=seed
    )
    return train_id, test_id_154


def run_bow(train_texts, train_labels, test_texts, test_labels, name):
    pipe = Pipeline([
        ("bow", CountVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000, random_state=42,
                                    class_weight="balanced")),
    ])
    pipe.fit(train_texts, train_labels)
    preds = pipe.predict(test_texts)
    probs = pipe.predict_proba(test_texts)[:, 1]

    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(test_labels, probs)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(test_labels, preds)

    print(f"\n{'='*70}\n {name}\n{'='*70}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-macro : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Confusion matrix:")
    print(f"                Predicted Real  Predicted Fake")
    print(f"  Actual Real   {cm[0,0]:>13d}   {cm[0,1]:>13d}")
    print(f"  Actual Fake   {cm[1,0]:>13d}   {cm[1,1]:>13d}")
    print(f"\n{classification_report(test_labels, preds, target_names=['Real (0)', 'Fake (1)'], zero_division=0)}")

    # Top discriminative features
    bow = pipe.named_steps["bow"]
    clf = pipe.named_steps["clf"]
    names = bow.get_feature_names_out()
    coefs = clf.coef_[0]
    top_fake = np.argsort(coefs)[-20:][::-1]
    top_real = np.argsort(coefs)[:20]
    top_fake_feats = [(names[i], float(coefs[i])) for i in top_fake]
    top_real_feats = [(names[i], float(coefs[i])) for i in top_real]

    print("\n  Top 20 features → FAKE:")
    for feat, c in top_fake_feats:
        print(f"    {feat:35s}  coef={c:+.4f}")
    print("\n  Top 20 features → REAL:")
    for feat, c in top_real_feats:
        print(f"    {feat:35s}  coef={c:+.4f}")

    return {
        "name": name, "n_train": len(train_texts), "n_test": len(test_texts),
        "accuracy": acc, "f1_macro": f1, "auc_roc": auc,
        "confusion_matrix": cm.tolist(),
        "top_fake_features": top_fake_feats,
        "top_real_features": top_real_feats,
    }


def interpret_c1(f1_blinded):
    """Reviewer C-1 interpretation table."""
    if f1_blinded >= 0.90:
        return ("≥ 0.90", "Most of the signal is publisher style. §5.6 becomes "
                "a methodological warning; transformer's margin over BoW is "
                "the real contribution.")
    elif f1_blinded >= 0.75:
        return ("0.75–0.85 (partial shortcut)",
                f"Requires quantified discussion in §6, but the main "
                f"conclusions survive. BoW captures {f1_blinded:.1%} of the "
                f"signal from lexical shortcuts alone.")
    elif f1_blinded <= 0.60:
        return ("≤ 0.60 (relatively clean)",
                "Indonesian corpus is relatively clean; single largest threat "
                "to validity is eliminated. §5.6 turns from a weakness into a "
                "strength.")
    else:
        return (f"{f1_blinded:.3f} (between thresholds)",
                "Report the actual number without heavy claim; describe as a "
                "moderate but real lexical shortcut.")


def main():
    print("=" * 70)
    print(" BoW + Logistic Regression probe — 154-instance Indonesian test")
    print(" (Prof. Huang email, 21 Sep 2026, Item 3)")
    print(" (Reviewer C-1, Section 3, Priority 1)")
    print("=" * 70)

    results = {}

    # ─── Condition 1: Blinded (post-preprocessing) ───────────
    print("\n▶ CONDITION 1: Blinded Indonesian (train ID → test ID 154)")
    train_bl, test_bl = load_indonesian_split(seed=42, use_blinded=True)
    print(f"  Train: {len(train_bl)} samples "
          f"(real={(train_bl.label==0).sum()}, fake={(train_bl.label==1).sum()})")
    print(f"  Test : {len(test_bl)} samples "
          f"(real={(test_bl.label==0).sum()}, fake={(test_bl.label==1).sum()})")
    results["blinded"] = run_bow(
        train_bl.content.tolist(), train_bl.label.tolist(),
        test_bl.content.tolist(), test_bl.label.tolist(),
        f"Blinded Indonesian (train={len(train_bl)} → test={len(test_bl)})",
    )

    # ─── Condition 2: Raw (pre-preprocessing) ───────────────
    print("\n▶ CONDITION 2: Raw Indonesian (train ID → test ID 154, no blinding)")
    train_raw, test_raw = load_indonesian_split(seed=42, use_blinded=False)
    print(f"  Train: {len(train_raw)} samples")
    print(f"  Test : {len(test_raw)} samples")
    results["raw"] = run_bow(
        train_raw.content.tolist(), train_raw.label.tolist(),
        test_raw.content.tolist(), test_raw.label.tolist(),
        f"Raw Indonesian (train={len(train_raw)} → test={len(test_raw)})",
    )

    # ─── Interpretation ──────────────────────────────────────
    f1_bl = results["blinded"]["f1_macro"]
    f1_raw = results["raw"]["f1_macro"]

    print("\n" + "=" * 70)
    print("  SUMMARY (as Prof. Huang requested)")
    print("=" * 70)
    print(f"\n  Blinded ID → 154 test : F1={f1_bl:.4f}  AUC={results['blinded']['auc_roc']:.4f}  Acc={results['blinded']['accuracy']:.4f}")
    print(f"  Raw ID     → 154 test : F1={f1_raw:.4f}  AUC={results['raw']['auc_roc']:.4f}  Acc={results['raw']['accuracy']:.4f}")
    print(f"  Blinding delta        : ΔF1={f1_bl-f1_raw:+.4f}  (negative = blinding removed some signal)")

    bucket, interp = interpret_c1(f1_bl)
    print(f"\n  REVIEWER C-1 BUCKET: {bucket}")
    print(f"\n  INTERPRETATION:\n    {interp}")

    # Save
    out = os.path.join(RESULTS_DIR, "bow_probe_154instance.json")
    with open(out, "w") as f:
        json.dump({
            "purpose": "BoW probe on 154-instance ID test set as requested by Prof. Huang",
            "conditions": results,
            "summary": {
                "blinded_f1": f1_bl,
                "raw_f1": f1_raw,
                "blinding_delta_f1": f1_bl - f1_raw,
                "reviewer_c1_bucket": bucket,
                "interpretation": interp,
            }
        }, f, indent=2)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
