"""
run_source_blind.py
===================
C1 - Source-Blind Preprocessing & Rerun 5 Baselines

BEFORE results diambil dari hasil nyata thesis (result_.pdf):
  B1  XLM-R ZS      : Acc=0.7792  AUC=0.9439  Sen=0.8333  Spe=0.8333  F1=0.7632
  B2  IndoBERT       : Acc=0.9935  AUC=0.9998  Sen=0.9881  Spe=0.9881  F1=0.9918
  B3  Translate-Train: Acc=0.8766  AUC=0.9341  Sen=0.8259  Spe=0.8259  F1=0.8383
  B4  CLKD-ZS        : Acc=0.8506  AUC=0.9120  Sen=0.8229  Spe=0.8229  F1=0.8158
  B5  CLKD-FT        : Acc=0.9935  AUC=0.9970  Sen=0.9881  Spe=0.9881  F1=0.9918

Alur kerja:
  1. Load data blinded dari data/prepare.py
  2. Rerun 5 konfigurasi model dengan data blinded
  3. Cetak tabel before vs after + delta
  4. Simpan ke results/source_blind_comparison.json
              results/source_blind_report.txt

Cara pakai:
  python data/prepare.py       # strip source markers dulu
  python run_source_blind.py   # rerun + dokumentasi
"""

import json, os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config
from utils import FakeNewsDataset, evaluate_model, clean_gpu, device, bow_diagnostic
from train_baselines import train_baseline1, train_baseline2, train_baseline_translate
from train_clkd import train_teacher, distill_student, finetune_student

os.makedirs(config.RESULTS_DIR, exist_ok=True)

METRIC_KEYS   = ["accuracy", "auc_roc", "sensitivity", "specificity", "f1"]
METRIC_LABELS = ["Accuracy", "AUC-ROC", "Sensitivity", "Specificity", "F1-Score"]

# ── Hasil BEFORE dari result_.pdf (Table 4.1, 4.3, 4.4) ───────
BEFORE_RESULTS = {
    "B1: XLM-R ZS"       : {"accuracy": 0.7792, "auc_roc": 0.9439,
                             "sensitivity": 0.8333, "specificity": 0.8333, "f1": 0.7632},
    "B2: IndoBERT"        : {"accuracy": 0.9935, "auc_roc": 0.9998,
                             "sensitivity": 0.9881, "specificity": 0.9881, "f1": 0.9918},
    "B3: Translate-Train" : {"accuracy": 0.8766, "auc_roc": 0.9341,
                             "sensitivity": 0.8259, "specificity": 0.8259, "f1": 0.8383},
    "B4: CLKD-ZS"         : {"accuracy": 0.8506, "auc_roc": 0.9120,
                             "sensitivity": 0.8229, "specificity": 0.8229, "f1": 0.8158},
    "B5: CLKD-FT"         : {"accuracy": 0.9935, "auc_roc": 0.9970,
                             "sensitivity": 0.9881, "specificity": 0.9881, "f1": 0.9918},
}

MODEL_ORDER = list(BEFORE_RESULTS.keys())


# ─────────────────────────────────────────────────────────────
# STEP 1 - Load & siapkan data blinded
# ─────────────────────────────────────────────────────────────
def load_blinded_splits(seed: int = config.RANDOM_STATE):
    """
    Load CSV blinded dari data/prepare.py.
    Split identik dengan prepare_datasets() di utils.py.
    """
    train_path = "data/train_english_blinded.csv"
    test_path  = "data/test_indonesia_blinded.csv"

    for p in [train_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"'{p}' not found.\n"
                "Run first: python data/prepare.py"
            )

    df_train = pd.read_csv(train_path, encoding="utf-8-sig")
    df_test  = pd.read_csv(test_path,  encoding="utf-8-sig")

    # Pakai text_blinded sebagai content
    df_train["content"] = df_train["text_blinded"].fillna("")
    df_test["content"]  = df_test["text_blinded"].fillna("")

    # Normalise label -> int
    label_map = {"real": 0, "fake": 1, 0: 0, 1: 1}
    df_train["label"] = df_train["label"].map(label_map).astype(int)
    df_test["label"]  = df_test["label"].map(label_map).astype(int)

    df_train = df_train.dropna(subset=["content", "label"]).reset_index(drop=True)
    df_test  = df_test.dropna(subset=["content", "label"]).reset_index(drop=True)

    print(f"  Train blinded : {len(df_train):,}  "
          f"(real={(df_train['label']==0).sum()} "
          f"fake={(df_train['label']==1).sum()})")
    print(f"  Test  blinded : {len(df_test):,}  "
          f"(real={(df_test['label']==0).sum()} "
          f"fake={(df_test['label']==1).sum()})")

    # English split - balance fake/real lalu split
    df_fake = df_train[df_train.label == 1]
    df_real = df_train[df_train.label == 0].sample(
        n=len(df_fake), random_state=seed
    )
    df_en = pd.concat([df_fake, df_real]).sample(frac=1, random_state=seed)
    train_en, _ = train_test_split(
        df_en, test_size=config.TEST_SIZE_EN,
        stratify=df_en["label"], random_state=seed,
    )

    # Indonesian train - 100 per class (low-resource)
    n   = config.ID_TRAIN_PER_CLASS
    id0 = df_test[df_test.label == 0].sample(
        n=min(n, (df_test.label == 0).sum()), random_state=seed
    )
    id1 = df_test[df_test.label == 1].sample(
        n=min(n, (df_test.label == 1).sum()), random_state=seed
    )
    train_id = (pd.concat([id0, id1])
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True))

    test_id = df_test.copy().reset_index(drop=True)

    print(f"  EN  train split : {len(train_en)}")
    print(f"  ID  train split : {len(train_id)} (100 per class)")
    print(f"  ID  test  split : {len(test_id)}")
    return train_en, train_id, test_id


# ─────────────────────────────────────────────────────────────
# STEP 2 - Rerun 5 konfigurasi
# ─────────────────────────────────────────────────────────────
def run_all_models(train_en, train_id, test_id) -> dict:
    after = {}

    # B1: XLM-R Zero-Shot
    print(f"\n{'─'*55}\n  B1: XLM-R Zero-Shot (blinded)\n{'─'*55}")
    try:
        after["B1: XLM-R ZS"] = train_baseline1(train_en, test_id)
        clean_gpu()
    except Exception as e:
        print(f"  ️  B1 error: {e}"); clean_gpu()

    # B2: IndoBERT Supervised
    print(f"\n{'─'*55}\n  B2: IndoBERT Supervised (blinded)\n{'─'*55}")
    try:
        after["B2: IndoBERT"] = train_baseline2(train_id, test_id)
        clean_gpu()
    except Exception as e:
        print(f"  ️  B2 error: {e}"); clean_gpu()

    # B3: Translate-Train
    print(f"\n{'─'*55}\n  B3: Translate-Train (blinded)\n{'─'*55}")
    try:
        after["B3: Translate-Train"] = train_baseline_translate(train_en, test_id)
        clean_gpu()
    except Exception as e:
        print(f"  ️  B3 error: {e}"); clean_gpu()

    # Teacher shared untuk B4 & B5
    print(f"\n{'─'*55}\n  Training Teacher (RoBERTa-Large)\n{'─'*55}")
    try:
        teacher, tok_t = train_teacher(train_en)

        # B4: CLKD Zero-Shot
        print(f"\n{'─'*55}\n  B4: CLKD-ZS (blinded)\n{'─'*55}")
        s_zs, tok_zs, _ = distill_student(
            teacher, tok_t, train_en, zs_mode=True
        )
        tl = DataLoader(
            FakeNewsDataset(test_id.content, test_id.label, tok_zs),
            batch_size=8
        )
        after["B4: CLKD-ZS"] = evaluate_model(s_zs, tl, "CLKD-ZS blinded")
        del s_zs; clean_gpu()

        # B5: CLKD Fine-Tuned
        print(f"\n{'─'*55}\n  B5: CLKD-FT (blinded)\n{'─'*55}")
        s_ft, tok_ft, _ = distill_student(
            teacher, tok_t, train_en, zs_mode=False
        )
        after["B5: CLKD-FT"] = finetune_student(
            s_ft, tok_ft, train_id, test_id
        )
        del s_ft; clean_gpu()

        del teacher; clean_gpu()
    except Exception as e:
        print(f"  ️  CLKD error: {e}"); clean_gpu()

    return after


# ─────────────────────────────────────────────────────────────
# STEP 3 - Dokumentasi before vs after
# ─────────────────────────────────────────────────────────────
def print_and_save_report(before: dict, after: dict):
    lines = []
    W = 112

    def row(line=""):
        print(line)
        lines.append(line)

    # ── Header ────────────────────────────────────────────────
    row("=" * W)
    row("  C1 - SOURCE-BLIND PREPROCESSING: BEFORE vs AFTER")
    row("  Expected by Prof. Huang: F1 turun dari ~99% ke 80-88%")
    row("  (source bias confirmed if delta F1 < -0.05)")
    row("=" * W)

    # ── Tabel detail per metrik ────────────────────────────────
    row(f"{'Model':<24} {'Metric':<12} {'Before':>8} {'After':>8} "
        f"{'Delta':>8}  Interpretasi")
    row("-" * W)

    for name in MODEL_ORDER:
        b = before.get(name)
        a = after.get(name)
        for i, (k, lbl) in enumerate(zip(METRIC_KEYS, METRIC_LABELS)):
            bv   = b[k] if b else 0.0
            av   = a[k] if a else None
            pref = name if i == 0 else ""
            if av is None:
                row(f"{pref:<24} {lbl:<12} {bv:>8.4f} {'N/A':>8} "
                    f"{'N/A':>8}  model failed to run")
            else:
                delta  = av - bv
                interp = ""
                if k == "f1":
                    if delta < -0.10:
                        interp = " source bias CONFIRMED (drop > 10 pp)"
                    elif delta < -0.05:
                        interp = " source bias detected (drop 5-10 pp)"
                    elif av > 0.95:
                        interp = "️  still high - add more marker stripping"
                    else:
                        interp = "️  within acceptable range"
                row(f"{pref:<24} {lbl:<12} {bv:>8.4f} {av:>8.4f} "
                    f"{delta:>+8.4f}  {interp}")
        row("-" * W)

    # ── Ringkasan F1 saja ──────────────────────────────────────
    row("")
    row("=" * W)
    row("  F1-SCORE SUMMARY (Primary Metric)")
    row("=" * W)
    row(f"{'Model':<24} {'Methodology':<25} {'Before F1':>10} "
        f"{'After F1':>10} {'Delta F1':>10}  Status")
    row("-" * W)

    methodology = {
        "B1: XLM-R ZS"       : "Vanilla Transfer",
        "B2: IndoBERT"        : "Supervised Upper Bound",
        "B3: Translate-Train" : "Translation-Based",
        "B4: CLKD-ZS"         : "Proposed (Cross-Lingual)",
        "B5: CLKD-FT"         : "Proposed (+ Adaptation)",
    }

    for name in MODEL_ORDER:
        b  = before.get(name)
        a  = after.get(name)
        bf = b["f1"] if b else 0.0
        af = a["f1"] if a else None
        mt = methodology.get(name, "")
        if af is None:
            row(f"{name:<24} {mt:<25} {bf:>10.4f} {'N/A':>10} "
                f"{'N/A':>10}  -")
        else:
            delta  = af - bf
            status = (" DROP confirmed"  if delta < -0.10 else
                      " moderate drop"          if delta < -0.05 else
                      "️  still high"        if af > 0.95    else
                      "️  within acceptable range")
            row(f"{name:<24} {mt:<25} {bf:>10.4f} {af:>10.4f} "
                f"{delta:>+10.4f}  {status}")
    row("=" * W)

    # ── Interpretasi untuk thesis ──────────────────────────────
    row("")
    row("INTERPRETATION FOR THESIS (Chapter 4 / 5):")
    row("  • Drop F1 > 10 pp  -> source bias CONFIRMED, previous numbers are artifact")
    row("  • Drop F1 5-10 pp  -> source bias detected, discuss in Chapter 5")
    row("  • F1 still > 95%   -> add more marker stripping in data/prepare.py")
    row("  • If CLKD-FT drops less than IndoBERT -> contrastive alignment is effective")
    row("  • Cite: Schuster et al. (2020), Bozarth & Budak (2020)")
    row("")
    row("FRAMING SUGGESTED by Prof. Huang (if F1 drops to 75-88%):")
    row("  'We diagnose and quantify source-style bias in Indonesian fake news datasets")
    row("   and demonstrate that SG-CLKD maintains [X]% F1 under source-blind")
    row("   preprocessing, while IndoBERT drops to [Y]% - a statistically significant")
    row("   gap (Wilcoxon p < 0.05, Cohen's d > 0.5).'")

    # ── Simpan TXT ────────────────────────────────────────────
    txt_path = os.path.join(config.RESULTS_DIR, "source_blind_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n TXT report  -> {txt_path}")

    # ── Simpan JSON ───────────────────────────────────────────
    comparison = {
        "experiment"  : "C1 Source-Blind Preprocessing",
        "note"        : "BEFORE dari result_.pdf (Table 4.1); AFTER dari rerun dengan data blinded",
        "before"      : before,
        "after"       : {
            k: {mk: round(v[mk], 4) for mk in METRIC_KEYS}
            for k, v in after.items()
        },
        "delta"       : {
            name: {
                k: round(after[name][k] - before[name][k], 4)
                for k in METRIC_KEYS
            }
            for name in after if name in before
        },
        "source_bias_confirmed": {
            name: bool(before[name]["f1"] - after[name]["f1"] > 0.05)
            for name in after if name in before
        },
    }

    json_path = os.path.join(config.RESULTS_DIR, "source_blind_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2,
                  default=lambda x: x if not hasattr(x, "tolist") else x.tolist())
    print(f" JSON report -> {json_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  C1 - SOURCE-BLIND RERUN (5 Baselines)")
    print("=" * 65)

    print("\nBEFORE results (from result_.pdf - Table 4.1):")
    print(f"  {'Model':<24} {'F1':>8} {'Acc':>8} {'AUC':>8}")
    print(f"  {'-'*52}")
    for name, m in BEFORE_RESULTS.items():
        print(f"  {name:<24} {m['f1']:>8.4f} {m['accuracy']:>8.4f} "
              f"{m['auc_roc']:>8.4f}")

    # Step 1 - load data blinded
    print("\n[1/3] Loading blinded datasets ...")
    train_en, train_id, test_id = load_blinded_splits()

    # Step 2 - rerun 5 model
    print("\n[2/3] Rerunning 5 model configurations with blinded data ...")
    after = run_all_models(train_en, train_id, test_id)
    print(f"\n  {len(after)}/5 models successfully run")

    # Step 2b - FIX #4: BoW + LogReg diagnostic for lexical leakage
    print("\n[2b/3] Running BoW + LogReg diagnostic on blinded data ...")
    bow_result = bow_diagnostic(
        train_en.content, train_en.label,
        test_id.content, test_id.label,
        name="Blinded Data (source-blind)"
    )

    # Step 3 - dokumentasi
    print("\n[3/3] Generating before vs after report ...")
    print_and_save_report(BEFORE_RESULTS, after)

    print("\n[DONE] Add source_blind_report.txt table to Chapter 4 thesis.")


if __name__ == "__main__":
    main()
