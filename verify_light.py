"""
verify_light.py — subset-identity check between main-run and sweep.
No torch, no transformers needed — just pandas and sklearn.
Run from ~/project_fixed_sgclkd/
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from sklearn.model_selection import train_test_split
import config


def load_data(file_path):
    """Simplified version of utils.load_data — no verbose print."""
    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
    else:
        df = pd.read_excel(file_path)
    df.columns = [c.replace("﻿", "").strip() for c in df.columns]
    label_map = {"real": 0, "fake": 1, "Real": 0, "Fake": 1, 0: 0, 1: 1}
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["content", "label"])
    df["label"] = df["label"].astype(int)
    return df


print(f"Loading {config.ID_DATA_PATH}...")
df_id = load_data(config.ID_DATA_PATH)
print(f"Loaded {len(df_id)} rows")

SEEDS = [42, 123, 999, 2026, 777]
N = 100
all_match = True

print(f"\nComparing subsets at N={N} per class across {len(SEEDS)} seeds:")
print("-" * 70)

for seed in SEEDS:
    # Shared split (both pipelines call this identically)
    train_id_full, test_id = train_test_split(
        df_id, test_size=config.TEST_SIZE_ID,
        stratify=df_id['label'], random_state=seed,
    )

    # ── Main-run approach (utils.prepare_datasets lines 213-216) ──
    main_subset = pd.concat([
        train_id_full[train_id_full.label == 0].sample(n=N, random_state=seed),
        train_id_full[train_id_full.label == 1].sample(n=N, random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Sweep approach (run_experiment2.prepare_sweep_data lines 117-139) ──
    pool_real = (train_id_full[train_id_full.label == 0]
                 .sample(frac=1, random_state=seed)
                 .reset_index(drop=True))
    pool_fake = (train_id_full[train_id_full.label == 1]
                 .sample(frac=1, random_state=seed)
                 .reset_index(drop=True))
    sweep_subset = pd.concat([
        pool_real.iloc[:N],
        pool_fake.iloc[:N],
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Compare by content strings (indices are reset in both, so use content)
    main_contents = set(main_subset['content'].astype(str).tolist())
    sweep_contents = set(sweep_subset['content'].astype(str).tolist())

    overlap = len(main_contents & sweep_contents)
    match = main_contents == sweep_contents
    all_match = all_match and match

    print(f"  seed {seed}: main={len(main_contents)} sweep={len(sweep_contents)} "
          f"overlap={overlap}/200  match={match}")

print("-" * 70)
print(f"\nALL SEEDS MATCH ROW-FOR-ROW: {all_match}")

if all_match:
    print("\n✓ Paper §4.4 subset-identity claim is EMPIRICALLY VERIFIED.")
    print("  Paired Wilcoxon tests in Table 2 remain valid.")
else:
    print("\n✗ Subsets differ. Paper §4.4 claim must be corrected.")
    print("  Table 2 Wilcoxon tests (CLKD-FT vs B1-B4, B2) should switch to")
    print("  descriptive stats or Mann-Whitney (unpaired), per Prof's guidance.")
