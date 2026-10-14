"""
verify_subsets.py — quick subset-identity check between main-run and sweep.

Runs from inside ~/project_fixed_sgclkd/ (where run_experiment2.py and
utils.py live). Compares the N=100 Indonesian training subset drawn by:
  - prepare_sweep_data()  from run_experiment2.py
  - prepare_datasets()    from utils.py (used by main.py and run_experiment1.py)

Prints per-seed match status and a verdict.

Usage:
    cd ~/project_fixed_sgclkd
    python3 verify_subsets.py
"""
import sys
sys.path.insert(0, '.')

from run_experiment2 import prepare_sweep_data
from utils import prepare_datasets

SEEDS = [42, 123, 999, 2026, 777]
N = 100

all_match = True
for seed in SEEDS:
    sweep_full = prepare_sweep_data(seed)
    sweep_subset = sweep_full['nested_subsets'][N]
    sweep_ids = set(sweep_subset.index.tolist())

    main_full = prepare_datasets(random_state=seed, verbose=False)
    main_subset = main_full['train_id']
    main_ids = set(main_subset.index.tolist())

    overlap = len(sweep_ids & main_ids)
    match = sweep_ids == main_ids
    all_match = all_match and match

    print(f"seed {seed}: sweep={len(sweep_ids)} main={len(main_ids)} "
          f"overlap={overlap} match={match}")

print()
print(f"ALL SEEDS MATCH ROW-FOR-ROW: {all_match}")
if all_match:
    print("→ The §4.4 subset-identity claim in the paper is verified.")
else:
    print("→ Subsets differ. Paper §4.4 claim needs to be weakened,")
    print("  and Table 2 Wilcoxon tests against B1-B4/B2 should switch to")
    print("  descriptive stats or Mann-Whitney (unpaired).")
