import json
import numpy as np
from scipy.stats import wilcoxon

def compute_cohens_dz(x, y):
    diff = np.array(x) - np.array(y)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    return mean_diff / std_diff if std_diff != 0 else 0.0

print("=== 1. EXPERIMENT 2: LABEL SWEEP (CLKD-FT vs B5) ===")
with open('experiment2_sweep_results.json', 'r') as f:
    exp2 = json.load(f)

for n_key, n_val in exp2['summary'].items():
    clkd = n_val['clkd_ft']['per_seed']
    b5 = n_val['b5']['per_seed']
    stat, p = wilcoxon(clkd, b5, alternative='greater')
    dz = compute_cohens_dz(clkd, b5)
    delta = np.mean(clkd) - np.mean(b5)
    print(f"{n_key} (N={n_val['N_per_class']}/class): Delta F1={delta:+.4f} | W={stat:.1f} | p={p:.4f} | dz={dz:+.2f}")

print("\n=== 2. MAIN RESULTS AT N=100 (TABLE 1) ===")
with open('all_results.json', 'r') as f:
    all_res = json.load(f)

runs = all_res['all_seed_runs']
clkd_ft = [r['ft']['f1'] for r in runs]
b1 = [r['b1']['f1'] for r in runs]
b2 = [r['b2']['f1'] for r in runs]
b3 = [r['b3']['f1'] for r in runs]
b4 = [r['b4']['f1'] for r in runs]

with open('experiment1_b5_results.json', 'r') as f:
    exp1 = json.load(f)
b5_main = [r['f1'] for r in exp1['per_seed_runs']]

comps = [
    ("CLKD-FT vs B1 (XLM-R ZS)", clkd_ft, b1, 'greater'),
    ("CLKD-FT vs B3 (Trans-Train)", clkd_ft, b3, 'greater'),
    ("CLKD-FT vs B4 (Trans-Test)", clkd_ft, b4, 'greater'),
    ("B2 (IndoBERT) vs CLKD-FT", b2, clkd_ft, 'greater'),
    ("CLKD-FT vs B5 (XLM-R FT)", clkd_ft, b5_main, 'greater')
]

for label, a, b, alt in comps:
    stat, p = wilcoxon(a, b, alternative=alt)
    dz = compute_cohens_dz(a, b)
    delta = np.mean(a) - np.mean(b)
    sig = "Significant (p < 0.05)" if p < 0.05 else "Not Significant"
    print(f"{label:30s}: Delta={delta:+.4f} | W={stat:4.1f} | p={p:.4f} | dz={dz:+.2f} | [{sig}]")
