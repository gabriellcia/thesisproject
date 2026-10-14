# SG-CLKD: Cross-Lingual Distillation for Indonesian Fake News Detection

Code for *Cross-Lingual Distillation for Indonesian Fake News Detection: A Cautionary Label-Efficiency Study* (submitted to the Journal of Intelligent Information Systems, 2026).

**Authors:** Gabriella Alicia Setiawan, Sheng-Chih Huang (Tamkang University)

## Setting

Seven cross-lingual and monolingual configurations compared on a 154-instance Indonesian fake-news test set, across six label budgets ($N \in \{10, 25, 50, 100, 200, 240\}$ per class), with ten stability seeds at the two smallest budgets. The main finding is budget-dependent and is reported as three distinct statements in the paper; see §5 and §6.

## Repository layout

| File / directory | Purpose |
|---|---|
| `main.py` | Main-run pipeline (produces Table 2 N=100 baseline numbers) |
| `run_experiment1.py` | B5 (no-distillation fine-tuning control) baseline |
| `run_experiment2.py` | Label-efficiency sweep (produces Table 3 and Figure 2) |
| `run_source_blind.py` | Source-blind preprocessing diagnostic |
| `run_translate_train.py` | Translate-Train baseline (B3) |
| `run_stability.py`, `run_ablation.py`, `run_sensitivity.py` | Secondary diagnostics |
| `run_bow_154_portable.py` | BoW + LR baseline |
| `train_clkd.py` | SG-CLKD training (teacher, distillation, fine-tune) |
| `train_baselines.py` | Baseline model training |
| `utils.py` | Dataset class, evaluation metrics, statistical tests |
| `config.py` | Hyperparameters, seeds, model IDs |
| `prepare.py` | Data loading and preprocessing |
| `verify_subsets.py`, `verify_light.py` | Subset-identity check between main-run and sweep (CHECK1) |
| `visualize_metrics.py` | Figure generation |

## Data

The Indonesian corpus is not redistributed for copyright reasons. The URL list is available under academic-use terms from the corresponding author on reasonable request. The 85/15 train/test split is reproducible from the seed list in `config.py` (`RANDOM_SEEDS = [42, 123, 999, 2026, 777]`).

## Reproducing the paper

The exact numbers reported in the paper were produced on an NVIDIA GeForce RTX 5070 Ti (Blackwell architecture, sm_120) under Ubuntu 24.04.3 LTS with CUDA 12.8 and PyTorch 2.5+. GPU non-determinism may introduce differences of ±0.005 F1 across hardware.

Pinned package versions are in `requirements.txt`. To reproduce:

```bash
pip install -r requirements.txt
python main.py              # produces all_results.json (Table 2 baselines)
python run_experiment1.py   # B5 control (uses all_results.json for comparisons)
python run_experiment2.py   # label-efficiency sweep (Tables 3, 4, Figure 2)
python verify_subsets.py    # CHECK1 subset-identity verification
```

## Citation

```bibtex
@article{setiawan2026sgclkd,
  title   = {Cross-Lingual Distillation for Indonesian Fake News Detection: A Cautionary Label-Efficiency Study},
  author  = {Setiawan, Gabriella Alicia and Huang, Sheng-Chih},
  journal = {Journal of Intelligent Information Systems},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released for academic reproducibility. For commercial use, please contact the authors.
