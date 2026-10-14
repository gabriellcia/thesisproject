#!/bin/bash
set -e
cd ~/project_fixed_sgclkd

echo "=== STEP 1: Source-blind preprocessing ==="
python data/prepare.py

echo "=== STEP 2: BoW diagnostic ==="
python run_bow_diagnostic.py 2>&1 | tee log_bow.txt

echo "=== STEP 3: Main experiment ==="
python main.py 2>&1 | tee log_main_v3.txt

echo "=== STEP 4: Multi-seed ablation ==="
python run_ablation.py 2>&1 | tee log_ablation_multiseed.txt

echo "=== STEP 5: Visualize results ==="
python visualize_metrics.py 2>&1 | tee log_visualize.txt

echo "=== ALL DONE ==="
