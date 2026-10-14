"""run_sensitivity.py - Temperature Sensitivity (all 5 metrics per T)."""
import json, os
from torch.utils.data import DataLoader
import config
from utils import prepare_datasets, FakeNewsDataset, evaluate_model, clean_gpu, print_gpu_report
from train_clkd import train_teacher, distill_student

def main():
    clean_gpu()
    print("=" * 60 + "\n  SENSITIVITY ANALYSIS - Temperature T\n" + "=" * 60)
    data = prepare_datasets(); clean_gpu()
    teacher, tok_t = train_teacher(data["train_en"])
    results = []
    for T in config.TEMPERATURE_CANDIDATES:
        print(f"\n Testing T = {T}...")
        s, tok_s, _ = distill_student(teacher, tok_t, data["train_en"], mode="full", temperature=T)
        tl = DataLoader(FakeNewsDataset(data["test_id"].content, data["test_id"].label, tok_s), batch_size=8)
        m = evaluate_model(s, tl, f"T={T}", verbose=False)
        print(f"   -> T={T}: F1={m['f1']:.4f} Acc={m['accuracy']:.4f} AUC={m['auc_roc']:.4f} Sen={m['sensitivity']:.4f} Spe={m['specificity']:.4f}")
        results.append({"temperature": T, **m}); del s; clean_gpu()

    best = max(results, key=lambda x: x["f1"])
    print(f"\n{'='*75}\n SENSITIVITY RESULTS\n{'T':>4}  {'Accuracy':>9} {'AUC-ROC':>8} {'Sensitiv':>9} {'Specif':>8} {'F1':>8}")
    print("─"*75)
    for r in results:
        mark = " ← Best" if r["temperature"] == best["temperature"] else ""
        print(f"{r['temperature']:>4.0f}  {r['accuracy']:>9.4f} {r['auc_roc']:>8.4f} {r['sensitivity']:>9.4f} {r['specificity']:>8.4f} {r['f1']:>8.4f}{mark}")
    print(f"\n   Optimal T = {best['temperature']} (F1={best['f1']:.4f}, Acc={best['accuracy']:.4f})")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "sensitivity_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f" Saved sensitivity_results.json"); print_gpu_report()

if __name__ == "__main__":
    main()
