"""
visualize_metrics.py
====================
Visualisasi lengkap semua metrik hasil eksperimen CLKD:
  1. Bar Chart — perbandingan semua model × 5 metrik
  2. Radar Chart — profil metrik per model
  3. ROC Curve — perbandingan AUC-ROC
  4. Confusion Matrix — heatmap per model utama
  5. Tabel Lengkap — teks terformat + PNG
  6. Ablation Bar — kontribusi tiap loss component
  7. Sensitivity Bell Curve — F1 vs Temperature

Output: results/figures/*.png  (dan satu dashboard gabungan)

Usage: python visualize_metrics.py
       (jalankan setelah main.py selesai → results/all_results.json harus ada)
"""

import json, os, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import make_interp_spline

# ══════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "DejaVu Serif"],
    "font.size":      9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":     150,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.35,
    "lines.linewidth": 1.6,
})

# Palette
C = {
    "b1":   "#4A90D9",  # XLM-R ZS   → biru
    "b2":   "#27AE60",  # IndoBERT   → hijau
    "b3":   "#E67E22",  # Translate  → oranye
    "b4":   "#F39C12",  # Trans-Test → kuning tua
    "zs":   "#8E44AD",  # CLKD ZS    → ungu
    "ft":   "#C0392B",  # CLKD FT    → merah
    "teal": "#1D9E75",
    "gray": "#7F8C8D",
    "light":"#ECF0F1",
    "dark": "#2C3E50",
}

MODELS = ["B1: XLM-R ZS", "B2: IndoBERT", "B3: Translate", "B4: Trans-Test", "CLKD ZS", "CLKD FT"]
COLORS = [C["b1"], C["b2"], C["b3"], C["b4"], C["zs"], C["ft"]]
METRICS = ["accuracy", "auc_roc", "sensitivity", "specificity", "f1"]
M_LABELS = ["Accuracy", "AUC-ROC", "Sensitivity\n(Recall)", "Specificity", "F1-Score\n(Macro)"]

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
JSON_PATH = "results/all_results.json"
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(
        f"'{JSON_PATH}' tidak ditemukan.\n"
        "Jalankan python main.py terlebih dahulu untuk menghasilkan hasil eksperimen."
    )

with open(JSON_PATH) as fh:
    R = json.load(fh)

# Representative seed used for ROC curve / confusion matrix (single-run
# figures that can't be simply averaged across seeds). Falls back to
# "seed[0]" label for older result files that predate this fix.
REP_SEED = R.get("representative_seed", R.get("seeds_used", ["?"])[0])

# ── Load file JSON tambahan jika ada ─────────────────────────────
def _load_json(path):
    """Load JSON file jika ada, return None jika tidak ada."""
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None

R_SENSITIVITY  = _load_json("results/sensitivity_results.json")
R_ABLATION     = _load_json("results/ablation_results.json")
R_STABILITY    = _load_json("results/stability_results.json")
R_TRANSLATE    = _load_json("results/translate_train_results.json")

# Merge data tambahan ke R jika tidak ada di all_results.json
if R_ABLATION and not R.get("ablation"):
    R["ablation"] = R_ABLATION
if R_TRANSLATE and not R.get("baselines", {}).get("translate_train"):
    R.setdefault("baselines", {})["translate_train"] = R_TRANSLATE
if R_STABILITY:
    # Merge stability data jika lebih lengkap
    if not R.get("stability"):
        R["stability"] = {}
    if R_STABILITY.get("runs") and not R.get("stability", {}).get("zero_shot", {}).get("runs"):
        R["stability"]["zero_shot"] = {
            "runs":  R_STABILITY.get("runs", []),
            "stats": R_STABILITY.get("stats", {}),
        }

def _get(key_path, default=None):
    """Ambil nilai nested dari dict R dengan key_path list."""
    obj = R
    for k in key_path:
        if obj is None or not isinstance(obj, dict): return default
        obj = obj.get(k, default)
    return obj

# ── Model metrics ─────────────────────────────────────────────────
def model_metrics(path, agg_key=None):
    """Load metrics. Prefer aggregated mean if available."""
    if agg_key:
        agg = _get(["aggregated", agg_key])
        if agg and isinstance(agg, dict):
            out = {}
            for k in METRICS:
                v = agg.get(k)
                if isinstance(v, dict) and "mean" in v:
                    out[k] = float(v["mean"])
                elif v is not None:
                    out[k] = float(v)
                else:
                    out[k] = 0.0
            return out
    m = _get(path)
    if m is None: return {k: 0.0 for k in METRICS}
    return {k: float(m.get(k, 0.0)) for k in METRICS}

def model_std(agg_key):
    """Load std from aggregated data."""
    agg = _get(["aggregated", agg_key])
    if agg and isinstance(agg, dict):
        out = {}
        for k in METRICS:
            v = agg.get(k)
            if isinstance(v, dict) and "std" in v:
                out[k] = float(v["std"])
            else:
                out[k] = 0.0
        return out
    return {k: 0.0 for k in METRICS}

data_models = {
    "B1: XLM-R ZS":   model_metrics(["baselines", "xlmr_zeroshot"],       "b1"),
    "B2: IndoBERT":    model_metrics(["baselines", "indobert_supervised"], "b2"),
    "B3: Translate":   model_metrics(["baselines", "translate_train"],     "b3"),
    "B4: Trans-Test":  model_metrics(["baselines", "translate_test"],      "b4"),
    "CLKD ZS":         model_metrics(["clkd", "zero_shot"],               "zs"),
    "CLKD FT":         model_metrics(["clkd", "fine_tuned"],              "ft"),
}

data_stds = {
    "B1: XLM-R ZS":  model_std("b1"), "B2: IndoBERT":   model_std("b2"),
    "B3: Translate":  model_std("b3"), "B4: Trans-Test": model_std("b4"),
    "CLKD ZS":        model_std("zs"), "CLKD FT":        model_std("ft"),
}

has_aggregated = _get(["aggregated"]) is not None

# ── ROC data ──────────────────────────────────────────────────────
def roc_data(path):
    m = _get(path)
    if m is None: return [0.0, 1.0], [0.0, 1.0], 0.0
    return m.get("fpr", [0,1]), m.get("tpr", [0,1]), float(m.get("auc_roc", 0.0))

# ── Confusion matrix ──────────────────────────────────────────────
def cm_data(path):
    m = _get(path)
    if m is None: return [[0,0],[0,0]]
    return m.get("cm", [[0,0],[0,0]])

# ── Ablation — prioritas: ablation_aggregated → ablation → file terpisah ─
ablation_agg = _get(["ablation_aggregated"], [])
ablation_raw_single = _get(["ablation"], []) or R_ABLATION or []

if ablation_agg:
    # Use multi-seed aggregated data (mean values)
    ablation_raw = ablation_agg
    abl_modes = [r["mode"] for r in ablation_raw]
    abl_f1    = [float(r.get("f1_mean", r.get("f1", 0)))          for r in ablation_raw]
    abl_acc   = [float(r.get("accuracy_mean", r.get("accuracy", 0)))     for r in ablation_raw]
    abl_auc   = [float(r.get("auc_roc_mean", r.get("auc_roc", 0)))      for r in ablation_raw]
    abl_sen   = [float(r.get("sensitivity_mean", r.get("sensitivity", 0)))  for r in ablation_raw]
    abl_spe   = [float(r.get("specificity_mean", r.get("specificity", 0)))  for r in ablation_raw]
    print("  Ablation: using aggregated mean (multi-seed)")
else:
    ablation_raw = ablation_raw_single
    abl_modes = [r["mode"] for r in ablation_raw]
    abl_f1    = [float(r.get("f1", 0))          for r in ablation_raw]
    abl_acc   = [float(r.get("accuracy", 0))     for r in ablation_raw]
    abl_auc   = [float(r.get("auc_roc", 0))      for r in ablation_raw]
    abl_sen   = [float(r.get("sensitivity", 0))  for r in ablation_raw]
    abl_spe   = [float(r.get("specificity", 0))  for r in ablation_raw]
    print("  Ablation: using single-seed data (no aggregated found)")

# ── Sensitivity — baca dari all_results.json dulu, fallback ke file terpisah ─
sens_raw = (
    _get(["sensitivity_results"], None)    # embedded di all_results.json (dari main.py)
    or R_SENSITIVITY                        # file terpisah (dari run_sensitivity.py)
    or None
)

# Nilai referensi dari all_results.json (100% otomatis, tidak ada hardcode)
f1_zs = float(_get(["clkd", "zero_shot",  "f1"], 0.0))
f1_b1 = float(_get(["baselines", "xlmr_zeroshot", "f1"], 0.0))
std_zs = float(
    _get(["stability", "zero_shot", "stats", "f1", "std"], None)
    or _get(["stability", "zero_shot", "std_f1"], 0.02)
    or 0.02
)


# ══════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════
def styled_ax(ax):
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BDC3C7")
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#ECF0F1", linewidth=0.5, zorder=0)


# ══════════════════════════════════════════════════════════════════
# 1. BAR CHART — 5 metrik × 5 model
# ══════════════════════════════════════════════════════════════════
def _draw_bar_single(metric, mlabel, suffix):
    """Draw one bar chart for a single metric and save as separate PNG."""
    short_models = [
        "B1: XLM-R\nZero-Shot", "B2:\nIndoBERT",
        "B3:\nTranslate", "B4:\nTrans-Test", "CLKD\nZero-Shot", "CLKD\nFine-Tuned"
    ]
    vals = [data_models[m][metric] for m in MODELS]
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"{mlabel} - All Models", fontsize=13, fontweight="bold",
                 color=C["dark"], y=1.01)
    bars = ax.bar(range(len(MODELS)), vals, color=COLORS, width=0.55,
                  edgecolor="white", linewidth=0.8, zorder=3)
    best_idx = int(np.argmax(vals))
    bars[best_idx].set_edgecolor(C["dark"])
    bars[best_idx].set_linewidth(2.0)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=C["dark"])
    styled_ax(ax)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(short_models, fontsize=9.5, ha="center", va="top", linespacing=1.5)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylim(max(0, min(vals) - 0.12), min(1.10, max(vals) + 0.12))
    ax.set_ylabel("Score", fontsize=11)
    ax.plot(best_idx, vals[best_idx] + 0.065, marker="^", ms=9,
            color="#F4C430", markeredgecolor="#888", markeredgewidth=0.8, zorder=5)
    plt.tight_layout()
    path = f"{OUT_DIR}/1_bar_{suffix}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f" Saved: {path}")

def plot_bar_metrics():
    # Combined (all 5 in one figure)
    fig, axes = plt.subplots(1, 5, figsize=(22, 6))
    fig.suptitle("Comparison of 5 Evaluation Metrics - All Models", fontsize=13,
                 fontweight="bold", color=C["dark"], y=1.01)
    short_models = [
        "B1: XLM-R\nZero-Shot", "B2:\nIndoBERT",
        "B3:\nTranslate", "B4:\nTrans-Test", "CLKD\nZero-Shot", "CLKD\nFine-Tuned"
    ]
    for ax, metric, mlabel in zip(axes, METRICS, M_LABELS):
        vals = [data_models[m][metric] for m in MODELS]
        bars = ax.bar(range(len(MODELS)), vals, color=COLORS, width=0.6,
                      edgecolor="white", linewidth=0.8, zorder=3)
        best_idx = int(np.argmax(vals))
        bars[best_idx].set_edgecolor(C["dark"])
        bars[best_idx].set_linewidth(2.0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=C["dark"])
        styled_ax(ax)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(short_models, fontsize=8, ha="center", va="top", linespacing=1.4)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_ylim(max(0, min(vals) - 0.12), min(1.10, max(vals) + 0.10))
        ax.set_title(mlabel, fontweight="bold", pad=8, color=C["dark"])
        ax.set_ylabel("Score")
        ax.plot(best_idx, vals[best_idx] + 0.055, marker="^", ms=8,
                color="#F4C430", markeredgecolor="#999", markeredgewidth=0.8, zorder=5)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    path = f"{OUT_DIR}/1_bar_metrics.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(); print(f" Saved: {path}")

    # Separate PNG per metric
    metric_suffixes = [
        ("accuracy",    "accuracy",    "Accuracy"),
        ("auc_roc",     "auc_roc",     "AUC-ROC"),
        ("sensitivity", "sensitivity", "Sensitivity"),
        ("specificity", "specificity", "Specificity"),
        ("f1",          "f1",          "F1-Score (Macro)"),
    ]
    for metric, suffix, label in metric_suffixes:
        _draw_bar_single(metric, label, suffix)


# ══════════════════════════════════════════════════════════════════
# 2. RADAR CHART
# ══════════════════════════════════════════════════════════════════
def plot_radar():
    N = len(METRICS)
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.suptitle("Radar Chart — Profil Metrik per Model", fontsize=12, fontweight="bold",
                 color=C["dark"])
    ax.set_facecolor("white")
    ax.spines["polar"].set_color("#BDC3C7")
    ax.grid(color="#ECF0F1", linewidth=0.6)

    for model_name, color in zip(MODELS, COLORS):
        vals = [data_models[model_name][m] for m in METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, linestyle="solid", label=model_name)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(M_LABELS, size=9, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=7.5, color=C["gray"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), framealpha=0.95,
              edgecolor="#BDC3C7", title="Model", title_fontsize=8)
    plt.tight_layout()
    path = f"{OUT_DIR}/2_radar_metrics.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════
# 3. ROC CURVE
# ══════════════════════════════════════════════════════════════════
def plot_roc():
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("ROC Curve — Perbandingan AUC-ROC", fontsize=12, fontweight="bold", color=C["dark"], y=0.98)
    fig.text(0.5, 0.925,
              f"(single representative run, seed={REP_SEED} — main table di atas menggunakan mean {R.get('n_seeds','?')}-seed)",
              ha="center", fontsize=8.5, color=C["gray"], style="italic")
    styled_ax(ax); ax.grid(True, color="#ECF0F1", linewidth=0.5)

    roc_paths = [
        ("baselines", "xlmr_zeroshot"),
        ("baselines", "indobert_supervised"),
        ("baselines", "translate_train"),
        ("baselines", "translate_test"),
        ("clkd", "zero_shot"),
        ("clkd", "fine_tuned"),
    ]
    for (p1, p2), name, color in zip(roc_paths, MODELS, COLORS):
        fpr, tpr, auc = roc_data([p1, p2])
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{name}  (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], color=C["gray"], linewidth=1.0, linestyle="--", alpha=0.7, label="Random Chance")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color=C["gray"])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=9)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="#BDC3C7")
    plt.tight_layout()
    fig.subplots_adjust(top=0.86)  # reserve space for suptitle + subtitle
    path = f"{OUT_DIR}/3_roc_curve.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════
# 4. CONFUSION MATRIX — dua model utama (ZS & FT)
# ══════════════════════════════════════════════════════════════════
def _draw_single_cm(ax, cm, title, color):
    cm_arr = np.array(cm)
    cmap = LinearSegmentedColormap.from_list("custom", ["white", color], N=256)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=cmap, vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real (0)", "Fake (1)"], fontsize=10)
    ax.set_yticklabels(["Real (0)", "Fake (1)"], fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title(title, fontweight="bold", color=color, pad=10, fontsize=12)
    for i in range(2):
        for j in range(2):
            total = cm_arr.sum()
            pct   = cm_arr[i, j] / total * 100 if total > 0 else 0
            text_color = "white" if cm_arr[i, j] > cm_arr.max() * 0.5 else C["dark"]
            ax.text(j, i, f"{cm_arr[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=text_color)

def plot_confusion_matrices():
    # Combined figure (both side by side)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Confusion Matrix - CLKD Zero-Shot vs Fine-Tuned", fontsize=12,
                 fontweight="bold", color=C["dark"])
    fig.text(0.5, 0.935, f"(single representative run, seed={REP_SEED})",
              ha="center", fontsize=8.5, color=C["gray"], style="italic")
    _draw_single_cm(axes[0], cm_data(["clkd", "zero_shot"]),  "CLKD Zero-Shot",  C["zs"])
    _draw_single_cm(axes[1], cm_data(["clkd", "fine_tuned"]), "CLKD Fine-Tuned", C["ft"])
    plt.tight_layout()
    path = f"{OUT_DIR}/4_confusion_matrix.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(); print(f" Saved: {path}")

    # Separate: Zero-Shot only
    fig, ax = plt.subplots(figsize=(5.5, 5))
    fig.suptitle("Confusion Matrix - CLKD Zero-Shot", fontsize=12, fontweight="bold", color=C["dark"])
    _draw_single_cm(ax, cm_data(["clkd", "zero_shot"]), "CLKD Zero-Shot", C["zs"])
    plt.tight_layout()
    path_zs = f"{OUT_DIR}/4a_confusion_matrix_zeroshot.png"
    plt.savefig(path_zs, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(); print(f" Saved: {path_zs}")

    # Separate: Fine-Tuned only
    fig, ax = plt.subplots(figsize=(5.5, 5))
    fig.suptitle("Confusion Matrix - CLKD Fine-Tuned", fontsize=12, fontweight="bold", color=C["dark"])
    _draw_single_cm(ax, cm_data(["clkd", "fine_tuned"]), "CLKD Fine-Tuned", C["ft"])
    plt.tight_layout()
    path_ft = f"{OUT_DIR}/4b_confusion_matrix_finetuned.png"
    plt.savefig(path_ft, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(); print(f" Saved: {path_ft}")


# ══════════════════════════════════════════════════════════════════
# 5. TABEL LENGKAP (rendered as figure)
# ══════════════════════════════════════════════════════════════════
def plot_metrics_table():
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")
    fig.suptitle("Tabel Perbandingan Metrik Evaluasi — Semua Model",
                 fontsize=12, fontweight="bold", color=C["dark"], y=0.98)

    col_labels = ["Model", "Methodology", "Accuracy", "AUC-ROC", "Sensitivity\n(Recall)",
                  "Specificity", "F1-Score\n(Macro)"]
    methodology_map = {
        "B1: XLM-R ZS":  "Vanilla Transfer",
        "B2: IndoBERT":   "Upper Bound",
        "B3: Translate":  "Translation",
        "B4: Trans-Test": "Translate-Test",
        "CLKD ZS":        "Proposed (Cross-Lingual)",
        "CLKD FT":        "Proposed (+ Adaptation)",
    }
    rows = []
    numeric_rows = []  # pure floats, used for finding best-per-column
    for name in MODELS:
        m = data_models[name]
        s = data_stds.get(name, {k: 0.0 for k in METRICS})
        # Show mean±std if aggregated data exists and std > 0
        def _fmt(k):
            if has_aggregated and s.get(k, 0) > 0.001:
                return f"{m[k]:.4f}±{s[k]:.3f}"
            return f"{m[k]:.4f}"
        rows.append([
            name,
            methodology_map[name],
            _fmt('accuracy'),
            _fmt('auc_roc'),
            _fmt('sensitivity'),
            _fmt('specificity'),
            _fmt('f1'),
        ])
        numeric_rows.append([
            name, methodology_map[name],
            m['accuracy'], m['auc_roc'], m['sensitivity'],
            m['specificity'], m['f1'],
        ])

    # Find best per metric column (using pure numeric values)
    metric_cols = [2, 3, 4, 5, 6]
    best_per_col = {}
    for col in metric_cols:
        vals = [r[col] for r in numeric_rows]
        best_per_col[col] = max(vals)

    # Cell colors (compare numeric_rows, not display strings)
    cell_colors = []
    for i, row in enumerate(numeric_rows):
        row_c = []
        for j, val in enumerate(row):
            if j in metric_cols and abs(val - best_per_col[j]) < 1e-9:
                row_c.append("#D5F5E3")   # highlight best → light green
            elif i % 2 == 0:
                row_c.append("#F8F9FA")
            else:
                row_c.append("white")
        cell_colors.append(row_c)

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)

    # Header style
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor(C["dark"])
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Row color for CLKD rows
    for i, name in enumerate(MODELS):
        if "CLKD" in name:
            for j in range(len(col_labels)):
                if cell_colors[i][j] != "#D5F5E3":
                    tbl[(i + 1, j)].set_facecolor("#EBF5FB")

    plt.tight_layout()
    path = f"{OUT_DIR}/5_metrics_table.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════
# 6. ABLATION BAR — tiap metrik per mode
# ══════════════════════════════════════════════════════════════════
def plot_ablation():
    if not ablation_raw:
        print("Ablation data not found, skipping ablation plot.")
        return

    metrics_ab = [("F1-Score", abl_f1), ("Accuracy", abl_acc),
                  ("AUC-ROC", abl_auc), ("Sensitivity", abl_sen), ("Specificity", abl_spe)]

    # Color map by component type
    def _abl_color(mode):
        m = mode.lower()
        if "full" in m:           return "#2C2C2C"
        if "logits + contrast" in m: return "#5B8FD4"
        if "logits + sentiment" in m: return "#E08B3A"
        if "contrast + sentiment" in m: return "#4BAE8A"
        if "logits" in m:         return "#5B8FD4"
        if "contrast" in m:       return "#4BAE8A"
        if "sentiment" in m:      return "#E08B3A"
        return "#B0B0B0"

    abl_colors = [_abl_color(m) for m in abl_modes]

    # Short readable labels — max 2 lines, no overlap
    def _short(mode):
        replacements = {
            "Baseline (No-Distill)": "Baseline\n(No-Distill)",
            "No-Distill":            "No-\nDistill",
            "Logits-Only":           "Logits\nOnly",
            "Contrast-Only (InfoNCE)": "Contrast\nOnly",
            "Sentiment-Only":        "Sentiment\nOnly",
            "Logits + Contrast":     "Logits +\nContrast",
            "Logits + Sentiment":    "Logits +\nSentiment",
            "Contrast + Sentiment":  "Contrast +\nSentiment",
            "Full SG-CLKD (Ours)":   "Full\nSG-CLKD",
        }
        return replacements.get(mode, mode.replace(" ", "\n"))

    short_labels = [_short(m) for m in abl_modes]

    fig, axes = plt.subplots(1, 5, figsize=(24, 7))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("Ablation Study - Loss Component Contribution per Metric",
                 fontsize=14, fontweight="bold", color=C["dark"], y=1.01)

    import matplotlib.patches as mpatches

    for ax, (mlabel, vals) in zip(axes, metrics_ab):
        x = np.arange(len(abl_modes))
        bars = ax.bar(x, vals, color=abl_colors, width=0.65,
                      edgecolor="white", linewidth=0.6, zorder=3)

        # Highlight best bar
        best_i = int(np.argmax(vals))
        bars[best_i].set_edgecolor("#111111")
        bars[best_i].set_linewidth(1.8)

        # Value labels on top of each bar
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="500", color="#222")

        # Reference line at baseline (No-Distill) value
        ax.axhline(y=vals[0], color="#E24B4A", linestyle="--",
                   linewidth=1.2, alpha=0.7, zorder=2)

        styled_ax(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8, color="#333",
                           ha="center", va="top", linespacing=1.4)
        ax.tick_params(axis="x", length=0, pad=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylim(min(vals) - 0.07, max(vals) + 0.065)
        ax.set_title(mlabel, fontweight="bold", pad=10, fontsize=12)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_facecolor("#FAFAFA")

    # Legend
    legend_items = [
        mpatches.Patch(color="#B0B0B0", label="Ablation baseline"),
        mpatches.Patch(color="#5B8FD4", label="Logit-based"),
        mpatches.Patch(color="#4BAE8A", label="Contrast-based"),
        mpatches.Patch(color="#E08B3A", label="Sentiment-based"),
        mpatches.Patch(color="#2C2C2C", label="Full SG-CLKD (Ours)"),
        plt.Line2D([0],[0], color="#E24B4A", linestyle="--",
                   linewidth=1.2, label="Baseline (No-Distill)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=6,
               fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = f"{OUT_DIR}/6_ablation_metrics.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(); print(f" Saved: {path}")

    # Separate PNG per metric
    abl_metric_map = [
        ("F1-Score",    abl_f1,  "f1"),
        ("Accuracy",    abl_acc, "accuracy"),
        ("AUC-ROC",     abl_auc, "auc_roc"),
        ("Sensitivity", abl_sen, "sensitivity"),
        ("Specificity", abl_spe, "specificity"),
    ]
    for mlabel, vals, suffix in abl_metric_map:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        fig2.patch.set_facecolor("#FAFAFA")
        fig2.suptitle(f"Ablation Study - {mlabel}", fontsize=13,
                      fontweight="bold", color=C["dark"], y=1.01)
        x = np.arange(len(abl_modes))
        bars2 = ax2.bar(x, vals, color=abl_colors, width=0.6,
                        edgecolor="white", linewidth=0.6, zorder=3)
        best_i2 = int(np.argmax(vals))
        bars2[best_i2].set_edgecolor("#111111")
        bars2[best_i2].set_linewidth(2.0)
        for bar, v in zip(bars2, vals):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.004,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontsize=9.5, fontweight="bold", color="#222")
        ax2.axhline(y=vals[0], color="#E24B4A", linestyle="--",
                    linewidth=1.2, alpha=0.7, zorder=2,
                    label="Baseline (No-Distill)")
        styled_ax(ax2)
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_labels, fontsize=9, color="#333",
                            ha="center", va="top", linespacing=1.5)
        ax2.tick_params(axis="x", length=0, pad=10)
        ax2.tick_params(axis="y", labelsize=10)
        ax2.set_ylim(min(vals) - 0.08, max(vals) + 0.08)
        ax2.set_ylabel("Score", fontsize=11)
        ax2.set_facecolor("#FAFAFA")
        ax2.legend(fontsize=9, loc="lower right", framealpha=0.85)
        plt.tight_layout()
        path2 = f"{OUT_DIR}/6_ablation_{suffix}.png"
        plt.savefig(path2, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
        plt.close(); print(f" Saved: {path2}")


# ══════════════════════════════════════════════════════════════════
# 7. SENSITIVITY CURVE — F1 vs Temperature (bell curve)
# ══════════════════════════════════════════════════════════════════
def plot_sensitivity():
    if sens_raw is not None:
        temps = np.array([s["temperature"] for s in sens_raw], dtype=float)
        f1s   = np.array([s.get("f1", 0) for s in sens_raw])
        accs  = np.array([s.get("accuracy", 0) for s in sens_raw])
        aucs  = np.array([s.get("auc_roc", 0) for s in sens_raw])
    else:
        print("⚠️  Data sensitivity tidak ditemukan.")
        print("   Pastikan python main.py sudah dijalankan sampai selesai.")
        print("   Data sensitivity akan otomatis tersimpan di all_results.json.")
        print("   Plot 7_sensitivity_curve.png dilewati.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Sensitivity Analysis — F1, Accuracy & AUC-ROC vs Temperature T",
                 fontsize=13, fontweight="bold", color=C["dark"])

    pairs = [("F1-Score", f1s, C["teal"]), ("Accuracy", accs, C["b1"]), ("AUC-ROC", aucs, C["zs"])]

    for ax, (mlabel, vals, color) in zip(axes, pairs):
        styled_ax(ax)

        # ── Smooth spline curve + fill ─────────────────────────────
        if len(temps) >= 4:
            t_sm = np.linspace(temps.min(), temps.max(), 400)
            spl  = make_interp_spline(temps, vals, k=min(3, len(temps)-1))
            y_sm = spl(t_sm)
            y_base = max(0, vals.min() - 0.04)
            ax.fill_between(t_sm, y_base, y_sm, color=color, alpha=0.13)
            ax.plot(t_sm, y_sm, color=color, linewidth=2.5, zorder=3)

        # ── Data points ────────────────────────────────────────────
        ax.scatter(temps, vals, color=color, s=45, zorder=5)

        best_i = int(np.argmax(vals))

        # ── Vertical & horizontal guide lines ─────────────────────
        ax.axvline(temps[best_i], color=color, linewidth=1.0, linestyle=":", alpha=0.75)
        ax.axhline(f1_b1 if mlabel == "F1-Score" else 0,
                   color=C["gray"], linewidth=1.0, linestyle="--", alpha=0.55,
                   label="Baseline 1" if mlabel == "F1-Score" else "")

        # ── Best point annotation (arrow, top-right of peak) ───────
        x_off = 0.45 if temps[best_i] < temps.max() - 1 else -1.2
        y_off = vals[best_i] + (vals.max() - vals.min()) * 0.12
        ax.annotate(
            f"T*={int(temps[best_i])}\n{vals[best_i]:.4f}",
            xy=(temps[best_i], vals[best_i]),
            xytext=(temps[best_i] + x_off, y_off),
            fontsize=9, color=color, fontweight="bold",
            ha="left" if x_off > 0 else "right",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.1, connectionstyle="arc3,rad=0.15"),
        )

        # ── Non-best point labels — clean fixed alternating offsets ──
        val_range = vals.max() - vals.min()
        n = len(temps)
        for idx, (t, v) in enumerate(zip(temps, vals)):
            if idx == best_i:
                continue

            # Alternate above/below based on index parity to avoid overlap
            if idx % 2 == 0:
                va, y_pt_off = "bottom", 14
            else:
                va, y_pt_off = "top", -14

            # Force edge points above always
            if idx == 0 or idx == n - 1:
                va, y_pt_off = "bottom", 14

            # Extra push for value very close to best
            if abs(v - vals[best_i]) < val_range * 0.12:
                y_pt_off = 22 if va == "bottom" else -22

            # Small horizontal nudge for points close together on x-axis
            x_nudge = 8 if idx < n // 2 else -8
            if idx == 0:
                x_nudge = 6
            elif idx == n - 1:
                x_nudge = -6

            ax.annotate(
                f"{v:.3f}",
                xy=(t, v),
                xytext=(x_nudge, y_pt_off),
                textcoords="offset points",
                ha="center", va=va,
                fontsize=8.5, color=C["gray"],
                fontweight="normal",
            )

        # ── Axes config ────────────────────────────────────────────
        pad  = (vals.max() - vals.min()) * 0.28
        ax.set_ylim(max(0, vals.min() - pad), vals.max() + pad * 1.5)
        ax.set_xlabel("Temperature T  (→ softer labels)", fontsize=9, labelpad=6)
        ax.set_ylabel(mlabel, fontsize=9)
        ax.set_xticks(temps.astype(int))
        ax.set_xlim(temps.min() - 0.6, temps.max() + 0.6)
        ax.set_title(f"{mlabel} vs Temperature", fontweight="bold", pad=10, fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
        if mlabel == "F1-Score":
            ax.legend(fontsize=8, loc="lower left", framealpha=0.85)

    plt.tight_layout(pad=2.0)
    path = f"{OUT_DIR}/7_sensitivity_curve.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(); print(f" Saved: {path}")

    # Separate PNG per metric
    sens_metric_map = [
        ("F1-Score",  f1s,  C["teal"], "f1",  True),
        ("Accuracy",  accs, C["b1"],   "accuracy", False),
        ("AUC-ROC",   aucs, C["zs"],   "auc_roc",  False),
    ]
    for mlabel, vals, color, suffix, show_baseline in sens_metric_map:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        fig2.suptitle(f"Sensitivity Analysis - {mlabel} vs Temperature T",
                      fontsize=13, fontweight="bold", color=C["dark"])
        styled_ax(ax2)
        if len(temps) >= 4:
            t_sm = np.linspace(temps.min(), temps.max(), 400)
            spl  = make_interp_spline(temps, vals, k=min(3, len(temps)-1))
            y_sm = spl(t_sm)
            ax2.fill_between(t_sm, max(0, vals.min()-0.04), y_sm, color=color, alpha=0.13)
            ax2.plot(t_sm, y_sm, color=color, linewidth=2.5, zorder=3)
        ax2.scatter(temps, vals, color=color, s=60, zorder=5)
        best_i = int(np.argmax(vals))
        ax2.axvline(temps[best_i], color=color, linewidth=1.0, linestyle=":", alpha=0.75)
        if show_baseline:
            ax2.axhline(f1_b1, color=C["gray"], linewidth=1.0, linestyle="--",
                        alpha=0.55, label="Baseline 1")
        x_off = 0.45 if temps[best_i] < temps.max() - 1 else -1.2
        y_off = vals[best_i] + (vals.max() - vals.min()) * 0.12
        ax2.annotate(
            f"T*={int(temps[best_i])}\n{vals[best_i]:.4f}",
            xy=(temps[best_i], vals[best_i]),
            xytext=(temps[best_i] + x_off, y_off),
            fontsize=10, color=color, fontweight="bold",
            ha="left" if x_off > 0 else "right",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                            connectionstyle="arc3,rad=0.15"),
        )
        val_range = vals.max() - vals.min()
        n = len(temps)
        for idx, (t, v) in enumerate(zip(temps, vals)):
            if idx == best_i: continue
            va2, yoff2 = ("bottom", 16) if idx % 2 == 0 else ("top", -16)
            if idx == 0 or idx == n-1: va2, yoff2 = "bottom", 16
            if abs(v - vals[best_i]) < val_range * 0.12:
                yoff2 = 24 if va2 == "bottom" else -24
            xnudge = 10 if idx < n//2 else -10
            ax2.annotate(f"{v:.3f}", xy=(t, v), xytext=(xnudge, yoff2),
                         textcoords="offset points", ha="center", va=va2,
                         fontsize=9.5, color=C["gray"])
        pad = (vals.max() - vals.min()) * 0.30
        ax2.set_ylim(max(0, vals.min() - pad), vals.max() + pad * 1.6)
        ax2.set_xlabel("Temperature T  (softer labels as T increases)", fontsize=10, labelpad=6)
        ax2.set_ylabel(mlabel, fontsize=10)
        ax2.set_xticks(temps.astype(int))
        ax2.set_xlim(temps.min() - 0.6, temps.max() + 0.6)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v2, _: f"{v2:.3f}"))
        if show_baseline:
            ax2.legend(fontsize=9, loc="lower left", framealpha=0.85)
        plt.tight_layout()
        path2 = f"{OUT_DIR}/7_sensitivity_{suffix}.png"
        plt.savefig(path2, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(); print(f" Saved: {path2}")


# ══════════════════════════════════════════════════════════════════
# DASHBOARD — semua panel dalam satu figure
# ══════════════════════════════════════════════════════════════════
def plot_dashboard():
    fig = plt.figure(figsize=(20, 22), facecolor="white")
    fig.suptitle("CLKD Fake News Detection — Dashboard Evaluasi Lengkap",
                 fontsize=15, fontweight="bold", color=C["dark"], y=0.99)

    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.55, wspace=0.45,
                           left=0.06, right=0.97, top=0.96, bottom=0.04)

    # ── Row 0: Bar charts (5 metrics) ──────────────────────────────
    for col, (metric, mlabel) in enumerate(zip(METRICS, M_LABELS)):
        ax = fig.add_subplot(gs[0, col])
        vals = [data_models[m][metric] for m in MODELS]
        bars = ax.bar(range(len(MODELS)), vals, color=COLORS, width=0.6,
                      edgecolor="white", linewidth=0.8, zorder=3)
        best_i = int(np.argmax(vals))
        bars[best_i].set_edgecolor(C["dark"]); bars[best_i].set_linewidth(2.0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.3f}", ha="center", fontsize=6.5, fontweight="bold")
        ax.text(best_i, vals[best_i]+0.05, "*BEST*", ha="center", fontsize=7, color="darkorange", fontweight="bold")
        styled_ax(ax)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=40, ha="right", fontsize=6.5)
        ax.set_ylim(max(0, min(vals)-0.12), min(1.12, max(vals)+0.10))
        ax.set_title(mlabel, fontweight="bold", fontsize=9, pad=6)
        if col == 0: ax.set_ylabel("Score", fontsize=8)

    # ── Row 1 col 0-1: Radar ───────────────────────────────────────
    ax_radar = fig.add_subplot(gs[1, 0:2], polar=True)
    N = len(METRICS)
    angles = np.linspace(0, 2*math.pi, N, endpoint=False).tolist(); angles += angles[:1]
    ax_radar.set_facecolor("white"); ax_radar.grid(color="#ECF0F1", linewidth=0.5)
    for name, color in zip(MODELS, COLORS):
        vals = [data_models[name][m] for m in METRICS] + [data_models[name][METRICS[0]]]
        ax_radar.plot(angles, vals, color=color, linewidth=1.8, label=name)
        ax_radar.fill(angles, vals, color=color, alpha=0.07)
    ax_radar.set_xticks(angles[:-1]); ax_radar.set_xticklabels(M_LABELS, size=7.5)
    ax_radar.set_ylim(0, 1); ax_radar.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax_radar.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], size=6.5)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7, framealpha=0.9)
    ax_radar.set_title("Radar Chart — Profil Metrik", fontsize=9, fontweight="bold", pad=15)

    # ── Row 1 col 2-4: ROC Curve ───────────────────────────────────
    ax_roc = fig.add_subplot(gs[1, 2:])
    styled_ax(ax_roc); ax_roc.grid(True, color="#ECF0F1", linewidth=0.4)
    roc_paths = [("baselines","xlmr_zeroshot"),("baselines","indobert_supervised"),
                 ("baselines","translate_train"),("baselines","translate_test"),
                 ("clkd","zero_shot"),("clkd","fine_tuned")]
    for (p1,p2), name, color in zip(roc_paths, MODELS, COLORS):
        fpr, tpr, auc = roc_data([p1, p2])
        ax_roc.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.4f})")
    ax_roc.plot([0,1],[0,1], color=C["gray"], ls="--", lw=1, alpha=0.6, label="Chance")
    ax_roc.set_xlabel("False Positive Rate", fontsize=8); ax_roc.set_ylabel("True Positive Rate", fontsize=8)
    ax_roc.legend(loc="lower right", fontsize=7.5, framealpha=0.95)
    ax_roc.set_title("ROC Curve — AUC-ROC Comparison", fontsize=9, fontweight="bold")

    # ── Row 2: Confusion Matrices (ZS + FT) + Ablation ─────────────
    for col_off, (cm_path, cm_title, cm_color) in enumerate([
        (["clkd","zero_shot"],  "CM: CLKD Zero-Shot",  C["zs"]),
        (["clkd","fine_tuned"], "CM: CLKD Fine-Tuned", C["ft"]),
    ]):
        ax_cm = fig.add_subplot(gs[2, col_off*1 : col_off*1+2])
        cm_arr = np.array(cm_data(cm_path))
        cmap   = LinearSegmentedColormap.from_list("cm_cmap", ["white", cm_color], N=256)
        im     = ax_cm.imshow(cm_arr, cmap=cmap, vmin=0)
        plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
        ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
        ax_cm.set_xticklabels(["Real","Fake"], fontsize=8)
        ax_cm.set_yticklabels(["Real","Fake"], fontsize=8)
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
        ax_cm.set_title(cm_title, fontweight="bold", color=cm_color, fontsize=9)
        for i in range(2):
            for j in range(2):
                tot = cm_arr.sum()
                pct = cm_arr[i,j]/tot*100 if tot>0 else 0
                tc  = "white" if cm_arr[i,j] > cm_arr.max()*0.5 else C["dark"]
                ax_cm.text(j, i, f"{cm_arr[i,j]}\n({pct:.1f}%)", ha="center", va="center",
                           fontsize=9, fontweight="bold", color=tc)

    # Ablation bars (F1)
    ax_abl = fig.add_subplot(gs[2, 2:])
    if ablation_raw:
        abl_c = [C["gray"], C["b1"], C["b3"], C["teal"]][:len(abl_modes)]
        bars_a = ax_abl.bar(range(len(abl_modes)), abl_f1, color=abl_c, width=0.5,
                            edgecolor="white", linewidth=0.8, zorder=3)
        best_a = int(np.argmax(abl_f1))
        bars_a[best_a].set_edgecolor(C["dark"]); bars_a[best_a].set_linewidth(2)
        for bar, v in zip(bars_a, abl_f1):
            ax_abl.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
        styled_ax(ax_abl)
        ax_abl.set_xticks(range(len(abl_modes)))
        ax_abl.set_xticklabels([m.replace(" ", "\n") for m in abl_modes], fontsize=8)
        ax_abl.set_ylim(max(0, min(abl_f1)-0.08), min(1.08, max(abl_f1)+0.08))
        ax_abl.set_title("Ablation Study — F1-Score per Loss Mode", fontweight="bold", fontsize=9)
        ax_abl.set_ylabel("F1-Score")

    # ── Row 3: Tabel metrik ────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[3, :])
    ax_tbl.axis("off")
    col_labels = ["Model", "Methodology", "Accuracy", "AUC-ROC", "Sensitivity\n(Recall)",
                  "Specificity", "F1-Score\n(Macro)"]
    methodology_map = {
        "B1: XLM-R ZS": "Vanilla Transfer", "B2: IndoBERT": "Upper Bound",
        "B3: Translate": "Translation", "B4: Trans-Test": "Translate-Test",
        "CLKD ZS": "Proposed (Cross-Lingual)", "CLKD FT": "Proposed (+ Adapt.)",
    }
    rows_t = []
    for name in MODELS:
        m = data_models[name]
        rows_t.append([name, methodology_map[name],
                       f"{m['accuracy']:.4f}", f"{m['auc_roc']:.4f}",
                       f"{m['sensitivity']:.4f}", f"{m['specificity']:.4f}", f"{m['f1']:.4f}"])
    metric_cols = [2,3,4,5,6]
    best_vals = {c: max(float(r[c]) for r in rows_t) for c in metric_cols}
    cell_c = [["#D5F5E3" if j in metric_cols and float(rows_t[i][j])==best_vals[j]
                else ("#F8F9FA" if i%2==0 else "white")
               for j in range(len(col_labels))]
              for i in range(len(rows_t))]
    for i, name in enumerate(MODELS):
        if "CLKD" in name:
            for j in range(len(col_labels)):
                if cell_c[i][j] != "#D5F5E3": cell_c[i][j] = "#EBF5FB"
    tbl = ax_tbl.table(cellText=rows_t, colLabels=col_labels, cellLoc="center",
                       loc="center", cellColours=cell_c)
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 2.2)
    for j in range(len(col_labels)):
        tbl[(0,j)].set_facecolor(C["dark"]); tbl[(0,j)].set_text_props(color="white", fontweight="bold")

    path = f"{OUT_DIR}/0_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  VISUALIZE METRICS — CLKD Experiment")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}/\n")

    plot_bar_metrics()
    plot_radar()
    plot_roc()
    plot_confusion_matrices()
    plot_metrics_table()
    plot_ablation()
    plot_sensitivity()
    plot_dashboard()

    print(f"\n{'='*60}")
    print(f"✅ Semua visualisasi selesai!")
    print(f"📁 File tersimpan di: {OUT_DIR}/")
    print(f"   0_dashboard.png       — Semua panel dalam 1 gambar")
    print(f"   1_bar_metrics.png     — Bar chart 5 metrik")
    print(f"   2_radar_metrics.png   — Radar chart profil model")
    print(f"   3_roc_curve.png       — ROC curve & AUC-ROC")
    print(f"   4_confusion_matrix.png — Confusion matrix heatmap")
    print(f"   5_metrics_table.png   — Tabel lengkap metrik")
    print(f"   6_ablation_metrics.png — Ablation study chart")
    print(f"   7_sensitivity_curve.png — Temperature sensitivity")
    print(f"{'='*60}")
