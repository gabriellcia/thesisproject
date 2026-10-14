"""
config.py - SG-CLKD: Sentiment-Guided Contrastive Cross-Lingual Distillation
Sesuai blueprint Prof. Huang §4.2, §5.2, §9

GPU : NVIDIA GeForce RTX 5070 Ti (Blackwell sm_120, 16.6 GB)
OS  : Ubuntu 24.04.3 LTS
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    _props = torch.cuda.get_device_properties(0)
    print(f" GPU  : {_props.name}")
    print(f"   VRAM : {_props.total_memory/1e9:.1f} GB")
else:
    print("️  CPU mode")

# ── PATH DATA ─────────────────────────────────────────────────
DATA_DIR         = "data"
TRAIN_EN_FILE    = "train_english_clean.xlsx"
TEST_ID_FILE     = "test_indonesia_clean.xlsx"
EN_DATA_PATH     = f"{DATA_DIR}/{TRAIN_EN_FILE}"
ID_DATA_PATH     = f"{DATA_DIR}/{TEST_ID_FILE}"
TRAIN_EN_BLINDED = f"{DATA_DIR}/train_english_blinded.csv"
TEST_ID_BLINDED  = f"{DATA_DIR}/test_indonesia_blinded.csv"

TEST_SIZE          = 0.20
TEST_SIZE_EN       = TEST_SIZE
TEST_SIZE_ID       = 0.15
RANDOM_STATE       = 42
ID_TRAIN_PER_CLASS = 100

# ── MODEL ─────────────────────────────────────────────────────
TEACHER_MODEL    = "roberta-large"
STUDENT_MODEL    = "xlm-roberta-base"
INDOBERT_MODEL   = "indobenchmark/indobert-base-p2"
# Sentiment teacher following blueprint §3.1
SENTIMENT_MODEL  = "mdhugol/indonesia-bert-sentiment-classification"
NUM_LABELS       = 2

# ── TRAINING ──────────────────────────────────────────────────
MAX_LEN         = 128
BATCH_SIZE      = 8    # RTX 5070 Ti 16.6 GB
ACC_STEPS       = 2    # effective batch = 8 × 2 = 16
EPOCHS_BASELINE = 5
BASELINE_EPOCHS = EPOCHS_BASELINE
EPOCHS_DISTILL  = 10
EPOCHS_FINETUNE = 5

# LR differential following blueprint §5.2
LEARNING_RATE      = 2e-5   # backbone
LEARNING_RATE_HEAD = 1e-4   # sentiment head + projection layer
LR_BASELINE        = 2e-5

# ── FIX #7: Label Smoothing for Teacher ───────────────────────
# Set to 0.1 to increase teacher entropy (dark knowledge transfer).
# Set to 0.0 to keep original behavior.
# Teacher entropy was 0.0071 (very low) — label smoothing helps.
LABEL_SMOOTHING = 0.1

# ── GPU ───────────────────────────────────────────────────────
USE_AMP       = True
AMP_DTYPE     = torch.bfloat16  # Blackwell BF16 native
USE_8BIT_ADAM = True
USE_COMPILE   = False
PIN_MEMORY    = True
NUM_WORKERS   = 4

# ── CLKD LOSS WEIGHTS - Fine-Tuned (blueprint §4.2) ──────────
# α_task=0.50, α_logit=0.20, α_contrast=0.20, α_sent=0.10
TEMPERATURE  = 2.0
ALPHA_TASK   = 0.50
ALPHA_LOGIT  = 0.20
ALPHA_CONT   = 0.20
ALPHA_SENT   = 0.10

assert abs(ALPHA_TASK + ALPHA_LOGIT + ALPHA_CONT + ALPHA_SENT - 1.0) < 1e-6, (
    f"[D3] FT weights must equal 1.0, currently = "
    f"{ALPHA_TASK+ALPHA_LOGIT+ALPHA_CONT+ALPHA_SENT:.4f}"
)

# ── CLKD LOSS WEIGHTS - Zero-Shot (blueprint §4.2) ────────────
# α_task=0.30, α_logit=0.30, α_contrast=0.25, α_sent=0.15
TEMPERATURE_ZS    = 4.0
ALPHA_TASK_ZS     = 0.30
ALPHA_LOGIT_ZS    = 0.30
ALPHA_CONT_ZS     = 0.25
ALPHA_SENT_ZS     = 0.15
DISTILL_EPOCHS_ZS = 15

assert abs(ALPHA_TASK_ZS+ALPHA_LOGIT_ZS+ALPHA_CONT_ZS+ALPHA_SENT_ZS - 1.0) < 1e-6, (
    f"[D3] ZS weights must equal 1.0, currently = "
    f"{ALPHA_TASK_ZS+ALPHA_LOGIT_ZS+ALPHA_CONT_ZS+ALPHA_SENT_ZS:.4f}"
)

# ── TEMPERATURE PARAMETERS ────────────────────────────────────
INFONCE_TEMP   = 0.07   # blueprint §9, SimCLR default
SENTIMENT_TEMP = 2.0    # blueprint §9

# Backward compatibility
ALPHA_FEAT    = ALPHA_CONT
ALPHA_FEAT_ZS = ALPHA_CONT_ZS

# ── OUTPUT PATHS ──────────────────────────────────────────────
RESULTS_DIR        = "results"
TEACHER_SAVE_PATH  = "outputs/teacher_model.bin"
STUDENT_SAVE_PATH  = "outputs/student_sgclkd_model.bin"
TEACHER_CHECKPOINT = TEACHER_SAVE_PATH
STUDENT_CHECKPOINT = STUDENT_SAVE_PATH

# ── EXPERIMENT ────────────────────────────────────────────────
RANDOM_SEEDS    = [42, 123, 999, 2026, 777]
STABILITY_SEEDS = RANDOM_SEEDS
TEACHER_EPOCHS  = EPOCHS_BASELINE
DISTILL_EPOCHS  = EPOCHS_DISTILL

# 8 ablation modes following blueprint §4.3
ABLATION_MODES = [
    "no_distill",
    "logits_only",
    "contrast_only",
    "sent_only",
    "logits_contrast",
    "logits_sent",
    "contrast_sent",
    "full",
]

TEMPERATURE_CANDIDATES = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0]
