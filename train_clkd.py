"""
train_clkd.py - SG-CLKD: Sentiment-Guided Contrastive Cross-Lingual Distillation
==================================================================================
Implementasi following blueprint Prof. Huang §2, §3, §4, §5, §8

Perubahan dari versi sebelumnya:
   InfoNCE: Linear(1024->768, bias=False) + symmetric loss (s2t + t2s)/2
   Sentiment Teacher: mdhugol/indonesia-bert-sentiment-classification
   Student sentiment head: nn.Linear(768, 3)
   Loss weights 4 komponen following blueprint §4.2
   LR differential: backbone=2e-5, head+proj=1e-4
   Ablation 8 mode following blueprint §4.3
   Sanity checks: teacher entropy + cosine similarity tracking
   BF16 + 8-bit Adam untuk RTX 5070 Ti
"""

import os, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    RobertaForSequenceClassification,
    XLMRobertaForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import config
from utils import FakeNewsDataset, evaluate_model, clean_gpu, device


# ─────────────────────────────────────────────────────────────
# COMPONENT 1: ContrastiveDistillationLoss (blueprint §2.4)
# ─────────────────────────────────────────────────────────────
class ContrastiveDistillationLoss(nn.Module):
    """
    InfoNCE antara student [CLS] dan projected teacher [CLS].
    Sesuai blueprint §2.4 - SYMMETRIC loss (s2t + t2s) / 2.

    Positive pair : (student[i], teacher[i]) - artikel SAMA
    Negative pair : (student[i], teacher[j]) - artikel BERBEDA
    """
    def __init__(self, teacher_dim: int = 1024,
                 student_dim: int = 768,
                 temperature: float = 0.07):
        super().__init__()
        # Linear projection following blueprint §2.4 (bias=False)
        self.projection  = nn.Linear(teacher_dim, student_dim, bias=False)
        self.temperature = temperature

    def forward(self, student_cls: torch.Tensor,
                teacher_cls: torch.Tensor) -> torch.Tensor:
        # Project teacher ke student space: (N, 1024) -> (N, 768)
        teacher_proj = self.projection(teacher_cls)

        # L2 normalise
        s = F.normalize(student_cls,  dim=-1)   # (N, 768)
        t = F.normalize(teacher_proj, dim=-1)   # (N, 768)

        # Pairwise cosine similarity (N, N)
        logits = s @ t.t() / self.temperature

        # Diagonal = positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # Symmetric InfoNCE (blueprint §2.4)
        loss_s2t = F.cross_entropy(logits,   labels)
        loss_t2s = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_s2t + loss_t2s)

    def cosine_stats(self, student_cls: torch.Tensor,
                     teacher_cls: torch.Tensor) -> dict:
        """Sanity check §8.3: track matched vs unmatched cosine similarity."""
        with torch.no_grad():
            t_proj = self.projection(teacher_cls)
            s = F.normalize(student_cls, dim=-1)
            t = F.normalize(t_proj, dim=-1)
            sim_matrix = s @ t.t()
            N = sim_matrix.size(0)
            matched   = sim_matrix.diag().mean().item()
            unmatched = (sim_matrix.sum() - sim_matrix.diag().sum()) \
                        / (N * N - N)
            unmatched = unmatched.item()
        return {"matched": matched, "unmatched": unmatched,
                "gap": matched - unmatched}


# ─────────────────────────────────────────────────────────────
# COMPONENT 2: SentimentDistillationLoss (blueprint §3.3)
# ─────────────────────────────────────────────────────────────
class SentimentDistillationLoss(nn.Module):
    """
    KL divergence antara sentiment teacher dan student sentiment head.
    L_sent = τ_s² · KL(softmax(z_T/τ_s) ∥ softmax(z_S/τ_s))
    Sesuai blueprint §3.3
    """
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_sent_logits: torch.Tensor,
                teacher_sent_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        loss = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(student_sent_logits / T, dim=-1),
            F.softmax(teacher_sent_logits.float() / T, dim=-1),
        ) * (T ** 2)
        return loss


# ─────────────────────────────────────────────────────────────
# COMPONENT 3: StudentWithSentimentHead (blueprint §3.2)
# ─────────────────────────────────────────────────────────────
class StudentWithSentimentHead(nn.Module):
    """
    XLM-R student + sentiment head (3-class polarity).
    Sesuai blueprint §3.2: sentiment_head = nn.Linear(768, 3)
    """
    def __init__(self, base_model):
        super().__init__()
        self.base          = base_model
        self.sentiment_head = nn.Linear(768, 3)  # blueprint §3.2

    def forward(self, input_ids, attention_mask,
                labels=None, output_hidden_states=False):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        cls_hidden    = out.hidden_states[-1][:, 0, :]  # (B, 768)
        sent_logits   = self.sentiment_head(cls_hidden)  # (B, 3)
        out.sent_logits = sent_logits
        out.cls_hidden  = cls_hidden
        return out


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def _get_optimizer(backbone_params, head_params=None):
    """
    LR differential following blueprint §5.2:
      backbone: 2e-5
      sentiment head + projection: 1e-4
    """
    if head_params is None:
        head_params = []

    param_groups = [
        {"params": backbone_params, "lr": config.LEARNING_RATE},
    ]
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr"    : config.LEARNING_RATE_HEAD,
        })

    if config.USE_8BIT_ADAM and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            print(f"   Optimizer : 8-bit Adam   "
                  f"(backbone lr={config.LEARNING_RATE}, "
                  f"head lr={config.LEARNING_RATE_HEAD})")
            return bnb.optim.Adam8bit(param_groups)
        except ImportError:
            print("   ️  bitsandbytes not found -> fallback to AdamW")
    from torch.optim import AdamW
    return AdamW(param_groups, weight_decay=0.01)


def _autocast():
    if not config.USE_AMP or not torch.cuda.is_available():
        return torch.cuda.amp.autocast(enabled=False)
    return torch.cuda.amp.autocast(enabled=True, dtype=config.AMP_DTYPE)


def _get_scaler():
    needs = (config.USE_AMP and torch.cuda.is_available()
             and config.AMP_DTYPE == torch.float16)
    return torch.cuda.amp.GradScaler(enabled=needs)


# ─────────────────────────────────────────────────────────────
# SANITY CHECK §8.1: Teacher Entropy
# ─────────────────────────────────────────────────────────────
def check_teacher_entropy(teacher, tokenizer, sample_texts):
    """
    Blueprint §8.1: cek apakah teacher overconfident.
    Healthy range: 0.1 – 0.4
    < 0.05 -> dark knowledge will not transfer
    """
    teacher.eval()
    all_entropy = []
    with torch.no_grad():
        for i in range(0, min(len(sample_texts), 50), 8):
            batch = sample_texts[i:i+8]
            enc   = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=config.MAX_LEN
            ).to(device)
            logits  = teacher(**enc).logits
            probs   = F.softmax(logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(-1)
            all_entropy.extend(entropy.cpu().tolist())

    mean_entropy = sum(all_entropy) / len(all_entropy)
    status = (" healthy" if 0.1 <= mean_entropy <= 0.4
              else "️  LOW - dark knowledge may not transfer"
              if mean_entropy < 0.1 else "️  HIGH")
    print(f"\n   [Sanity §8.1] Teacher entropy: {mean_entropy:.4f} {status}")
    return mean_entropy


# ─────────────────────────────────────────────────────────────
# TEACHER TRAINING
# ─────────────────────────────────────────────────────────────
def train_teacher(train_en, seed=None, label_smoothing=None):
    """
    Train veracity teacher (RoBERTa-Large).
    FIX #7: Optional label_smoothing (e.g. 0.1) to increase teacher entropy
    and improve dark knowledge transfer.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    ls = label_smoothing or getattr(config, "LABEL_SMOOTHING", 0.0)

    print(f"\n{'─'*60}")
    print(f"  [1/3] Training Veracity Teacher (RoBERTa-Large)")
    print(f"  Device: {device} | BF16: {config.USE_AMP} | LabelSmooth: {ls}")
    print(f"{'─'*60}")

    tokenizer = AutoTokenizer.from_pretrained(config.TEACHER_MODEL)
    teacher   = RobertaForSequenceClassification.from_pretrained(
        config.TEACHER_MODEL, num_labels=config.NUM_LABELS
    ).to(device)
    teacher.gradient_checkpointing_enable()

    optimizer = _get_optimizer(teacher.parameters())
    scaler    = _get_scaler()

    loader = DataLoader(
        FakeNewsDataset(train_en.content, train_en.label, tokenizer),
        batch_size=config.BATCH_SIZE, shuffle=True,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    # FIX #7: Label smoothing loss (if ls > 0, use custom CE instead of HF default)
    ls_criterion = (nn.CrossEntropyLoss(label_smoothing=ls)
                    if ls > 0 else None)

    teacher.train()
    for epoch in range(config.TEACHER_EPOCHS):
        total, cnt = 0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            with _autocast():
                ids  = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                lbls = batch["labels"].to(device, non_blocking=True)

                if ls_criterion is not None:
                    # Forward without labels to get logits, then custom loss
                    out  = teacher(ids, attention_mask=mask)
                    loss_val = ls_criterion(out.logits, lbls)
                else:
                    out  = teacher(ids, attention_mask=mask, labels=lbls)
                    loss_val = out.loss
                loss = loss_val / config.ACC_STEPS
            scaler.scale(loss).backward()
            total += loss_val.item(); cnt += 1
            if (i + 1) % config.ACC_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        print(f"   Epoch {epoch+1}/{config.TEACHER_EPOCHS} "
              f"| Loss: {total/cnt:.4f}")

    # Sanity check §8.1
    sample_texts = train_en.content.tolist()[:50]
    check_teacher_entropy(teacher, tokenizer, sample_texts)

    os.makedirs(os.path.dirname(config.TEACHER_CHECKPOINT), exist_ok=True)
    torch.save(teacher.state_dict(), config.TEACHER_CHECKPOINT)
    print(f" Teacher saved -> '{config.TEACHER_CHECKPOINT}'")
    return teacher, tokenizer


# ─────────────────────────────────────────────────────────────
# LOAD SENTIMENT TEACHER (blueprint §3.1)
# ─────────────────────────────────────────────────────────────
def load_sentiment_teacher():
    """
    Load Indonesian sentiment model sebagai auxiliary teacher.
    Blueprint §3.1: mdhugol/indonesia-bert-sentiment-classification
    This model is FROZEN - only used for forward pass.
    """
    print(f"\n   Loading Sentiment Teacher: {config.SENTIMENT_MODEL}")
    sent_tokenizer = AutoTokenizer.from_pretrained(config.SENTIMENT_MODEL)
    sent_teacher   = AutoModelForSequenceClassification.from_pretrained(
        config.SENTIMENT_MODEL
    )
    # Pindah ke CPU untuk hemat VRAM (blueprint §11: risk mitigation)
    sent_teacher.eval()
    for p in sent_teacher.parameters():
        p.requires_grad = False
    print(f"    Sentiment teacher loaded (CPU mode - save VRAM)")
    return sent_teacher, sent_tokenizer


# ─────────────────────────────────────────────────────────────
# SG-CLKD DISTILLATION (blueprint §5)
# ─────────────────────────────────────────────────────────────
def distill_student(teacher, tokenizer_t, train_en,
                    mode="full", temperature=None,
                    seed=None, zs_mode=False):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Konfigurasi following blueprint §4.2
    if zs_mode:
        T            = config.TEMPERATURE_ZS
        alpha_task   = config.ALPHA_TASK_ZS
        alpha_logit  = config.ALPHA_LOGIT_ZS
        alpha_cont   = config.ALPHA_CONT_ZS
        alpha_sent   = config.ALPHA_SENT_ZS
        n_epochs     = config.DISTILL_EPOCHS_ZS
    else:
        T            = temperature or config.TEMPERATURE
        alpha_task   = config.ALPHA_TASK
        alpha_logit  = config.ALPHA_LOGIT
        alpha_cont   = config.ALPHA_CONT
        alpha_sent   = config.ALPHA_SENT
        n_epochs     = config.DISTILL_EPOCHS
    if temperature:
        T = temperature

    print(f"\n{'─'*60}")
    print(f"  [2/3] SG-CLKD Distillation")
    print(f"  mode={mode} | T={T} | zs={zs_mode}")
    print(f"  α_task={alpha_task} α_logit={alpha_logit} "
          f"α_cont={alpha_cont} α_sent={alpha_sent}")
    print(f"  Loss: CE + KL-Divergence + InfoNCE + Sentiment-KL")
    print(f"{'─'*60}")

    # Load sentiment teacher (blueprint §3)
    sent_teacher, sent_tokenizer = load_sentiment_teacher()
    sent_loss_fn = SentimentDistillationLoss(
        temperature=config.SENTIMENT_TEMP
    )

    # Student dengan sentiment head (blueprint §3.2)
    tokenizer_s  = AutoTokenizer.from_pretrained(config.STUDENT_MODEL)
    base_model   = XLMRobertaForSequenceClassification.from_pretrained(
        config.STUDENT_MODEL, num_labels=config.NUM_LABELS
    ).to(device)
    base_model.gradient_checkpointing_enable()
    student = StudentWithSentimentHead(base_model).to(device)

    # InfoNCE contrastive loss (blueprint §2.4)
    contrastive_loss = ContrastiveDistillationLoss(
        teacher_dim=1024, student_dim=768,
        temperature=config.INFONCE_TEMP
    ).to(device)

    # LR differential following blueprint §5.2
    backbone_params = list(student.base.parameters())
    head_params     = (list(student.sentiment_head.parameters())
                       + list(contrastive_loss.parameters()))
    optimizer = _get_optimizer(backbone_params, head_params)
    scaler    = _get_scaler()

    # Freeze veracity teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    loader = DataLoader(
        FakeNewsDataset(train_en.content, train_en.label, tokenizer_s),
        batch_size=config.BATCH_SIZE, shuffle=True,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    student.train()
    total_b, part_b = 0, 0

    for epoch in range(n_epochs):
        total, cnt, part = 0, 0, 0
        optimizer.zero_grad()
        cosine_stats_epoch = []

        for i, batch in enumerate(loader):
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["labels"].to(device, non_blocking=True)

            # Student forward
            with _autocast():
                s_out = student(ids, attention_mask=mask)

            # Veracity teacher forward (no grad)
            with torch.no_grad(), _autocast():
                texts = [
                    tokenizer_s.decode(g, skip_special_tokens=True)
                    for g in batch["input_ids"]
                ]
                t_in  = tokenizer_t(
                    texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=config.MAX_LEN,
                ).to(device)
                t_out = teacher(
                    t_in["input_ids"],
                    attention_mask=t_in["attention_mask"],
                    output_hidden_states=True,
                )

            # Sentiment teacher forward (CPU, blueprint §11 risk mitigation)
            with torch.no_grad():
                sent_enc = sent_tokenizer(
                    texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=config.MAX_LEN,
                )
                sent_out = sent_teacher(**sent_enc)
                t_sent_logits = sent_out.logits  # (B, 3) on CPU

            # CLS representations (FP32 untuk stabilitas)
            s_cls    = s_out.cls_hidden.float()
            t_cls    = t_out.hidden_states[-1][:, 0, :].float()
            s_logits = s_out.logits.float()
            t_logits = t_out.logits.float()

            # ── 4 komponen loss (blueprint §4.1) ─────────────
            # 1. Task loss
            loss_task = nn.CrossEntropyLoss()(s_logits, lbls)

            # 2. Logit KD (KL-Divergence)
            loss_logit = nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
            ) * (T ** 2)

            # 3. Contrastive InfoNCE (blueprint §2)
            loss_cont = contrastive_loss(s_cls, t_cls)

            # 4. Sentiment KD (blueprint §3.3)
            s_sent = s_out.sent_logits.float()
            t_sent = t_sent_logits.float()
            loss_sent = sent_loss_fn(s_sent, t_sent.to(s_sent.device))

            # Cosine stats sanity check §8.3
            if i % 20 == 0:
                stats = contrastive_loss.cosine_stats(s_cls, t_cls)
                cosine_stats_epoch.append(stats)

            if (loss_logit.item() + loss_cont.item()) > 0.01:
                part += 1

            # ── Gabungkan sesuai mode (blueprint §4.3) ───────
            if mode == "full":
                bl = (alpha_task  * loss_task
                      + alpha_logit * loss_logit
                      + alpha_cont  * loss_cont
                      + alpha_sent  * loss_sent)
            elif mode == "no_distill":
                bl = loss_task
            elif mode == "logits_only":
                bl = 0.50 * loss_task + 0.50 * loss_logit
            elif mode == "contrast_only":
                bl = 0.50 * loss_task + 0.50 * loss_cont
            elif mode == "sent_only":
                bl = 0.50 * loss_task + 0.50 * loss_sent
            elif mode == "logits_contrast":
                bl = 0.40*loss_task + 0.30*loss_logit + 0.30*loss_cont
            elif mode == "logits_sent":
                bl = 0.40*loss_task + 0.40*loss_logit + 0.20*loss_sent
            elif mode == "contrast_sent":
                bl = 0.40*loss_task + 0.40*loss_cont  + 0.20*loss_sent
            else:
                # backward compatibility
                if mode == "features_only":
                    bl = 0.50 * loss_task + 0.50 * loss_cont
                else:
                    raise ValueError(f"Unknown mode: {mode}")

            scaler.scale(bl / config.ACC_STEPS).backward()
            total += bl.item(); cnt += 1

            if (i + 1) % config.ACC_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(student.parameters())
                    + list(contrastive_loss.parameters()), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        total_b += cnt; part_b += part

        # Print epoch stats + cosine sanity
        avg_cos = (sum(s["matched"] for s in cosine_stats_epoch)
                   / max(len(cosine_stats_epoch), 1))
        print(f"   Epoch {epoch+1}/{n_epochs} | Loss: {total/cnt:.4f} "
              f"| Active: {part}/{cnt} "
              f"| Cosine(matched): {avg_cos:.3f}")

    pct = (part_b / total_b * 100) if total_b > 0 else 0
    os.makedirs(os.path.dirname(config.STUDENT_CHECKPOINT), exist_ok=True)
    torch.save(student.state_dict(), config.STUDENT_CHECKPOINT)
    print(f" Student saved | Participation: {pct:.1f}%")
    return student, tokenizer_s, pct


# ─────────────────────────────────────────────────────────────
# FINE-TUNING (blueprint §5.1 Stage 4)
# ─────────────────────────────────────────────────────────────
def finetune_student(student, tokenizer_s, train_id, test_id):
    print(f"\n{'─'*60}")
    print(f"  [3/3] Fine-Tuning Student on Indonesian Data")
    print(f"{'─'*60}")

    student.train()
    if hasattr(student, 'base'):
        student.base.gradient_checkpointing_enable()

    backbone_params = (list(student.base.parameters())
                       if hasattr(student, 'base')
                       else list(student.parameters()))
    head_params = (list(student.sentiment_head.parameters())
                   if hasattr(student, 'sentiment_head') else [])

    optimizer = _get_optimizer(backbone_params, head_params)
    scaler    = _get_scaler()

    train_loader = DataLoader(
        FakeNewsDataset(train_id.content, train_id.label, tokenizer_s),
        batch_size=config.BATCH_SIZE, shuffle=True,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        FakeNewsDataset(test_id.content, test_id.label, tokenizer_s),
        batch_size=16,
    )

    for epoch in range(config.EPOCHS_FINETUNE):
        total, cnt = 0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            with _autocast():
                out  = student(
                    batch["input_ids"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                    labels=batch["labels"].to(device, non_blocking=True),
                )
                loss = out.loss / config.ACC_STEPS
            scaler.scale(loss).backward()
            total += out.loss.item(); cnt += 1
            if (i + 1) % config.ACC_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        print(f"   Epoch {epoch+1}/{config.EPOCHS_FINETUNE} "
              f"| Loss: {total/cnt:.4f}")

    # Evaluasi menggunakan base model untuk compatibility
    eval_model = student.base if hasattr(student, 'base') else student
    return evaluate_model(
        eval_model, test_loader,
        "SG-CLKD Fine-Tuned (Indonesian)"
    )
