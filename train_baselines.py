"""
train_baselines.py - Baseline 1 (XLM-R), Baseline 2 (IndoBERT), Baseline 3 (Translate-Train)
Optimized for RTX 4070 Ti / Ubuntu 24.04:
   BF16 autocast
   8-bit Adam
   gradient clipping
   pin_memory + num_workers=4
"""

import time, os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    XLMRobertaForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm

import config
from utils import FakeNewsDataset, evaluate_model, clean_gpu, device


def _get_optimizer(params):
    if config.USE_8BIT_ADAM and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            return bnb.optim.Adam8bit(params, lr=config.LEARNING_RATE)
        except ImportError:
            pass
    from torch.optim import AdamW
    return AdamW(params, lr=config.LEARNING_RATE)


def _autocast():
    if not config.USE_AMP or not torch.cuda.is_available():
        return torch.cuda.amp.autocast(enabled=False)
    return torch.cuda.amp.autocast(enabled=True, dtype=config.AMP_DTYPE)


def _get_scaler():
    needs = (config.USE_AMP and torch.cuda.is_available()
             and config.AMP_DTYPE == torch.float16)
    return torch.cuda.amp.GradScaler(enabled=needs)


def _train_loop(model, loader, epochs, label=""):
    """Loop training generik dengan BF16 + gradient clipping."""
    optimizer = _get_optimizer(model.parameters())
    scaler    = _get_scaler()
    model.train()

    for epoch in range(epochs):
        total, cnt = 0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            with _autocast():
                out  = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(
                        device, non_blocking=True),
                    labels=batch["labels"].to(device, non_blocking=True),
                )
                loss = out.loss / config.ACC_STEPS

            scaler.scale(loss).backward()
            total += out.loss.item(); cnt += 1

            if (i + 1) % config.ACC_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        print(f"   {label} Epoch {epoch+1}/{epochs} "
              f"| Loss: {total/cnt:.4f}")


def _make_loader(texts, labels, tokenizer, batch_size=16, shuffle=False):
    return DataLoader(
        FakeNewsDataset(texts, labels, tokenizer),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.NUM_WORKERS > 0,
    )


# ── BASELINE 1: XLM-R Zero-Shot ───────────────────────────────
def train_baseline1(train_en, test_id):
    print(f"\n{'='*50}")
    print(f"  BASELINE 1 - XLM-R Zero-Shot")
    print(f"  Device: {device} | BF16: {config.USE_AMP}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL)
    model     = XLMRobertaForSequenceClassification.from_pretrained(
        config.STUDENT_MODEL, num_labels=config.NUM_LABELS
    ).to(device)

    loader = _make_loader(
        train_en.content, train_en.label, tokenizer,
        batch_size=16, shuffle=True
    )
    _train_loop(model, loader, config.BASELINE_EPOCHS, "B1")

    test_loader = _make_loader(test_id.content, test_id.label, tokenizer)
    metrics = evaluate_model(model, test_loader, "Baseline 1 - XLM-R Zero-Shot")
    del model; clean_gpu()
    return metrics


# ── BASELINE 2: IndoBERT Supervised ───────────────────────────
def train_baseline2(train_id, test_id):
    print(f"\n{'='*50}")
    print(f"  BASELINE 2 - IndoBERT Supervised")
    print(f"  Device: {device} | BF16: {config.USE_AMP}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(config.INDOBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        config.INDOBERT_MODEL, num_labels=config.NUM_LABELS
    ).to(device)

    loader = _make_loader(
        train_id.content, train_id.label, tokenizer,
        batch_size=16, shuffle=True
    )
    _train_loop(model, loader, config.BASELINE_EPOCHS, "B2")

    test_loader = _make_loader(test_id.content, test_id.label, tokenizer)
    metrics = evaluate_model(model, test_loader, "Baseline 2 - IndoBERT Supervised")
    del model; clean_gpu()
    return metrics


# ── BASELINE 3: Translate-Train ───────────────────────────────
def train_baseline_translate(train_en, test_id):
    print(f"\n{'='*50}")
    print(f"  BASELINE 3 - Translate-Train")
    print(f"  Device: {device} | BF16: {config.USE_AMP}")
    print(f"{'='*50}")

    cache_path = os.path.join(config.RESULTS_DIR, "translated_train.csv")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        df_translated = pd.read_csv(cache_path)
        print(f" Loaded cached translations: {cache_path}")
        # ── FIX #1: Ensure label mapping consistency ──────────────
        label_map = {"real": 0, "fake": 1, "Real": 0, "Fake": 1, 0: 0, 1: 1}
        df_translated["label"] = df_translated["label"].map(label_map)
        df_translated = df_translated.dropna(subset=["content", "label"])
        df_translated["label"] = df_translated["label"].astype(int)
        print(f"   Label dist: {df_translated['label'].value_counts().to_dict()}")
    else:
        try:
            from deep_translator import GoogleTranslator
        except ImportError:
            print("️  pip install deep-translator")
            return None

        translator = GoogleTranslator(source="en", target="id")
        texts, translated = train_en["content"].tolist(), []
        for i in tqdm(range(len(texts)), ncols=80, desc="Translating"):
            try:
                r = translator.translate(str(texts[i])[:4500])
                translated.append(r if r else str(texts[i]))
            except Exception:
                translated.append(str(texts[i]))
            if (i + 1) % 100 == 0:
                time.sleep(1)

        df_translated = pd.DataFrame({
            "content": translated,
            "label"  : train_en["label"].values,
        })
        df_translated.to_csv(cache_path, index=False)

    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL)
    model     = XLMRobertaForSequenceClassification.from_pretrained(
        config.STUDENT_MODEL, num_labels=config.NUM_LABELS
    ).to(device)

    loader = _make_loader(
        df_translated.content, df_translated.label, tokenizer,
        batch_size=config.BATCH_SIZE, shuffle=True
    )
    _train_loop(model, loader, config.BASELINE_EPOCHS, "B3")

    test_loader = _make_loader(test_id.content, test_id.label, tokenizer)
    metrics = evaluate_model(model, test_loader, "Baseline 3 - Translate-Train")
    del model; clean_gpu()
    return metrics


# ── BASELINE 4: Translate-Test (FIX #2) ──────────────────────
def train_baseline_translate_test(teacher, tokenizer_t, test_id):
    """
    FIX #2 (D1 Translate-Test baseline):
    Translate Indonesian test set → English, then classify with
    the frozen English teacher directly. No training on translated data.
    This is the strongest cross-lingual baseline.
    """
    print(f"\n{'='*50}")
    print(f"  BASELINE 4 - Translate-Test")
    print(f"  (Translate ID test → EN, classify with frozen teacher)")
    print(f"  Device: {device}")
    print(f"{'='*50}")

    cache_path = os.path.join(config.RESULTS_DIR, "translated_test.csv")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        df_translated = pd.read_csv(cache_path)
        print(f" Loaded cached test translations: {cache_path}")
    else:
        try:
            from deep_translator import GoogleTranslator
        except ImportError:
            print("  pip install deep-translator")
            return None

        translator = GoogleTranslator(source="id", target="en")
        texts = test_id["content"].tolist()
        translated = []
        for i in tqdm(range(len(texts)), ncols=80, desc="Translating test"):
            try:
                r = translator.translate(str(texts[i])[:4500])
                translated.append(r if r else str(texts[i]))
            except Exception:
                translated.append(str(texts[i]))
            if (i + 1) % 50 == 0:
                time.sleep(1)

        df_translated = pd.DataFrame({
            "content": translated,
            "label":   test_id["label"].values,
        })
        df_translated.to_csv(cache_path, index=False)
        print(f" Translations saved: {cache_path}")

    # Ensure labels are int
    label_map = {"real": 0, "fake": 1, "Real": 0, "Fake": 1, 0: 0, 1: 1}
    df_translated["label"] = df_translated["label"].map(label_map).astype(int)

    # Evaluate frozen teacher on translated test set
    teacher.eval()
    test_loader = _make_loader(
        df_translated.content, df_translated.label, tokenizer_t,
        batch_size=16, shuffle=False
    )
    metrics = evaluate_model(teacher, test_loader, "Baseline 4 - Translate-Test")
    return metrics
