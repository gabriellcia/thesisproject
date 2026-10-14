"""run_translate_train.py - Standalone Translate-Train Baseline."""
import json, os, time
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import XLMRobertaForSequenceClassification, AutoTokenizer
import config
from utils import prepare_datasets, FakeNewsDataset, evaluate_model, clean_gpu, print_gpu_report, device

def main():
    clean_gpu()
    print("=" * 55 + "\n  BASELINE 3: TRANSLATE-TRAIN\n" + "=" * 55)
    data = prepare_datasets(); train_en = data["train_en"].copy()
    cache_path = os.path.join(config.RESULTS_DIR, "translated_train.csv")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    if os.path.exists(cache_path):
        df_translated = pd.read_csv(cache_path)
        print(f" Loaded cached translations")
        # FIX #1: Ensure label mapping consistency
        label_map = {"real": 0, "fake": 1, "Real": 0, "Fake": 1, 0: 0, 1: 1}
        df_translated["label"] = df_translated["label"].map(label_map)
        df_translated = df_translated.dropna(subset=["content", "label"])
        df_translated["label"] = df_translated["label"].astype(int)
        print(f"   Label dist: {df_translated['label'].value_counts().to_dict()}")
    else:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="en", target="id")
        texts, translated = train_en["content"].tolist(), []
        for i in tqdm(range(len(texts)), ncols=80):
            try:
                r = translator.translate(str(texts[i])[:4500]); translated.append(r if r else str(texts[i]))
            except Exception: translated.append(str(texts[i]))
            if (i + 1) % 100 == 0: time.sleep(1)
        df_translated = pd.DataFrame({"content": translated, "label": train_en["label"].values})
        df_translated.to_csv(cache_path, index=False)
    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL)
    model     = XLMRobertaForSequenceClassification.from_pretrained(config.STUDENT_MODEL, num_labels=config.NUM_LABELS).to(device)
    loader    = DataLoader(FakeNewsDataset(df_translated.content, df_translated.label, tokenizer), batch_size=config.BATCH_SIZE, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    model.train()
    for epoch in range(config.BASELINE_EPOCHS):
        total, cnt = 0, 0; optimizer.zero_grad()
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}", ncols=80)):
            out  = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            loss = out.loss / config.ACC_STEPS; loss.backward()
            total += out.loss.item(); cnt += 1
            if (i + 1) % config.ACC_STEPS == 0: optimizer.step(); optimizer.zero_grad()
        optimizer.step(); optimizer.zero_grad()
        print(f"Epoch {epoch+1} | Loss: {total/cnt:.4f}")
    tl = DataLoader(FakeNewsDataset(data["test_id"].content, data["test_id"].label, tokenizer), batch_size=16)
    metrics = evaluate_model(model, tl, "Baseline 3: Translate-Train")
    with open(os.path.join(config.RESULTS_DIR, "translate_train_results.json"), "w") as f:
        json.dump({"method": "translate-train", **metrics}, f, indent=2)
    print(f" Saved translate_train_results.json"); del model; clean_gpu(); print_gpu_report()

if __name__ == "__main__":
    main()
