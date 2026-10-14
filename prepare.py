"""
data/prepare.py
===============
Source-Blind Preprocessing Pipeline (C1 - Blueprint §6.11)

6 categories of source markers removed:
  1. Nama portal berita real   (Kompas.com, Tempo.co, detikXxx, CNN Indonesia …)
  2. Label bracket fact-checker ([SALAH], [HOAKS], [KLARIFIKASI] …)
  3. Template fact-checker     (Klaim:, Penjelasan:, TurnBackHoax, Mafindo …)
  4. Byline & copyright        (Editor:, Reporter:, © 2024 …)
  5. URL                       (https://, www., bit.ly/)
  6. Timestamp source-spesifik (HH:MM WIB / tanggal panjang)

Output:
    data/train_english_blinded.csv
    data/test_indonesia_blinded.csv
    results/source_blind_validation.json

Cara pakai:
    python data/prepare.py
"""

import re, json, os
import pandas as pd
from typing import List, Dict

# ── PATH ──────────────────────────────────────────────────────
TRAIN_INPUT  = "data/train_english_clean.xlsx"
TEST_INPUT   = "data/test_indonesia_clean.xlsx"
TRAIN_OUTPUT = "data/train_english_blinded.csv"
TEST_OUTPUT  = "data/test_indonesia_blinded.csv"
REPORT_JSON  = "results/source_blind_validation.json"

os.makedirs("data",    exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── 6-KATEGORI SOURCE PATTERNS (following blueprint §6.11) ───────
SOURCE_PATTERNS: List[str] = [

    # Kategori 1 - Nama portal berita real
    r"\(KOMPAS\.com\)", r"KOMPAS\.com", r"Kompas\.com", r"kompas\.com",
    r"Tempo\.co", r"tempo\.co",
    r"detik\.com", r"detik[A-Za-z]+",
    r"CNN\s+Indonesia",
    r"Tribunnews\.com", r"tribunnews\.com",
    r"Antaranews\.com", r"antaranews\.com",
    r"Liputan6\.com",   r"liputan6\.com",
    r"Merdeka\.com",    r"merdeka\.com",
    r"Okezone\.com",    r"okezone\.com",
    r"Republika\.co\.id", r"republika\.co\.id",
    r"Suara\.com",      r"suara\.com",

    # Kategori 2 - Label bracket fact-checker
    r"\[SALAH\]", r"\[HOAKS\]", r"\[KLARIFIKASI\]", r"\[PENIPUAN\]",
    r"\[DISINFORMASI\]", r"\[MISINFORMASI\]", r"\[CEK\s*FAKTA\]",
    r"\[FAKTA\]",  r"\[BENAR\]",
    r"\bHOAKS\b",  r"\bSALAH\b",

    # Kategori 3 - Template & nama platform fact-checker
    r"Klaim\s*:", r"Penjelasan\s*:", r"Hasil\s+Cek\s+Fakta\s*:",
    r"Narasi\s*:", r"Fakta\s*:", r"Klarifikasi\s*:",
    r"TurnBackHoax", r"Turn\s+Back\s+Hoax",
    r"Mafindo", r"Kominfo\s+Hoaks\s+Buster",
    r"Cek\s+Fakta\s+Liputan6", r"Cek\s+Fakta",

    # Kategori 4 - Byline & copyright
    r"Editor\s*:\s*[^\n]*",
    r"Reporter\s*:\s*[^\n]*",
    r"Penulis\s*:\s*[^\n]*",
    r"Kontributor\s*:\s*[^\n]*",
    r"Sumber\s*:\s*[^\n]*",
    r"\(c\)\s*\d{4}", r"©\s*\d{4}[^\n]*",
    r"Copyright\s+\d{4}[^\n]*",
    r"All\s+rights\s+reserved",
    r"Hak\s+cipta\s+dilindungi",

    # Kategori 5 - URL
    r"https?://[^\s<>\"]+",
    r"www\.[^\s<>\"]+",
    r"bit\.ly/[^\s]+",

    # Kategori 6 - Timestamp source-spesifik
    r"\d{1,2}:\d{2}\s+(?:WIB|WITA|WIT)",
    r"\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|"
    r"September|Oktober|November|Desember)\s+\d{4}\s+\d{1,2}:\d{2}",
]


def source_blind(text: str) -> str:
    """Hapus semua source marker dari satu teks."""
    if not isinstance(text, str):
        return ""
    for pattern in SOURCE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def source_blind_title(text: str) -> str:
    """Hapus source tag dalam kurung di akhir title.
    Contoh: 'Judul Berita (Kompas TV)' -> 'Judul Berita'
    """
    if not isinstance(text, str):
        return ""
    portal = (
        r"Kompas(?:\s+TV)?|Tempo|Detik[A-Za-z]*|CNN\s+Indonesia|"
        r"Liputan6|Antara(?:news)?|Tribun[A-Za-z]*|Mafindo|Merdeka|"
        r"Okezone|Republika|Suara"
    )
    text = re.sub(
        rf"\s*\([^)]*(?:{portal})[^)]*\)\s*$", "", text, flags=re.IGNORECASE
    )
    return text.strip()


def validate_effectiveness(
    df_orig: pd.DataFrame,
    df_blind: pd.DataFrame,
    col_orig:  str = "content",
    col_blind: str = "content_blinded",
) -> Dict:
    """
    Sesuai blueprint §6.11 validate_effectiveness().
    Hitung removal rate untuk setiap marker kuat.
    """
    strong_markers = {
        "[SALAH]"       : r"\[SALAH\]",
        "[HOAKS]"       : r"\[HOAKS\]",
        "KOMPAS.com"    : r"KOMPAS\.com",
        "TurnBackHoax"  : r"TurnBackHoax",
        "Mafindo"       : r"Mafindo",
        "URL (https)"   : r"https?://",
        "Copyright ©"   : r"©",
        "Editor:"       : r"Editor\s*:",
        "Klaim:"        : r"Klaim\s*:",
        "WIB timestamp" : r"\d{1,2}:\d{2}\s+WIB",
    }
    report = {}
    for name, pattern in strong_markers.items():
        before = int(df_orig[col_orig].astype(str)
                     .str.contains(pattern, case=False, na=False, regex=True).sum())
        after  = int(df_blind[col_blind].astype(str)
                     .str.contains(pattern, case=False, na=False, regex=True).sum())
        rate   = 1 - after / max(before, 1)
        status = "OK" if rate >= 0.95 else "WARN"
        report[name] = {
            "before": before, "after": after,
            "removal_rate": round(rate, 4), "status": status,
        }
        icon = "" if status == "OK" else "️ "
        print(f"  {icon} {name:22s}: {before:5d} -> {after:3d}  (removal: {rate:.1%})")
    return report


def prepare_dataset(input_path: str, output_path: str, name: str):
    """Load -> blind title & content -> simpan CSV. Return (df_orig, df_blind)."""
    print(f"\n[{name}] Loading: {input_path}")
    df = pd.read_excel(input_path) if input_path.endswith(".xlsx") \
         else pd.read_csv(input_path)
    print(f"  Shape  : {df.shape}")
    print(f"  Label  : {df['label'].value_counts().to_dict()}")

    df_blind = df.copy()
    df_blind["title_blinded"]   = df["title"].apply(source_blind_title)
    df_blind["content_blinded"] = df["content"].apply(source_blind)
    # Combined column - used as model input
    df_blind["text_blinded"] = (
        df_blind["title_blinded"].fillna("") + " "
        + df_blind["content_blinded"].fillna("")
    ).str.strip()

    df_blind.to_csv(output_path, index=False, encoding="utf-8-sig")
    changed = (df["content"].astype(str) != df_blind["content_blinded"]).sum()
    print(f"   Saved -> {output_path}  (content changed: {changed:,} rows)")
    return df, df_blind


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SOURCE-BLIND PREPROCESSING (C1 - Blueprint Sec 6.11)")
    print("=" * 60)

    df_tr_orig, df_tr_blind = prepare_dataset(TRAIN_INPUT, TRAIN_OUTPUT, "TRAIN ENGLISH")
    print("\n  Validate removal rate - TRAIN:")
    report_train = validate_effectiveness(df_tr_orig, df_tr_blind)

    df_te_orig, df_te_blind = prepare_dataset(TEST_INPUT, TEST_OUTPUT, "TEST INDONESIA")
    print("\n  Validate removal rate - TEST:")
    report_test = validate_effectiveness(df_te_orig, df_te_blind)

    validation = {
        "train_english"  : report_train,
        "test_indonesia" : report_test,
        "train_stats"    : {
            "total": len(df_tr_blind),
            "real" : int((df_tr_blind["label"] == "real").sum()),
            "fake" : int((df_tr_blind["label"] == "fake").sum()),
        },
        "test_stats"     : {
            "total": len(df_te_blind),
            "real" : int((df_te_blind["label"] == "real").sum()),
            "fake" : int((df_te_blind["label"] == "fake").sum()),
        },
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)

    print(f"\n   Validation report -> {REPORT_JSON}")
    print("=" * 60)
    print("  DONE - Use column 'text_blinded' as model input")
    print("=" * 60)
