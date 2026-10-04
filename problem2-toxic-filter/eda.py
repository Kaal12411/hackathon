from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/train.csv"))
    p.add_argument("--sample", type=int, default=200000)
    a = p.parse_args()

    df = pd.read_csv(a.data, usecols=["comment_text", "target"])
    df = df.dropna(subset=["comment_text", "target"]).copy()
    if a.sample and len(df) > a.sample:
        df = df.sample(a.sample, random_state=42)

    df["label"] = (df.target >= 0.5).astype(int)
    df["chars"] = df.comment_text.astype(str).str.len()
    df["words"] = df.comment_text.astype(str).str.split().str.len()

    print("=== QUICK EDA ===")
    print(f"Rows analyzed: {len(df):,}")
    print(f"Toxic rate: {df.label.mean()*100:.2f}%")
    print(f"Non-toxic rate: {(1-df.label.mean())*100:.2f}%")
    print(f"Median chars: {df.chars.median():.0f}")
    print(f"P95 chars: {df.chars.quantile(.95):.0f}")
    print(f"Median words: {df.words.median():.0f}")
    print("\nClass counts:")
    print(df.label.value_counts().rename(index={0:"Non-Toxic",1:"Toxic"}))
    print("\nPreprocessing choices:")
    print("- keep punctuation/casing signal available to char n-grams")
    print("- normalize whitespace and URLs")
    print("- use class_weight='balanced' instead of deleting majority data")
    print("- tune decision threshold on validation F1")

if __name__ == "__main__":
    main()
