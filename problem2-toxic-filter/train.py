from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from model import MODEL_DIR, normalize_text

def best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.15, 0.81, 0.01):
        f1 = f1_score(y_true, probs >= t)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/train.csv"))
    p.add_argument("--max-rows", type=int, default=300000)
    a = p.parse_args()

    df = pd.read_csv(a.data, usecols=["comment_text", "target"])
    df = df.dropna(subset=["comment_text", "target"]).copy()
    df["label"] = (df.target >= 0.5).astype(int)

    if a.max_rows and len(df) > a.max_rows:
        df, _ = train_test_split(
            df, train_size=a.max_rows, stratify=df.label, random_state=42
        )

    df["text"] = df.comment_text.map(normalize_text)
    train, valid = train_test_split(
        df, test_size=0.2, stratify=df.label, random_state=42
    )

    word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.995,
        sublinear_tf=True, strip_accents="unicode", max_features=220000
    )
    char = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=2,
        max_features=180000, sublinear_tf=True
    )

    X_train = hstack(
        [word.fit_transform(train.text), char.fit_transform(train.text)],
        format="csr"
    )
    X_valid = hstack(
        [word.transform(valid.text), char.transform(valid.text)],
        format="csr"
    )

    clf = LogisticRegression(
        C=4.0, max_iter=300, class_weight="balanced", solver="liblinear"
    )
    clf.fit(X_train, train.label)

    probs = clf.predict_proba(X_valid)[:, 1]
    threshold, f1 = best_threshold(valid.label.values, probs)
    pred = probs >= threshold

    print(f"Rows: {len(df):,}")
    print(f"Toxic prevalence: {df.label.mean()*100:.2f}%")
    print(f"Best threshold: {threshold:.2f}")
    print(f"Validation F1: {f1:.4f}")
    print(classification_report(valid.label, pred, digits=4))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "word_vectorizer": word,
        "char_vectorizer": char,
        "classifier": clf,
        "threshold": threshold,
    }, MODEL_DIR / "toxicity_model.joblib")
    print("Saved artifacts/toxicity_model.joblib")

if __name__ == "__main__":
    main()
