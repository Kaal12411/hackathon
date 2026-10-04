from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from model import ToxicityModel

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/train.csv"))
    p.add_argument("--sample",type=int,default=50000)
    a=p.parse_args()

    df=pd.read_csv(a.data,usecols=["comment_text","target"]).dropna()
    if a.sample and len(df)>a.sample:
        df=df.sample(a.sample,random_state=123)

    model=ToxicityModel.load()
    y=(df.target.values>=0.5).astype(int)
    pred=[]; lat=[]

    for text in df.comment_text.astype(str):
        r=model.predict(text)
        pred.append(r["label"]=="Toxic")
        lat.append(r["latency_ms"])

    print(f"F1: {f1_score(y,pred):.4f}")
    print(f"Precision: {precision_score(y,pred):.4f}")
    print(f"Recall: {recall_score(y,pred):.4f}")
    print(f"Median latency: {np.median(lat):.2f} ms")
    print(f"P95 latency: {np.percentile(lat,95):.2f} ms")
    print(f"Max latency: {np.max(lat):.2f} ms")
    print(f"Under 500ms: {np.mean(np.array(lat)<500)*100:.2f}%")

if __name__=="__main__":
    main()
