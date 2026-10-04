from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

from datasets import load_dataset

from rag.config import UNKNOWN
from rag.pipeline import answer_question
from rag.retriever import Retriever

OUT_DIR = Path("data/eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def token_f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return 0.0
    from collections import Counter
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)

def numeric_signature(text: str):
    return re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?", str(text))

def answer_match(pred: str, gold: str) -> bool:
    if norm(pred) == norm(gold):
        return True
    nums = numeric_signature(gold)
    if nums and all(n.replace(",","") in pred.replace(",","") for n in nums):
        return True
    return token_f1(pred, gold) >= 0.72

def gold_pages(row) -> set[int]:
    pages=set()
    for ev in row.get("evidence") or []:
        p=ev.get("evidence_page_num")
        if isinstance(p,int):
            pages.add(p)
    return pages

def retrieval_eval(limit=None):
    ds=load_dataset("PatronusAI/financebench", split="train")
    retriever=Retriever()
    rows=[]
    for i,row in enumerate(ds):
        if limit and i>=limit: break
        retrieved=retriever.search(row["question"])
        pages=gold_pages(row)
        top_pages={int(x["page"]) for x in retrieved}
        hit=bool(pages & top_pages) if pages else False
        rows.append({
            "financebench_id":row["financebench_id"],
            "question":row["question"],
            "doc_name":row["doc_name"],
            "gold_pages":sorted(pages),
            "retrieved_pages":sorted(top_pages),
            "retrieval_page_hit":hit,
        })
    summary={"n":len(rows),"retrieval_page_recall_at_k":mean([r["retrieval_page_hit"] for r in rows]) if rows else 0}
    (OUT_DIR/"retrieval_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUT_DIR/"retrieval_results.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

def full_eval(limit=None):
    ds=load_dataset("PatronusAI/financebench", split="train")
    rows=[]
    for i,row in enumerate(ds):
        if limit and i>=limit: break
        result=answer_question(row["question"])
        pred=result["answer"]
        pages=gold_pages(row)
        cited_pages={int(c["page"]) for c in result["citations"]}
        answer_ok=pred != UNKNOWN and answer_match(pred,row["answer"])
        citation_ok=bool(pages & cited_pages) if pages and cited_pages else False
        rows.append({
            "financebench_id":row["financebench_id"],
            "question":row["question"],
            "gold_answer":row["answer"],
            "predicted_answer":pred,
            "answer_correct":answer_ok,
            "citation_page_correct":citation_ok,
            "gold_pages":sorted(pages),
            "cited_pages":sorted(cited_pages),
            "decision":result.get("decision"),
        })
    summary={
        "n":len(rows),
        "answer_accuracy":mean([r["answer_correct"] for r in rows]) if rows else 0,
        "citation_page_accuracy":mean([r["citation_page_correct"] for r in rows]) if rows else 0,
        "joint_answer_and_citation_accuracy":mean([r["answer_correct"] and r["citation_page_correct"] for r in rows]) if rows else 0,
        "refusal_rate":mean([r["predicted_answer"]==UNKNOWN for r in rows]) if rows else 0,
    }
    (OUT_DIR/"full_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUT_DIR/"full_results.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

def refusal_eval():
    questions=[
        "Who won the 2022 World Cup?",
        "What is the capital of France?",
        "Write me a poem about Mars.",
        "What is tomorrow's weather?",
        "Who is the current president of the United States?",
        "What is the recipe for chocolate cake?",
        "How tall is Mount Everest?",
        "What is the latest iPhone model?",
        "Explain photosynthesis.",
        "Who painted the Mona Lisa?",
    ]
    rows=[]
    for q in questions:
        r=answer_question(q)
        rows.append({"question":q,"answer":r["answer"],"correct_refusal":r["answer"]==UNKNOWN})
    summary={"n":len(rows),"refusal_accuracy":mean([r["correct_refusal"] for r in rows])}
    (OUT_DIR/"refusal_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUT_DIR/"refusal_results.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["retrieval","full","refusal"],default="retrieval")
    p.add_argument("--limit",type=int,default=None)
    a=p.parse_args()
    if a.mode=="retrieval": retrieval_eval(a.limit)
    elif a.mode=="full": full_eval(a.limit)
    else: refusal_eval()

if __name__=="__main__":
    main()
