# Hackathon — Problem 1

This repository contains only **Problem 1: Trustworthy FinanceBench RAG**.

## Problem 1 — Trustworthy FinanceBench RAG
Folder: `problem1-rag/`

Features:
- local FAISS retrieval
- hybrid semantic + lexical reranking
- grounded LLM generation
- verified document/page/chunk citations
- exact fail-closed response when evidence is insufficient
- FinanceBench answer/evidence evaluation

Run:
```bash
cd problem1-rag
pip install -r requirements.txt
python -m rag.ingest --download-financebench --limit 5
streamlit run app.py
```

Problems 2 and 3 are maintained separately in:
`https://github.com/Kaal12411/hackathon2`
