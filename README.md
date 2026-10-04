# Hackathon Solutions

One repository, one clean structure, multiple independent problems.

## Problems

### Problem 1 — Trustworthy FinanceBench RAG
Folder: `problem1-rag/`

Builds a Retrieval-Augmented Generation assistant that:
- answers only from provided financial documents,
- uses local FAISS retrieval,
- returns document/page/chunk citations,
- fails closed with **"I don't know based on the provided documents"**,
- uses FinanceBench ground-truth answers and evidence pages for evaluation.

Run:
```bash
cd problem1-rag
pip install -r requirements.txt
python -m rag.ingest --download-financebench --limit 5
streamlit run app.py
```

FinanceBench evaluation:
```bash
python evaluate_financebench.py --mode retrieval --limit 20
python evaluate_financebench.py --mode full --limit 20
python evaluate_financebench.py --mode refusal
streamlit run eval_app.py
```

### Problem 2 — Real-Time Toxic Comment Filter with Explainability
Folder: `problem2-toxic-filter/`

Builds a fast toxicity classifier using:
- word TF-IDF,
- character TF-IDF for obfuscated/misspelled abuse,
- Logistic Regression,
- F1-tuned decision threshold,
- real-time latency measurement,
- token-level explanation/highlighting.

Run:
```bash
cd problem2-toxic-filter
pip install -r requirements.txt
# place Kaggle train.csv at data/train.csv
python eda.py --data data/train.csv
python train.py --data data/train.csv --max-rows 300000
python evaluate.py --data data/train.csv --sample 50000
streamlit run app.py
```

## Repository layout

```
hackathon/
├── problem1-rag/
│   ├── app.py
│   ├── eval_app.py
│   ├── evaluate_financebench.py
│   ├── rag/
│   ├── tests/
│   └── README.md
│
├── problem2-toxic-filter/
│   ├── app.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── eda.py
│   ├── tests/
│   └── README.md
│
└── README.md
```

Each problem is self-contained so judges can enter a folder and run it independently.

## Datasets

- Problem 1: PatronusAI FinanceBench
- Problem 2: Jigsaw Unintended Bias in Toxicity Classification

Future problems should be added as `problem3-.../`, `problem4-.../`, etc.
