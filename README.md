# Hackathon Solutions

One repository, one clean structure, three independent problems.

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

### Problem 2 — Real-Time Toxic Comment Filter with Explainability
Folder: `problem2-toxic-filter/`

Builds a fast toxicity classifier using:
- word + character TF-IDF,
- Logistic Regression,
- F1-tuned threshold,
- character n-grams for misspelled/obfuscated abuse,
- real-time latency measurement,
- token-level explanation/highlighting.

Run:
```bash
cd problem2-toxic-filter
pip install -r requirements.txt
# place Kaggle train.csv at data/train.csv
python eda.py --data data/train.csv
python train.py --data data/train.csv --max-rows 300000
streamlit run app.py
```

### Problem 3 — Natural Language Spreadsheet Analyst
Folder: `problem3-spreadsheet-analyst/`

Builds a safe natural-language CSV analyst that:
- auto-cleans nulls and malformed data,
- infers dates and numeric types,
- converts questions into a validated structured query plan,
- never executes LLM-generated Python,
- returns a one-line insight + table + Plotly chart,
- includes an offline fast-path for common NYC 311 questions.

Run:
```bash
cd problem3-spreadsheet-analyst
pip install -r requirements.txt
python download_sample.py --rows 50000
streamlit run app.py
```

Five-question smoke benchmark:
```bash
python evaluate_examples.py
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
├── problem3-spreadsheet-analyst/
│   ├── app.py
│   ├── data_loader.py
│   ├── planner.py
│   ├── executor.py
│   ├── charting.py
│   ├── download_sample.py
│   ├── evaluate_examples.py
│   ├── tests/
│   └── README.md
│
└── README.md
```

Each problem is self-contained so judges can enter a folder and run it independently.

## Datasets

- Problem 1: PatronusAI FinanceBench
- Problem 2: Jigsaw Unintended Bias in Toxicity Classification
- Problem 3: NYC Open Data — 311 Service Requests (erm2-nwe9)
