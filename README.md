# Trustworthy FinanceBench RAG

A hackathon-ready Retrieval-Augmented Generation system that answers **only from provided documents**, cites every supported answer, and refuses to guess.

> Required fallback: **I don't know based on the provided documents**

## Rubric-first design

| Rubric | Marks | How this solution targets it |
|---|---:|---|
| Correct answers + correct citations | 40 | Hybrid retrieval, exact page/chunk metadata, structured source IDs, second-pass grounding verifier |
| Correct "I don't know" behavior | 30 | Retrieval threshold + grounding-only prompt + citation validation + verifier fail-closed path |
| Demo quality / UI clarity | 20 | Streamlit dashboard with decision, score, verified citations, source excerpts, and audit trail |
| Explain design choices | 10 | Architecture and rationale documented below |

## Architecture

```
PDF/TXT/MD
   |
   v
Page-aware parser
   |
   v
900-char chunks / 150 overlap
   |
   v
MiniLM embeddings
   |
   v
FAISS local vector DB
   |
   v
Top-20 semantic candidates
   |
   v
Semantic + lexical reranking
   |
   v
Top-5 evidence chunks
   |
   +--> low confidence? --> exact "I don't know"
   |
   v
Grounded LLM answer (JSON + source IDs)
   |
   v
Independent grounding verifier
   |
   +--> unsupported / wrong citation? --> exact "I don't know"
   |
   v
Verified answer + document/page/chunk citations
```

## Why these design choices?

### Chunk size: 900 characters, overlap: 150
Financial filings often place a number, unit, period, and explanation in the same paragraph or nearby table text. Around 900 characters preserves that local context without making retrieval too broad. A 150-character overlap reduces boundary loss.

### Embeddings: all-MiniLM-L6-v2
It is free, local, fast, and strong enough for a hackathon-scale corpus. Running embeddings locally also keeps the retrieval layer independent of a paid API.

### Vector DB: FAISS
FAISS is local, free, simple to reproduce, and fast for FinanceBench-scale retrieval. It avoids infrastructure overhead while satisfying the vector-database requirement.

### Hybrid reranking
Pure semantic search can miss exact finance terminology, years, and metric names. The system retrieves semantic candidates with FAISS and adds a lightweight lexical-overlap score before selecting the final evidence.

### Two-stage answer safety
The generator is not the final authority. A second LLM pass independently checks whether the proposed answer is fully supported by the retrieved evidence, including numbers, units, dates, and calculations. If verification fails, the system refuses.

### Fail closed, not open
A weak retrieval score, missing citation, invented source ID, malformed JSON, unsupported claim, or failed verifier all resolve to the exact required fallback instead of a best guess.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add either a Groq or OpenAI key.

### Option A — use your own documents

Put PDFs/TXT/MD files in:

```
data/documents/
```

Build the index:

```bash
python -m rag.ingest --documents data/documents
```

### Option B — download FinanceBench source filings

```bash
python -m rag.ingest --download-financebench --limit 5
```

### Run the UI

```bash
streamlit run app.py
```

### CLI

```bash
python -m rag.cli "What was the company's capital expenditure in FY2018?"
```

## Best judge demo

**1. Correct-answer case**

Ask a question that is clearly supported by one of the indexed filings.

Show:
- answer,
- verified status,
- document name,
- page,
- chunk,
- supporting excerpt.

**2. Citation case**

Ask a question containing a specific year / metric / number. Expand the audit trail and show that the cited source contains the supporting evidence.

**3. Hallucination trap**

Ask something unrelated to the indexed documents, for example:

```
Who won the 2022 World Cup?
```

Expected result:

```
I don't know based on the provided documents
```

**4. Explain the trust model**

Tell the judges:

> "The answer generator is not trusted by itself. Retrieval must first clear a confidence threshold, every answer must reference retrieved source IDs, and a separate verifier checks that the answer is actually supported. Any failure returns the required I-don't-know response."

## Project structure

```
app.py
rag/
  config.py
  ingest.py
  retriever.py
  generator.py
  pipeline.py
  cli.py
tests/
  test_guardrails.py
data/
  documents/
```

## Run tests

```bash
pytest -q
```

The core guardrail tests verify that missing citations, invented citations, and explicit unknown answers fail closed.
