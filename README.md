# Trustworthy FinanceBench RAG

A hackathon-ready Retrieval-Augmented Generation system that answers **only from the documents provided**, cites every answer, and refuses to guess.

**Required fallback:** `I don't know based on the provided documents`

## What it does
- Parses PDF, TXT, and Markdown documents.
- Chunks text into **900 characters with 150-character overlap**.
- Creates local embeddings with **sentence-transformers/all-MiniLM-L6-v2**.
- Stores/searches vectors locally with **FAISS**.
- Retrieves top-k evidence and generates an answer with **Groq or OpenAI**.
- Rejects weak retrieval, malformed model output, or fabricated source IDs.
- Shows **document + page + chunk** citations for every supported answer.
- Includes a **Streamlit UI** for a strong hackathon demo.

## Architecture
```
Documents -> Parser -> Chunker -> Embeddings -> FAISS
                                      |
User question -> Query embedding -> Retrieve top-k
                                      |
                           Confidence guardrail
                                      |
                              LLM grounded prompt
                                      |
                         Answer + validated citations
```

## Quick start
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Set either a Groq key or OpenAI key in `.env`.

Put PDF/TXT/MD files under `data/documents/`, then:
```bash
python -m rag.ingest --documents data/documents
python -m rag.cli "What was 3M's capital expenditure in FY2018?"
streamlit run app.py
```

Optional FinanceBench source download:
```bash
python -m rag.ingest --download-financebench --limit 5
```

## Guardrails
1. Retrieval confidence threshold.
2. Grounding-only system prompt.
3. Structured JSON answer + source IDs.
4. Citation ID validation.
5. No-citation rejection.
6. Exact fail-closed phrase.

## Demo
Ask an answerable finance question, then ask something unrelated like:
`Who won the 2022 World Cup?`

The unrelated question returns exactly:
```
I don't know based on the provided documents
```
