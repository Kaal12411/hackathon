import json
import re
import faiss
from sentence_transformers import SentenceTransformer
from .config import CANDIDATE_K, EMBEDDING_MODEL, INDEX_DIR, LEXICAL_WEIGHT, TOP_K

_STOP = {
    "the","a","an","and","or","of","to","in","on","for","with","was","were",
    "is","are","what","which","how","did","does","do","from","by","at","as"
}

def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[A-Za-z0-9%$.-]+", text.lower())
        if len(t) > 1 and t not in _STOP
    }

def _lexical_score(question: str, text: str) -> float:
    q = _tokens(question)
    if not q:
        return 0.0
    d = _tokens(text)
    return len(q & d) / len(q)

class Retriever:
    def __init__(self):
        ip, mp = INDEX_DIR/"vectors.faiss", INDEX_DIR/"metadata.json"
        if not ip.exists() or not mp.exists():
            raise FileNotFoundError(
                "Index not found. Run: python -m rag.ingest --documents data/documents"
            )
        self.index = faiss.read_index(str(ip))
        self.metadata = json.loads(mp.read_text(encoding="utf-8"))
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, question: str, k: int = TOP_K):
        q = self.model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        candidate_k = min(max(k, CANDIDATE_K), self.index.ntotal)
        scores, ids = self.index.search(q, candidate_k)

        candidates = []
        for semantic, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            lexical = _lexical_score(question, item["text"])
            hybrid = float(semantic) + (LEXICAL_WEIGHT * lexical)
            item["semantic_score"] = float(semantic)
            item["lexical_score"] = lexical
            item["score"] = hybrid
            candidates.append(item)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        out = candidates[:k]
        for i, item in enumerate(out, 1):
            item["source_id"] = f"S{i}"
        return out
