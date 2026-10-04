import json, faiss
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL, INDEX_DIR, TOP_K

class Retriever:
    def __init__(self):
        ip, mp = INDEX_DIR/"vectors.faiss", INDEX_DIR/"metadata.json"
        if not ip.exists() or not mp.exists():
            raise FileNotFoundError("Index not found. Run: python -m rag.ingest --documents data/documents")
        self.index=faiss.read_index(str(ip))
        self.metadata=json.loads(mp.read_text(encoding="utf-8"))
        self.model=SentenceTransformer(EMBEDDING_MODEL)
    def search(self, question: str, k: int = TOP_K):
        q=self.model.encode([question],normalize_embeddings=True,convert_to_numpy=True).astype("float32")
        scores, ids=self.index.search(q,min(k,self.index.ntotal))
        out=[]
        for score, idx in zip(scores[0],ids[0]):
            if idx<0: continue
            item=dict(self.metadata[idx]); item["score"]=float(score); item["source_id"]=f"S{len(out)+1}"; out.append(item)
        return out
