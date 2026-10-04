from __future__ import annotations
import argparse, json, re
from pathlib import Path
import faiss, fitz, requests
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from .config import DOCUMENT_DIR, EMBEDDING_MODEL, INDEX_DIR

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = clean_text(text)
    if not text: return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        if end < len(text):
            cut = max(chunk.rfind("\n"), chunk.rfind(". "))
            if cut > size * 0.55:
                end = start + cut + 1
                chunk = text[start:end]
        if chunk.strip(): chunks.append(chunk.strip())
        if end >= len(text): break
        start = max(end - overlap, start + 1)
    return chunks

def parse_pdf(path: Path):
    doc = fitz.open(path)
    for page_number, page in enumerate(doc, 1):
        for chunk_index, chunk in enumerate(chunk_text(page.get_text("text")), 1):
            yield {"source":path.name,"page":page_number,"chunk_index":chunk_index,"text":chunk}

def parse_text_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    for chunk_index, chunk in enumerate(chunk_text(text), 1):
        yield {"source":path.name,"page":1,"chunk_index":chunk_index,"text":chunk}

def parse_documents(directory: Path):
    rows=[]
    for path in sorted(directory.rglob("*")):
        if not path.is_file(): continue
        if path.suffix.lower()==".pdf":
            try:
                rows.extend(parse_pdf(path))
            except Exception as e:
                print(f"Skipping unreadable PDF {path.name}: {e}")
        elif path.suffix.lower() in {".txt",".md"}:
            rows.extend(parse_text_file(path))
    return rows

def build_index(chunks, index_dir: Path = INDEX_DIR):
    if not chunks: raise ValueError("No parseable document content found.")
    index_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([x["text"] for x in chunks], normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1]); index.add(embeddings)
    faiss.write_index(index, str(index_dir/"vectors.faiss"))
    (index_dir/"metadata.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks from {len(set(x['source'] for x in chunks))} documents.")

def download_financebench_sources(out_dir: Path, limit=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds=load_dataset("PatronusAI/financebench", split="train")
    unique=[]; seen=set()
    for row in ds:
        name, link=row.get("doc_name"), row.get("doc_link")
        if not name or not link or (name,link) in seen: continue
        seen.add((name,link)); unique.append((name,link))
        if limit and len(unique)>=limit: break
    downloaded=0
    skipped=0
    headers={"User-Agent":"Mozilla/5.0"}
    for i,(name,url) in enumerate(unique,1):
        safe=re.sub(r"[^A-Za-z0-9._-]+","_",name).strip("_")
        if not safe.lower().endswith(".pdf"): safe += ".pdf"
        target=out_dir/safe
        if target.exists() and target.stat().st_size > 1000:
            downloaded += 1
            continue
        print(f"[{i}/{len(unique)}] downloading {target.name}")
        try:
            r=requests.get(url,timeout=60,headers=headers)
            r.raise_for_status()
            if "pdf" not in r.headers.get("content-type","").lower() and not r.content.startswith(b"%PDF"):
                raise ValueError("source did not return a PDF")
            target.write_bytes(r.content)
            downloaded += 1
        except Exception as e:
            skipped += 1
            print(f"WARNING: skipped {target.name}: {e}")
    print(f"FinanceBench download complete: {downloaded} available, {skipped} skipped.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--documents",type=Path,default=DOCUMENT_DIR)
    p.add_argument("--download-financebench",action="store_true")
    p.add_argument("--limit",type=int,default=None)
    a=p.parse_args()
    if a.download_financebench: download_financebench_sources(a.documents,a.limit)
    build_index(parse_documents(a.documents))

if __name__=="__main__": main()
