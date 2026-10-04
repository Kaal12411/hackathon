import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"
DOCUMENT_DIR = ROOT / "data" / "documents"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.35"))
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", "0.18"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

UNKNOWN = "I don't know based on the provided documents"
