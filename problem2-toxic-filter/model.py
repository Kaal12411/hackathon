from __future__ import annotations
import html
import re
import time
from pathlib import Path
import joblib
from scipy.sparse import hstack

MODEL_DIR = Path("artifacts")
MODEL_PATH = MODEL_DIR / "toxicity_model.joblib"

def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

class ToxicityModel:
    def __init__(self, bundle: dict):
        self.word_vectorizer = bundle["word_vectorizer"]
        self.char_vectorizer = bundle["char_vectorizer"]
        self.classifier = bundle["classifier"]
        self.threshold = float(bundle["threshold"])

    @classmethod
    def load(cls, path: Path = MODEL_PATH):
        return cls(joblib.load(path))

    def _features(self, texts):
        clean = [normalize_text(t) for t in texts]
        word = self.word_vectorizer.transform(clean)
        char = self.char_vectorizer.transform(clean)
        return hstack([word, char], format="csr"), clean

    def predict(self, text: str) -> dict:
        started = time.perf_counter()
        X, clean = self._features([text])
        prob = float(self.classifier.predict_proba(X)[0, 1])
        label = "Toxic" if prob >= self.threshold else "Non-Toxic"
        explanation = self.explain(clean[0], base_probability=prob)
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "text": clean[0],
            "probability": prob,
            "label": label,
            "threshold": self.threshold,
            "latency_ms": latency_ms,
            "explanation": explanation,
        }

    def explain(self, text: str, base_probability: float | None = None, top_k: int = 12):
        tokens = re.findall(r"\b\w+\b", text)
        if not tokens:
            return {"top_features": [], "token_scores": {}}

        if base_probability is None:
            X, _ = self._features([text])
            base_probability = float(self.classifier.predict_proba(X)[0, 1])

        unique_tokens, seen = [], set()
        for tok in tokens[:80]:
            key = tok.lower()
            if key not in seen:
                seen.add(key)
                unique_tokens.append(tok)

        variants = [
            re.sub(rf"\b{re.escape(tok)}\b", " ", text, count=1, flags=re.I)
            for tok in unique_tokens
        ]
        Xv, _ = self._features(variants)
        probs = self.classifier.predict_proba(Xv)[:, 1]

        scored = []
        for tok, p in zip(unique_tokens, probs):
            contribution = max(0.0, float(base_probability - p))
            if contribution > 0:
                scored.append((tok, contribution))

        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[:top_k]
        return {
            "top_features": best,
            "token_scores": {tok.lower(): score for tok, score in best},
        }

def render_highlighted_html(text: str, token_scores: dict) -> str:
    max_score = max(token_scores.values(), default=0.0)
    parts = re.findall(r"\w+|\W+", text)
    out = []
    for part in parts:
        key = part.lower()
        escaped = html.escape(part)
        if key in token_scores and max_score > 0:
            alpha = 0.18 + 0.62 * (token_scores[key] / max_score)
            out.append(
                f'<span style="background: rgba(255,70,70,{alpha:.3f});'
                f'padding:2px 3px;border-radius:4px;font-weight:600">{escaped}</span>'
            )
        else:
            out.append(escaped)
    return "".join(out)
