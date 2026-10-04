import json
from openai import OpenAI
from .config import GROQ_API_KEY,GROQ_MODEL,LLM_PROVIDER,OPENAI_API_KEY,OPENAI_MODEL,UNKNOWN

SYSTEM_PROMPT=f"""You are a high-precision enterprise RAG assistant.
RULES:
1. Answer ONLY from the CONTEXT provided.
2. Never use outside knowledge, memory, assumptions, or guesses.
3. If context does not fully support the answer, return exactly: {UNKNOWN}
4. Every factual answer MUST cite one or more source IDs from context.
5. Never invent source IDs.
6. Return valid JSON only: {{"answer":"...", "source_ids":["S1"]}}
7. If unknown: {{"answer":"{UNKNOWN}", "source_ids":[]}}"""

def _client_and_model():
    if LLM_PROVIDER=="groq":
        if not GROQ_API_KEY: raise RuntimeError("GROQ_API_KEY is not set.")
        return OpenAI(api_key=GROQ_API_KEY,base_url="https://api.groq.com/openai/v1"),GROQ_MODEL
    if LLM_PROVIDER=="openai":
        if not OPENAI_API_KEY: raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAI(api_key=OPENAI_API_KEY),OPENAI_MODEL
    raise ValueError("LLM_PROVIDER must be 'groq' or 'openai'.")

def generate_answer(question,retrieved):
    context="\n\n".join(f"[{x['source_id']}] document={x['source']} page={x['page']} chunk={x['chunk_index']} similarity={x['score']:.3f}\n{x['text']}" for x in retrieved)
    client,model=_client_and_model()
    r=client.chat.completions.create(model=model,temperature=0,response_format={"type":"json_object"},messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"}])
    try: return json.loads(r.choices[0].message.content or "")
    except json.JSONDecodeError: return {"answer":UNKNOWN,"source_ids":[]}
