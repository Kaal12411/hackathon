import json
from openai import OpenAI
from .config import (
    GROQ_API_KEY, GROQ_MODEL, LLM_PROVIDER,
    OPENAI_API_KEY, OPENAI_MODEL, UNKNOWN
)

SYSTEM_PROMPT = f"""You are a high-precision enterprise RAG assistant.

RULES:
1. Answer ONLY from the CONTEXT provided.
2. Never use outside knowledge, memory, assumptions, or guesses.
3. If context does not fully support the answer, return exactly: {UNKNOWN}
4. Every factual answer MUST cite one or more source IDs from context.
5. Never invent a source ID.
6. Preserve exact numbers, units, currencies, signs, percentages, and dates.
7. If a calculation is needed, only calculate from values explicitly present in context.
8. Return valid JSON only:
   {{"answer":"...", "source_ids":["S1"]}}
9. If unknown:
   {{"answer":"{UNKNOWN}", "source_ids":[]}}
"""

VERIFY_PROMPT = f"""You are a strict grounding verifier.
Given a QUESTION, a PROPOSED ANSWER, and retrieved EVIDENCE:
- Mark supported=true only if the proposed answer is fully entailed by the evidence.
- Check numbers, units, currencies, signs, dates, entities, and calculations carefully.
- Do not use outside knowledge.
- Return only source IDs that directly support the answer.
- If anything material is unsupported, ambiguous, or contradicted, supported=false.
Return JSON only:
{{"supported": true, "source_ids": ["S1"]}}
or
{{"supported": false, "source_ids": []}}
"""

def _client_and_model():
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set.")
        return (
            OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"),
            GROQ_MODEL,
        )
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAI(api_key=OPENAI_API_KEY), OPENAI_MODEL
    raise ValueError("LLM_PROVIDER must be 'groq' or 'openai'.")

def _context(retrieved):
    return "\n\n".join(
        f"[{x['source_id']}] document={x['source']} page={x['page']} "
        f"chunk={x['chunk_index']} score={x['score']:.3f}\n{x['text']}"
        for x in retrieved
    )

def _json_call(system_prompt: str, user_prompt: str) -> dict:
    client, model = _client_and_model()
    r = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt},
        ],
    )
    try:
        return json.loads(r.choices[0].message.content or "")
    except json.JSONDecodeError:
        return {}

def generate_answer(question, retrieved):
    payload = _json_call(
        SYSTEM_PROMPT,
        f"QUESTION:\n{question}\n\nCONTEXT:\n{_context(retrieved)}",
    )
    if not payload:
        return {"answer": UNKNOWN, "source_ids": []}
    return payload

def verify_answer(question: str, answer: str, retrieved: list[dict]) -> dict:
    return _json_call(
        VERIFY_PROMPT,
        f"QUESTION:\n{question}\n\nPROPOSED ANSWER:\n{answer}\n\nEVIDENCE:\n{_context(retrieved)}",
    )
