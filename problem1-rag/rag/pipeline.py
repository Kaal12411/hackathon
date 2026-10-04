from .config import MIN_SIMILARITY, UNKNOWN
from .generator import generate_answer, verify_answer
from .retriever import Retriever

def validate_model_output(payload, retrieved):
    answer = str(payload.get("answer","")).strip()
    source_ids = payload.get("source_ids", [])
    if not answer or answer == UNKNOWN:
        return UNKNOWN, []
    if not isinstance(source_ids, list) or not source_ids:
        return UNKNOWN, []
    allowed = {x["source_id"]: x for x in retrieved}
    if any(s not in allowed for s in source_ids):
        return UNKNOWN, []
    return answer, [allowed[s] for s in source_ids]

def _validated_verification(payload, retrieved):
    if payload.get("supported") is not True:
        return []
    source_ids = payload.get("source_ids", [])
    allowed = {x["source_id"]: x for x in retrieved}
    if not isinstance(source_ids, list) or not source_ids:
        return []
    if any(s not in allowed for s in source_ids):
        return []
    return [allowed[s] for s in source_ids]

def answer_question(question, retriever=None):
    retriever = retriever or Retriever()
    retrieved = retriever.search(question)

    if not retrieved or retrieved[0]["score"] < MIN_SIMILARITY:
        return {
            "answer": UNKNOWN,
            "citations": [],
            "retrieved": retrieved,
            "decision": "refused_low_retrieval_confidence",
        }

    proposed, initially_cited = validate_model_output(
        generate_answer(question, retrieved), retrieved
    )
    if proposed == UNKNOWN:
        return {
            "answer": UNKNOWN,
            "citations": [],
            "retrieved": retrieved,
            "decision": "refused_by_generator",
        }

    verified = _validated_verification(
        verify_answer(question, proposed, retrieved), retrieved
    )
    if not verified:
        return {
            "answer": UNKNOWN,
            "citations": [],
            "retrieved": retrieved,
            "decision": "refused_by_grounding_verifier",
        }

    # Verifier citations are authoritative because they were checked for direct support.
    citations = [
        {
            "source": x["source"],
            "page": x["page"],
            "chunk": x["chunk_index"],
            "source_id": x["source_id"],
            "score": x["score"],
            "excerpt": x["text"][:420],
        }
        for x in verified
    ]
    return {
        "answer": proposed,
        "citations": citations,
        "retrieved": retrieved,
        "decision": "answered_and_verified",
    }
