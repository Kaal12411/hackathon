from .config import MIN_SIMILARITY, UNKNOWN
from .generator import generate_answer
from .retriever import Retriever

def validate_model_output(payload,retrieved):
    answer=str(payload.get("answer","")).strip()
    source_ids=payload.get("source_ids",[])
    if not answer or answer==UNKNOWN: return UNKNOWN,[]
    if not isinstance(source_ids,list) or not source_ids: return UNKNOWN,[]
    allowed={x["source_id"]:x for x in retrieved}
    if any(s not in allowed for s in source_ids): return UNKNOWN,[]
    return answer,[allowed[s] for s in source_ids]

def answer_question(question,retriever=None):
    retriever=retriever or Retriever()
    retrieved=retriever.search(question)
    if not retrieved or retrieved[0]["score"]<MIN_SIMILARITY:
        return {"answer":UNKNOWN,"citations":[],"retrieved":retrieved}
    answer,cited=validate_model_output(generate_answer(question,retrieved),retrieved)
    citations=[{"source":x["source"],"page":x["page"],"chunk":x["chunk_index"],"source_id":x["source_id"],"score":x["score"]} for x in cited]
    return {"answer":answer,"citations":citations,"retrieved":retrieved}
