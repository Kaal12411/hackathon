from rag.config import UNKNOWN
from rag.pipeline import validate_model_output
R=[{"source_id":"S1","source":"example.pdf","page":7,"chunk_index":2,"score":0.82,"text":"Revenue was $10 million."}]
def test_valid_answer_keeps_citation():
    a,c=validate_model_output({"answer":"Revenue was $10 million.","source_ids":["S1"]},R); assert a!=""; assert c[0]["source"]=="example.pdf"
def test_missing_citation_fails_closed():
    a,c=validate_model_output({"answer":"Revenue was $10 million.","source_ids":[]},R); assert a==UNKNOWN and c==[]
def test_invented_source_fails_closed():
    a,c=validate_model_output({"answer":"Revenue was $10 million.","source_ids":["S99"]},R); assert a==UNKNOWN and c==[]
def test_unknown_stays_exact():
    a,c=validate_model_output({"answer":UNKNOWN,"source_ids":[]},R); assert a==UNKNOWN and c==[]
