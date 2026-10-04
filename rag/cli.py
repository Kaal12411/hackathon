import argparse
from .pipeline import answer_question

def main():
    p=argparse.ArgumentParser(); p.add_argument("question",nargs="+"); a=p.parse_args()
    r=answer_question(" ".join(a.question))
    print(f"\nAnswer: {r['answer']}")
    if r["citations"]:
        print("\nSources:")
        for c in r["citations"]:
            print(f"- {c['source']} | page {c['page']} | chunk {c['chunk']} | similarity {c['score']:.3f}")
if __name__=="__main__": main()
