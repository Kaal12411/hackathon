from pathlib import Path

from data_loader import load_csv, profile_dataframe
from executor import execute_plan, make_insight
from planner import make_plan

QUESTIONS = [
    "What are the top 10 complaint types?",
    "Which borough has the most complaints?",
    "Which agency receives the most requests?",
    "How have complaints changed over time by month?",
    "What are the top 5 complaint types in Brooklyn?",
]

def main():
    path = Path("data/nyc311_sample.csv")
    if not path.exists():
        raise SystemExit("Run: python download_sample.py --rows 50000")
    df = load_csv(path)
    profile = profile_dataframe(df)

    print("=== FIVE-QUESTION DEMO CHECK ===")
    for i, q in enumerate(QUESTIONS, 1):
        plan, source = make_plan(q, profile, list(df.columns))
        result = execute_plan(df, plan)
        print(f"\n{i}. {q}")
        print(f"Planner: {source}")
        print(make_insight(result))
        print(result.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
