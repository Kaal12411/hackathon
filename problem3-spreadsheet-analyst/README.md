# Problem 3 — Natural Language Spreadsheet Analyst

A safe, demo-ready analyst for messy CSV files. Users ask questions in plain English and receive:

1. a one-line insight,
2. a result table,
3. an automatically selected chart.

## Dataset

NYC Open Data — 311 Service Requests from 2020 to Present (dataset id: erm2-nwe9).

The official dataset is very large, so the demo downloads a manageable local sample for fast offline analysis.

## Rubric-first design

| Rubric | Marks | Design |
|---|---:|---|
| Accuracy on 5 predefined questions | 40 | schema-aware structured planning + offline rules for common NYC 311 questions |
| Robustness | 30 | null normalization, date inference, numeric inference, malformed-row skipping, encoding fallback |
| Chart quality | 20 | automatic bar/line/pie selection with Plotly labels and titles |
| Safety | 10 | no exec/eval, no generated Python/SQL, allow-listed JSON plan, schema validation |

## Architecture

Plain English question
-> offline fast-path for common questions OR LLM structured planner
-> validated JSON query plan
-> allow-listed Pandas executor
-> one-line insight + table + chart

The LLM is never allowed to execute Python.

## Safety model

Allowed planner operations are strictly limited to:
- filters: eq, neq, contains, gt, gte, lt, lte, between
- aggregations: count, sum, mean, median, min, max, nunique
- grouping: at most 2 columns
- time grains: day, week, month, quarter, year
- charts: auto, bar, line, pie, none

Unknown columns, unsupported operations, malformed plans, or unsafe requests fail closed.

## Quick start

~~~bash
cd problem3-spreadsheet-analyst
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
~~~

Copy .env.example to .env and add a Groq or OpenAI API key for general natural-language questions.

## Download a local NYC 311 sample

~~~bash
python download_sample.py --rows 50000
~~~

## Run the app

~~~bash
streamlit run app.py
~~~

## Five-question demo benchmark

~~~bash
python evaluate_examples.py
~~~

It runs:
- top complaint types
- busiest borough
- busiest agency
- monthly complaint trend
- top complaint types in Brooklyn

These common questions use the offline fast-path, so the live demo remains reliable even if the LLM API is unavailable.

## Run safety tests

~~~bash
pytest -q
~~~

## Judge-facing design explanation

Use this line:

"The LLM does not generate executable Python. It only proposes a small JSON query plan, and our validator rejects unknown columns or unsupported operations before a separate allow-listed Pandas executor runs anything."

That gives us natural-language flexibility without arbitrary code execution.
