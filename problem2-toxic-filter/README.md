# Real-Time Toxic Comment Filter with Explainability

Hackathon-ready solution for the Jigsaw Unintended Bias in Toxicity Classification dataset.

## Rubric-first design

| Rubric | Marks | Design |
|---|---:|---|
| F1 on hidden test | 40 | TF-IDF word + character n-grams, class balancing, threshold tuning |
| Latency <500ms | 30 | Logistic Regression, cached model, sparse inference |
| Explainability | 20 | Fast token-occlusion highlighting tied to model probability |
| Edge cases | 10 | Character n-grams catch misspellings/obfuscation such as "stooopid" |

## Dataset
Kaggle: Jigsaw Unintended Bias in Toxicity Classification

Place the competition training file at:
`data/train.csv`

Expected columns:
- `comment_text`
- `target`

A comment is toxic when `target >= 0.5`.

## Architecture
```
comment
  |
  +--> word TF-IDF (1-2 grams)
  |
  +--> character TF-IDF (3-5 grams)
          |
          v
   sparse feature union
          |
          v
 Logistic Regression
          |
          +--> toxicity probability
          +--> F1-tuned threshold
          +--> token occlusion explanation
```

## Why character n-grams?
They improve robustness to:
- `stooopid`
- `stuuupid`
- `i.d.i.o.t`
- punctuation/spacing obfuscation

## Run

```bash
pip install -r requirements.txt
python eda.py --data data/train.csv
python train.py --data data/train.csv --max-rows 300000
python evaluate.py --data data/train.csv --sample 50000
streamlit run app.py
```

## Explainability
The UI removes one visible token at a time and measures the drop in toxic probability. This captures contributions from both word and character n-grams, so misspelled toxic terms can still be highlighted.

## Latency
The UI measures end-to-end inference + explanation latency for every comment and shows whether it passes the 500ms target.
