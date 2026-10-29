# Evaluation results

Produced with `lecture-rag eval --eval-file eval/eval_set.json --k 5 --abstain-threshold 0.35 --output eval`.

- Eval file: `eval/eval_set.json`
- Questions: 33 (25 answerable, 8 unanswerable)
- Embedding model: `all-MiniLM-L6-v2`
- Indexed chunks: 8
- Abstain threshold: 0.35 (dense cosine similarity)

Hit and MRR are computed over answerable questions; abstain accuracy over all questions.

## Retrieval mode comparison

| Mode | Hit@5 | MRR@5 | Hit@1 | MRR@1 | Abstain acc |
|------|-------|-------|-------|-------|-------------|
| dense | 1.000 | 1.000 | 1.000 | 1.000 | 0.939 |
| bm25 | 1.000 | 0.980 | 0.960 | 0.960 | 0.939 |
| hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 0.939 |

## Abstention at threshold 0.35 (dense)

| Metric | Value |
|--------|-------|
| Abstain accuracy | 0.939 |
| Abstain precision | 0.800 |
| Abstain recall | 1.000 |
| Hit@5 after abstain | 0.920 |
| Mean cosine score (top 5) | 0.295 |

## Threshold sweep (dense)

| Threshold | Abstain acc |
|-----------|-------------|
| 0.20 | 0.939 |
| 0.25 | 0.909 |
| 0.30 | 0.939 |
| 0.35 | 0.939 |
| 0.40 | 0.939 |
| 0.45 | 0.879 |
| 0.50 | 0.818 |
| 0.55 | 0.727 |
| 0.60 | 0.576 |
