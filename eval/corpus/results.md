# Corpus evaluation results

The corpus in `data/corpus` contains 6 PDFs with 163 loaded pages, indexed as 861 chunks at chunk size 512 with `all-MiniLM-L6-v2`, and evaluated on 58 questions (48 answerable, 10 unanswerable) at k=5 with an abstain threshold of 0.35.

#### Retrieval mode comparison

| Mode | Rerank | Hit@5 | MRR@5 | Hit@1 | MRR@1 |
|------|--------|-------|-------|-------|-------|
| dense | no | 0.917 | 0.754 | 0.625 | 0.625 |
| dense | yes | 0.958 | 0.878 | 0.812 | 0.812 |
| bm25 | no | 0.958 | 0.837 | 0.750 | 0.750 |
| bm25 | yes | 0.958 | 0.878 | 0.812 | 0.812 |
| hybrid | no | 0.938 | 0.819 | 0.729 | 0.729 |
| hybrid | yes | 0.958 | 0.878 | 0.812 | 0.812 |

Abstention uses the first-stage dense similarity and is therefore the same with and without rerank.

| Mode | ms/query | ms/query with rerank |
|------|----------|----------------------|
| dense | 3.6 | 112.5 |
| bm25 | 4.4 | 90.4 |
| hybrid | 4.6 | 101.7 |

Latency was measured on one CPU machine and varies between runs.

#### Abstention at threshold 0.35 (dense)

| Metric | Value |
|--------|-------|
| Abstain accuracy | 0.966 |
| Abstain precision | 1.000 |
| Abstain recall | 0.800 |
| Hit@5 after abstain | 0.917 |

#### Threshold sweep (dense)

| Threshold | Abstain acc |
|-----------|-------------|
| 0.20 | 0.931 |
| 0.25 | 0.948 |
| 0.30 | 0.948 |
| 0.35 | 0.966 |
| 0.40 | 0.966 |
| 0.45 | 0.983 |
| 0.50 | 1.000 |
| 0.55 | 0.948 |
| 0.60 | 0.914 |

#### Chunk size comparison

| Chunk size | Chunks | Hit@5 | MRR@5 | Abstain acc |
|------------|--------|-------|-------|-------------|
| 256 | 1895 | 0.917 | 0.774 | 0.948 |
| 512 | 861 | 0.917 | 0.754 | 0.966 |
| 1024 | 450 | 0.896 | 0.727 | 0.966 |
