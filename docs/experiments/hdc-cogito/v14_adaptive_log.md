# v14 adaptive thought halting

Date: 2026-04-11

Eval corpus: 4 passages, 192 targets total.

## Full-sequence fixed n_thoughts (chat.py equivalent)

| n | ppl |
|---:|---:|
| 1 | 10139.36 |
| 2 | 201.33 |
| 3 | 79.91 |

Per-step incremental results below use only the two shortest passages (78 targets) for tractability.

## Per-step fixed

| n | ppl | time (s) | tok/s |
|---:|---:|---:|---:|
| 1 | 7436.61 | 1.9 | 41.25 |
| 2 | 164.72 | 2.9 | 26.96 |
| 3 | 64.10 | 3.9 | 20.00 |

## Adaptive halting

| threshold | ppl | avg thoughts | time (s) | dist (n=1/2/3) |
|---:|---:|---:|---:|---|
| 0.3 | 1900.59 | 1.59 | 3.8 | 52/6/20 |
| 0.5 | 397.62 | 2.29 | 6.2 | 23/9/46 |
| 0.7 | 149.95 | 2.64 | 7.3 | 10/8/60 |

## Reading

- Compare adaptive ppl against fixed n=3 ppl. If adaptive is
  within a few % of fixed-3 while using avg_thoughts < 3, we
  have adaptive compute for free on v14.
- The distribution dist tells us what fraction of positions
  actually needed the extra compute.