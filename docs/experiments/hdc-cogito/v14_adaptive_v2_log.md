# v14 adaptive thought halting v2

Date: 2026-04-11  device: CPU  eval: 2 Russian passages, per-step

## Fixed baselines

| n | ppl | time (s) | tok/s |
|---:|---:|---:|---:|
| 2 | 328.67 | 3.1 | 25.80 |
| 3 | 102.07 | 4.4 | 18.11 |

## Adaptive (start n=2, escalate to n=3 when entropy>H or top1-top2<G)

| H | G | ppl | avg thoughts | % n=3 | time (s) |
|---:|---:|---:|---:|---:|---:|
| 10.0 | -1.0 | 328.67 | 2.00 | 0% | 3.0 |
| 4.0 | 0.0 | 261.06 | 2.28 | 28% | 4.1 |
| 3.0 | 0.0 | 211.47 | 2.46 | 46% | 4.9 |
| 2.0 | 0.0 | 153.65 | 2.66 | 66% | 5.7 |
| -1.0 | 0.1 | 102.07 | 3.00 | 100% | 7.0 |
| 3.0 | 0.15 | 163.79 | 2.66 | 66% | 5.6 |
| 0.0 | 1000000000.0 | 102.07 | 3.00 | 100% | 7.1 |
