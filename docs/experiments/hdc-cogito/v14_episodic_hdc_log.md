# v14 + episodic HDC buffer beyond 512-token window

Date: 2026-04-12  device: CPU  n_thoughts=3

Long text: 599 BPE tokens (v14 window=512).
Scored positions 512..598 — every target has ≥1 token out of window.

## Pure v14 baseline

- ppl = **15759.43**

## With episodic HDC bias

| alpha | decay | ppl | vs baseline |
|---:|---:|---:|---|
| 0.05 | 0.999 | 30117.24 | +91.1% |
| 0.05 | 0.99 | 19546.07 | +24.0% |
| 0.05 | 0.95 | 12664.55 | -19.6% |
| 0.1 | 0.999 | 456429.37 | +2796.2% |
| 0.1 | 0.99 | 114029.72 | +623.6% |
| 0.1 | 0.95 | 15267.99 | -3.1% |
| 0.2 | 0.999 | 936313653.00 | +5941192.9% |
| 0.2 | 0.99 | 37321564.47 | +236720.6% |
| 0.2 | 0.95 | 68701.85 | +335.9% |
