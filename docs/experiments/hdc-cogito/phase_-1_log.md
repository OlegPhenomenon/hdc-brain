# Phase -1 Experiment Log — Pure VSA on bAbI task 1

**Date:** 2026-04-11
**Status:** GATE PASSED — proceed to Phase 0/1
**Code:** `hdc-cogito/hdc_lang.py`, `hdc-cogito/run_babi_task1.py`
**Runtime environment:** local laptop CPU, numpy 2.2.6, Python 3

## Goal

Validate that the HDC-Cogito internal language works **before** building
any learned components. Specifically: can pure VSA primitives (Bind,
Bundle, Permute, Unbind) solve bAbI task 1 (Single Supporting Fact) with
zero training and zero learned weights?

Success criterion from `docs/planning/hdc-cogito-proposal.md` §8.1: **> 95% accuracy**.

## Implementation

Minimal HDC library `hdc_lang.py` (~170 lines):

- `random_bipolar(dim, rng)` — sample {-1, +1}^dim
- `bind(a, b)` — elementwise multiply (self-inverse in bipolar space)
- `bundle(*vectors)` — sum (no normalization — weights carry info)
- `permute(v, k)` — cyclic shift
- `unbind(structure, role)` — same operation as bind
- `Codebook` — named vectors with nearest-neighbor cleanup
- `HDCMemory` — decaying holographic memory: `state = decay * state + new`

Task 1 encoding in `run_babi_task1.py`:

- Each "X moved to Y" → `bind(person_vec[X], place_vec[Y])`
- Memory accumulates bindings with exponential decay
- Query "Where is X?" → `unbind(memory, person_vec[X])` then cleanup
  against the place codebook

Note for Phase -1: person vectors play the role of "role" vectors. This
degenerate case is legitimate for a single-role task; later phases will
introduce the full role vocabulary (agent, location, time, ...).

## Unit tests

All 7 VSA primitive tests pass:

```
PASS  test_bind_is_self_inverse
PASS  test_unbind_retrieves_from_structure
PASS  test_codebook_cleanup_resolves_noisy
PASS  test_memory_single_item_recall
PASS  test_memory_decay_prefers_recent
PASS  test_cross_term_noise_is_small_in_high_dim
PASS  test_permute_is_reversible
```

## Main evaluation — dim sweep on synthetic bAbI task 1

500 stories, 1572 questions, 8 people × 8 places, 2–6 sentences per story.

| dim | accuracy | correct/total | avg confidence (correct) |
|-|-|-|-|
| 512  | 1.0000 | 1572/1572 | 0.9045 |
| 1024 | 1.0000 | 1572/1572 | 0.9045 |
| 2048 | 1.0000 | 1572/1572 | 0.9044 |
| 4096 | 1.0000 | 1572/1572 | 0.9045 |

**Gate passed at every dim tested.** Confidence is consistent across
dimensionality — Plate's HRR retrieval is essentially noise-free as soon
as the space has enough capacity.

## How small can we go?

Probing minimum useful dimensionality on the default task:

| dim | accuracy | memory per vector |
|-|-|-|
| 4  | 0.6361 | 16 bytes (float32) |
| 8  | 1.0000 | 32 bytes |
| 16 | 1.0000 | 64 bytes |
| 24 | 1.0000 | 96 bytes |
| 32 | 1.0000 | 128 bytes |

**D = 8 is the floor for this task.** At D=4 the space runs out of
capacity for 16 distinct bipolar vectors (8 people + 8 places). At D=8
it has just enough. From D=16 upward there is ample headroom.

## Stress test — larger vocabulary

20 people × 20 places, 500 stories, 2–10 sentences per story, 2559 questions:

| dim | accuracy |
|-|-|
| 64 | 1.0000 |
| 128 | 1.0000 |
| 256 | 1.0000 |
| 512 | 1.0000 |
| 1024 | 1.0000 |

## Stress test — hard recency

500 stories, **15 sentences/story on average**, only 4 people and 6 places,
so each person moves 3–5 times before the query. This is where pure sum
memory should start losing information without strong decay.

| dim | decay | accuracy |
|-|-|-|
| 64   | 0.30 | 1.0000 |
| 64   | 0.50 | 1.0000 |
| 64   | 0.70 | 0.9828 |
| 64   | 0.90 | 0.7501 |
| 256  | 0.30 | 1.0000 |
| 256  | 0.50 | 1.0000 |
| 256  | 0.70 | 0.9858 |
| 256  | 0.90 | 0.7992 |
| 1024 | 0.30 | 1.0000 |
| 1024 | 0.50 | 1.0000 |
| 1024 | 0.70 | 0.9863 |
| 1024 | 0.90 | 0.8240 |

Observations:
- **Decay matters more than dimensionality** on the recency-stressed task
- `decay=0.5` achieves 100% at every dim (including D=64)
- `decay=0.9` drops significantly — without recency bias, old locations
  for the same person create interference that pure cleanup can't resolve

## Speed

At D=512 on a single CPU core (numpy, no BLAS tricks):

- **1572 questions solved in 56 ms ⇒ ~28,000 questions/second**

This matches the theoretical prediction: Bind = elementwise multiply
(O(D)), Unbind = same (O(D)), cleanup = N cosine similarities (O(N·D)).
For N ≈ 8 places and D = 512 the whole pipeline is just a handful of
vector multiplications.

## Conclusions

1. **The VSA language works.** Phase -1 gate passed with massive margin.
   The internal language of HDC-Cogito is grounded in reality — Bind /
   Bundle / Permute / Unbind are sufficient primitives to solve a
   structured reasoning task without training.

2. **Capacity is not a concern for reasonable tasks.** Even at D=64 — 64
   float32 values = 256 bytes per vector — the system solves the hard
   recency task at 100%. The "runs on a calculator" goal is not
   hyperbole. It is overshoot.

3. **Decay is the most important design choice** for memory that must
   respect recency. Dim can be cut by 20x with no accuracy loss; decay
   cannot. This is a useful finding for designing later phases.

4. **Speed is not a concern.** 28,000 queries/second on CPU with no
   optimization. There is no CPU/GPU bottleneck for this kind of work.

5. **There is no "parameter count" here that means anything.** The system
   has a dimensionality, a vocabulary, a decay hyperparameter, and a
   memory state vector. None of these are "parameters" in the
   conventional sense. The axis labels from the proposal (§6.1) are the
   correct ones to use.

## Next step

Proceed to Phase 0 / Phase 1 as defined in `docs/planning/hdc-cogito-proposal.md`:

- **Phase 0**: wait for v14.1 English pretraining to finish on the server,
  then seed the concept codebook from its embeddings.
- **Phase 1**: verify that VSA structures can be extracted from v14.1's
  internal memory (go/no-go for reusing v14.1 directly).
- **Week 2 work**: extend `hdc_lang` to handle tasks 2–5 of bAbI, which
  require multi-supporting-fact reasoning and temporal order — this is
  where Permute starts earning its keep.

## Files produced

- `hdc-cogito/hdc_lang.py` — core VSA primitives
- `hdc-cogito/test_hdc_lang.py` — 7 unit tests, all passing
- `hdc-cogito/run_babi_task1.py` — synthetic bAbI task 1 evaluation
- `docs/experiments/hdc-cogito/phase_-1_log.md` — this file
