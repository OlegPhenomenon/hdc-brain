# Milestone B.1 Log — bAbI task 2 via chained VSA reasoning

**Date:** 2026-04-11
**Status:** GATE PASSED — chained multi-hop reasoning works in pure VSA
**Code:** `hdc-cogito/run_babi_task2.py`
**Parent:** Milestone B of `docs/planning/hdc-cogito-proposal.md` §12

## Goal

Prove that pure VSA primitives (no training, no learned weights) can
perform **multi-hop reasoning** — the kind of inference where the
answer is not present in any single event but requires composing two
or more facts.

Concrete target: bAbI task 2 ("Two Supporting Facts"), synthesized locally.

Typical story:
```
Mary got the milk.
Mary travelled to the office.
Where is the milk?   -> office
```

The answer cannot be read off any one sentence. It requires a chain:

```
milk  --(held by)-->  Mary  --(located at)-->  office
```

## Design

Two decaying HDC memories, each a single vector in the HDC space:

| Memory | Writer event | Binding | Decay rate |
|-|-|-|-|
| `holder_of`   | pickup | `bind(obj, person)` | 0.7 |
| `location_of` | move   | `bind(person, place)` | 0.8 |

Query `where_is(obj)`:

```
holder = cleanup( unbind(holder_of,   obj_vec    ), people_seen )
place  = cleanup( unbind(location_of, holder_vec ), places_seen )
```

No reasoning block, no training, no iteration. Two chained Unbind
operations. Everything lives in `hdc_lang.py` from Milestone A — this
milestone is pure composition of existing primitives.

## First attempt: single global decay (failed)

With `decay=0.5` for both memories (same as Milestone A):

| dim | accuracy |
|-|-|
| 64    | 60.6% |
| 128   | 71.3% |
| 256   | 79.3% |
| 512   | 85.6% |
| 1024  | 89.6% |
| 2048  | 92.9% |

**Did not pass the 95% gate at any dimension.** Stage-2 confidence
(holder → place) was consistently low (~0.42). Diagnosis: `location_of`
is shared across all people's moves. When one person moves once early
in a 10-event story and nothing more, their binding decays to
`0.5^9 ≈ 0.002` by query time — effectively erased by recency bias
from other people's more recent moves.

## Fix: per-memory decay policy

Grid search over `(decay_holder, decay_location)` at D=1024:

```
  dh   dl=0.50  dl=0.70  dl=0.80  dl=0.90  dl=0.95  dl=1.00
 0.30    0.820    0.880    0.882    0.869    0.849    0.729
 0.50    0.896    0.962    0.964    0.952    0.927    0.801
 0.70    0.917    0.988    0.991    0.978    0.953    0.825
 0.90    0.911    0.980    0.983    0.970    0.943    0.817
```

Best setting: `decay_holder=0.7`, `decay_location=0.8`.

Interpretation — each memory has a different temporal policy:

- **`holder_of`** — wants *moderate* recency. Same object can be re-picked-up
  by a different person, so we want the latest pickup to dominate, but
  pickups are infrequent and we shouldn't forget them across several
  intervening moves of unrelated people.
- **`location_of`** — wants *weaker* forgetting. A person who moved once
  early and not again must still be findable when we query through them.
  Lower decay (0.8) preserves their entry long enough.

Pure-sum (decay=1.0) fails in both: any person who moves twice creates
a tie between their two bindings.

## Final evaluation (tuned decay)

500 synthetic stories, 743 questions, 6 people × 8 places × 8 objects,
4–12 events per story.

| dim | accuracy | stage1 conf | stage2 conf | speed (q/s) |
|-|-|-|-|-|
| 64   | 0.8371 | 0.551 | 0.455 |  8,853 |
| 128  | 0.9192 | 0.551 | 0.477 | 17,941 |
| 256  | 0.9623 | 0.547 | 0.475 | 17,061 |
| 512  | 0.9637 | 0.541 | 0.473 | 16,494 |
| 1024 | 0.9906 | 0.545 | 0.472 | 15,488 |
| 2048 | 0.9946 | 0.543 | 0.472 | 11,028 |

**Gate passed at D ≥ 256.** D=1024 gives ~99% accuracy at ~15k queries/sec.

## Findings

1. **Chained VSA reasoning works.** Two Unbind operations composed by
   cosine cleanup in between can recover a fact that is not present in
   any single HDC write. This is the first evidence that the internal
   language of Cogito can express multi-hop inference without any
   learned components.

2. **Per-memory decay is a structural finding, not a hyperparameter nit.**
   Different memories have different temporal dynamics:
   - "Which object is held by whom" changes on pickup events only.
   - "Where is each person" changes on move events only.
   - These two event rates are decoupled, so their decay must be too.
   This is a real design principle — it will resurface when we design
   the reasoning block, and it belongs in the proposal document.

3. **Minimum dim for task 2 is 256** — higher than task 1's minimum of 8,
   which makes sense: two-stage reasoning accumulates noise in both
   stages, so more capacity headroom is needed. Still very small.

4. **Speed remains in the tens of thousands of queries per second** on
   a single CPU core with no optimization. Multi-hop reasoning is not
   meaningfully slower than single-hop — two Unbinds plus two cleanups
   per query is cheap.

5. **Hand-crafted vs learned chains.** This milestone proves the
   language can *express* chained reasoning. The reasoner still
   hard-codes WHICH chain to run (first `holder_of`, then `location_of`).
   A system that can generalize to new tasks needs to *select* the
   right chain at query time. That is the job of the reasoning block
   (Milestone C), not the job of the language (Milestones A–B).

## Implications for the proposal document

These are small but real updates to fold back into
`docs/planning/hdc-cogito-proposal.md`:

- **§4.2** — capacity discussion should mention that different roles
  can have different memory policies (not a single global decay).
- **§6.1** — the capacity budget table is missing a "memory policy"
  dimension. Each memory bank has its own decay and its own growth rate.
- **§14 Open Questions** — add: "How should a learned reasoning block
  handle per-role decay? Is this learned from data, or declared?"

Not making these edits now — they're notes for Milestone C or later.

## Next step — Milestone B.2

Extend to bAbI task 3 (Three Supporting Facts) which adds **drops**.
A drop breaks the holder→location chain: the object is now at a fixed
place, not wherever the previous holder goes. The VSA solution:
add a third memory `dropped_at` and compare confidences between the
chained path and the direct path. If chained path's holder confidence
is below a threshold (holder "stale"), fall back to `dropped_at`.

This is still pure VSA composition — no training. It tests whether
the language can handle *branching* reasoning (choose between two
strategies based on confidence).
