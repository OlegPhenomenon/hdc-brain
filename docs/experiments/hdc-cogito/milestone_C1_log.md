# Milestone C.1 Log — Depth stress test for iterative reasoning

**Date:** 2026-04-11
**Status:** GATE PASSED — clean depth staircase up to 20 hops, with an empirical D → depth scaling law
**Code:** `hdc-cogito/run_milestone_C1.py`, plus a bug fix in `hdc-cogito/run_milestone_C.py`
**Parent:** extension of Milestone C (`docs/planning/hdc-cogito-proposal.md` §12)

## Goal

Validate whether iteration count genuinely scales with problem depth,
or whether it saturates at 2 hops (as observed on bAbI tasks 2 and 3).

If iteration saturates at 2 even for deeper problems, the proposal's
"variable depth reasoning up to ~30 iterations" is over-engineered and
Milestone D must shrink. If iteration scales cleanly with depth, the
story holds and the reasoning block design (Milestone D / G) is
justified in planning for variable-depth inference.

## Task design

Synthetic nested containers:

```
put(apple, box)
put(box, bag)
put(bag, kitchen)
Where is the apple?   ->  kitchen
```

A chain of depth D requires D unbinds to follow:
item → c1 → c2 → ... → c(D-1) → place.

No shortcut exists — the `inside_of` memory stores only adjacent-level
bindings. Each story has 3 chains (1 target + 2 distractors) for
cross-term noise. Chain containers are drawn from disjoint pools so
different chains do not interfere structurally.

Same `ReasoningBlock` from Milestone C, no task-specific code.

## First run — unexpected failure at depth 3+

Initial results with the vanilla Milestone C block:

| depth | iter=1 | iter=2 | iter=3 | iter=4 | iter=5 | iter=7 |
|-|-|-|-|-|-|-|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | 0.000 | 0.000 | **0.080** | 0.080 | 0.080 | 0.080 |
| 4 | 0.000 | 0.000 | 0.000 | **0.337** | 0.337 | 0.337 |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | 0.000 |

Staircase for depths 1 and 2, but depths 3+ completely fail even when
`max_iter` is sufficient. Why?

### Root cause — symmetric bind

`bind` in bipolar HDC is its own inverse: `bind(a, b) == bind(b, a)`.
So the binding `bind(apple, box)` stored in `inside_of` looks
identical to `bind(box, apple)`. When the block is at state `h = bag`
at iteration 3 trying to move forward along `apple → box → bag → kitchen`,
`unbind(inside_of, bag)` returns a mixture dominated by two terms:

- `box` (the cross-term from `bind(box, bag)` — the **backward** direction)
- `kitchen` (the signal term from `bind(bag, kitchen)` — the **forward** direction)

Both terms have the same coefficient (1.0) in the recovered vector.
Cleanup picks one of them essentially at random based on small noise.
If `box` wins, the block sees that box was already visited and
**returns the wrong answer** via the early-termination cycle-detection
branch. The ~8% accuracy at depth 3 is the fraction of times `kitchen`
happened to beat `box` on a coin flip.

This is a real bug in the block design: the cycle-detection fires
*after* picking the winner, so a backward-pointing winner kills the
query.

### Fix — exclude visited concepts from cleanup

One-line structural change to `ReasoningBlock._best_step`: for each
candidate set, filter out concepts already on the current reasoning
path before running cleanup.

```python
candidates = [n for n in names if n not in visited]
```

This removes the backward direction from consideration at every hop.
Because the forward direction is the only remaining correct retrieval
in the candidate set, cleanup can lock onto it cleanly.

Tasks 2 and 3 were re-run after the fix to verify it does not regress
earlier milestones. Task 2: 98.65% (unchanged). Task 3: 98.13%
(unchanged). The fix is backward-compatible — it only matters for
tasks where the block traverses a bound chain in one memory, which
was not the case in B.1 / B.2.

## Main result — clean staircase

After the fix, D = 2048, decay = 1.0, 300 stories per depth:

| depth | iter=1 | iter=2 | iter=3 | iter=4 | iter=5 | iter=7 |
|-|-|-|-|-|-|-|
| 1 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 0.000 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 | 1.000 |
| 4 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 |

Perfect staircase. Accuracy flips from 0.000 to 1.000 exactly at
the main diagonal. No intermediate values, no noise.

Iteration-use histograms at `max_iter = 7`:

| depth | iterations used | count |
|-|-|-|
| 1 | 1 | 300 |
| 2 | 2 | 300 |
| 3 | 3 | 300 |
| 4 | 4 | 300 |
| 5 | 5 | 300 |

**The block uses exactly `depth` iterations per query.** Every single
one of 1500 queries. This is pure adaptive compute — the block stops
as soon as the retrieved concept is of the target type, which happens
exactly when the chain has been fully traversed.

## Extreme depth test

Pushing further:

### D = 1024

| depth | accuracy | notes |
|-|-|-|
| 1  | 1.000 | 1 iter on all 200 |
| 3  | 1.000 | 3 iter on all 200 |
| 5  | 1.000 | 5 iter on all 200 |
| 7  | 1.000 | 7 iter on all 200 |
| 10 | 1.000 | 10 iter on all 200 |
| 15 | 1.000 | 15 iter on all 200 |
| **20** | **0.380** | mixed: 4, 9, 10, 20, 22 iter counts |

At depth 20, D=1024 collapses. Mixed iteration counts mean the block
is wandering through noise, either running out of budget or
accidentally hitting a false-positive place match early.

### D = 2048

| depth | accuracy |
|-|-|
| 1 → 20 | **1.000** at every depth tested |

Doubling D restores perfect performance at depth 20.

### D = 4096

Same as D=2048 — perfect across the board.

## Findings

1. **Iteration count scales 1:1 with problem depth.** This is as
   clean a positive result as you can get: depth D requires exactly
   D iterations, always. The proposal's "variable depth reasoning"
   premise is empirically supported for chain-like problems up to
   depth 20.

2. **Dimensionality sets the depth ceiling.** There is an empirical
   scaling law of the form `max_reachable_depth ≈ f(D)`:

   - D = 1024 → reliably depth 15, collapses at 20
   - D = 2048 → reliably depth 20 (likely further, not tested)
   - D = 4096 → reliably depth 20 (likely further, not tested)

   This is not a parameter count — it is a property of HDC capacity
   theory. More dimensions means more headroom for noise accumulation.
   This is a real design knob that Milestone D / G must plan for.

3. **`bind` symmetry is a non-optional design constraint.** Any
   HDC reasoning block that traverses chains must track visited state
   and exclude it from cleanup. Otherwise the backward direction of
   a symmetric binding is just as strong a signal as the forward
   direction, and the block picks randomly. This is now a required
   property of `ReasoningBlock`, documented in the code.

4. **Adaptive compute works perfectly without any learned gate.** The
   block naturally stops as soon as the retrieved concept type matches
   the target type — no PonderNet loss, no confidence threshold
   tuning, no learnable halting. This simplifies Rung 1 significantly:
   the "confidence gate" may be unnecessary for type-bounded queries
   and only matter when the answer type is ambiguous.

5. **Failure mode at depth saturation is diagnosable.** When noise
   exceeds signal, the block either runs out of iteration budget
   (reports wrong final concept) or generates a false-positive type
   match (stops early on a random place that happens to cosine-match).
   Both are detectable in the iteration histogram — clean problems
   show a single peak, noise-dominated problems show a spread.

6. **The proposal's "up to 30 iterations" claim is now credible.**
   We hit 20 at D=2048 without any tuning. Scaling to 30 is almost
   certainly a matter of larger D. This justifies the adaptive-depth
   story in §4.2 / §4.5 of the proposal.

## Implications for the proposal document

- **§4.2 Knowledge = Data × Iterations** — add a concrete scaling
  claim: "at D = 2048, the block performs 20-hop reasoning at 100%
  accuracy. Larger D → deeper reachable reasoning, empirically."
- **§6.3 The reasoning block** — the block MUST maintain a visited set
  and exclude it from cleanup. This is a correctness requirement, not
  an optimization. Fold into the pseudocode.
- **§6.4 Confidence gate** — simplified: for type-bounded queries,
  adaptive halting is automatic (stop when the retrieval is of the
  target type). The learned gate only matters for ambiguous or
  open-ended queries. Worth noting before Milestone G designs a
  complicated gate.
- **§14 Open questions** — add: "What is the empirical function
  `max_reachable_depth = f(D)` for realistic chain distributions?
  Is it linear, logarithmic, or saturating?"
- **§11 Risks** — downgrade "iteration saturates at shallow depth"
  from High to Low. Empirically disproven.

I am folding these into the proposal now (not notes for later) because
they change the design calculus for Milestone G.

## Next step

**Milestone C.1 is complete.** Three reasonable paths forward:

- **Milestone C.2** — stress test with more diverse chain structures:
  branching trees (where a container holds multiple items), cycles,
  shared subchains. Show that the block handles these without
  extensions.
- **Milestone D — design review for the learned block.** We now have
  enough empirical grounding. Read Plate / Gayler / Kanerva, sketch
  the learned dynamics on paper, review before coding.
- **Phase 0 probe** — v14.1 is still training. When it finishes, do
  a preliminary probe: extract its output embeddings for a few words
  and check whether VSA-style bind/unbind patterns work on them.
  This is the first honest test of whether Cogito can attach to the
  real language model asset.

My recommendation: **Milestone D**. The empirical case for iteration
is now strong enough that designing the learned block on paper is a
productive use of work. C.2 can wait — chain reasoning on simple
containers is well-validated; exotic structures are a nice-to-have,
not a blocker for the learned block design.
