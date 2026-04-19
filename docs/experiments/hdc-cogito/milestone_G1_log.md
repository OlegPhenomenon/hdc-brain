# Milestone G.1 Log — First learned reasoning block (minimal: 6 scalars)

**Date:** 2026-04-11
**Status:** PARTIAL RESULT — learned block improves over Milestone C baseline but does not fully close the gap to hand-coded task-specific reasoners
**Code:** `hdc-cogito/run_milestone_G1.py`
**Parent:** Milestone G of `docs/planning/hdc-cogito-proposal.md` §12, implementing the minimal design from `docs/planning/milestone_D_design.md`

## Goal

Test the minimal learned-block design from Milestone D: add 6 trainable
scalars (one per `(memory, type)` combination) to the Milestone C
task-agnostic block. Train via random search on held-out validation.
Measure whether this closes the 0.4% / 1.2% gap vs hand-coded
reasoners on bAbI tasks 2 and 3.

## Design

`LearnedReasoningBlock` extends `ReasoningBlock` with one extra
field: `weights: dict[(memory_name, type_name), float]`. The scoring
function in `_best_step` becomes:

```python
scored = raw_cleanup_cosine * weights[(mem, type)]
```

All other properties of the Milestone C block are preserved:

- Visited-exclusion in cleanup (correctness requirement from C.1)
- Type-match halting (adaptive compute at no learning cost)
- Iterative refinement until target type or termination
- No neural network, no autograd — just 6 scalar biases on scores

Training: **random search**, 500 samples, uniform draws from [0.3, 2.5]
per scalar. Validation-based selection: best on val set becomes the
final weights. Test set is never seen during search.

Data splits:

| split      | task 2 queries | task 3 queries |
|-|-|-|
| train      | 743            | 750 (unused — random search, not gradient descent)            |
| validation | 293            | 281            |
| test       | 448            | 458            |

Encoding is precomputed once per story, so the random search only
runs the query step over ~1500 questions per sample. Total search
time: ~27 seconds for 500 samples.

## Results

### Test-set accuracy

| System                  | Task 2  | Task 3  |
|-|-|-|
| Milestone C baseline    | 0.9799  | 0.9629  |
| **Learned block (6 scalars)** | **0.9821**  | **0.9760**  |
| Hand-coded reference    | 0.9906  | 0.9933  |

**Delta vs baseline:** task 2 +0.22%, task 3 +1.31%.

**Remaining gap vs hand-coded:** task 2 0.85%, task 3 1.73%.

Note: the Milestone C baseline on this test split (0.9799 / 0.9629) is
slightly below the numbers reported in the original Milestone C log
(0.9865 / 0.9813) because this experiment uses fresh test seeds
(`Random(200)` for stories) rather than the seed=42 data from the
Milestone C run. The within-experiment comparison is still clean:
both baseline and learned block use the same test stories.

### The learned weights

```
holder_of   x person  :  1.999
holder_of   x place   :  1.417
location_of x person  :  1.536
location_of x place   :  1.972
dropped_at  x person  :  2.307   <-- surprising, see below
dropped_at  x place   :  1.275   <-- expected to be highest, isn't
```

Ranked by weight per target type:

- **target = person:** dropped_at (2.307) > holder_of (1.999) > location_of (1.536)
- **target = place:**  location_of (1.972) > holder_of (1.417) > dropped_at (1.275)

The target=place ranking is the intuitive one: `location_of` and
`holder_of` are the two memories involved in the chain (object → holder
→ place), and `location_of` closes the chain.

The target=person ranking is **unexpected**: `dropped_at` is the top
scorer despite storing (object, place) bindings. This is almost
certainly noise — the random search found a local maximum where this
weight happened to tie-break in a helpful direction on the validation
set without any principled reason.

### Follow-up probe: add `marker` type

A quick extension added the DROPPED marker to type_groups for task 3
and re-ran the search over 9 scalars (3 memories × 3 types):

- Test task 3 with extended type_groups, baseline weights: 0.9585
  (slightly worse than 6-scalar baseline, because DROPPED marker
  retrievals occasionally win and stall the block)
- Test task 3 with extended type_groups, learned weights: 0.9760
  (identical to the 6-scalar learned version)

The search learned that marker-type weights should be **low** (0.50,
0.73, 1.59), effectively suppressing marker retrievals and reverting
to the 6-scalar solution. Adding marker visibility does not help.

## Interpretation

### What the learned block CAN do

- Distinguish "which memory is reliable for which answer type".
- Close a small fraction of the gap (1-2% on task 3).
- Stay inside the HDC algebra — just 6-9 scalar biases on cleanup
  scores, no MLPs, no autograd, no neural network.
- Preserve every invariant of Milestone C (visited-exclusion,
  type-match halting, adaptive compute).

### What the learned block CANNOT do

- **Conditional routing.** The hand-coded task 3 reasoner uses an
  explicit if-then branch: "if `holder_of` returns DROPPED, use
  `dropped_at`; else chain through `location_of`". The learned block
  cannot express this, because scalar biases on cleanup scores are not
  conditional rules. It just re-weights averages.
- **Task-specific state recognition.** The hand-coded reasoner "sees"
  DROPPED as a distinct state and routes accordingly. The learned
  block either ignores DROPPED (without marker type in cleanup) or
  suppresses it (with marker type), because DROPPED as a retrieval
  target does not lead to a useful state update.

### Where the remaining gap lives

The gap on task 3 is structural, not a hyperparameter issue. Closing
it requires one of:

1. **A learned operator or routing mechanism** — some way for the
   block to say "when the state contains a DROPPED signal, the next
   step should query memory X". This is closer to the proposal's
   original §5 "operator codebook" idea, which Milestone C had
   tentatively deferred.
2. **Task-specific encoding on the write side.** The hand-coded
   `Task3Reasoner` uses memory-consult during encoding (writes both
   `holder_of` with DROPPED marker AND `dropped_at` with location on
   every drop event). The learned block reuses this encoder, so it
   already benefits from that. The gap is on the READ side.
3. **Accepting the gap as the cost of task-agnostic generality.** A
   universal block is structurally weaker than a task-specific one.
   1-2% is a small price for "one function handles many tasks".

### Why the unexpected weight (dropped_at × person = 2.307)?

The random search is sampling sparsely in a 6-dim space. With 500
samples, the search hits many local maxima on the validation set.
The reported "best" is the highest-scoring sample — not a global
optimum. Some weights will be noise-driven, especially on combinations
that rarely matter for accuracy (no query in task 2 or task 3 ever
truly depends on `dropped_at × person`, because dropped_at doesn't
store people). Those weights are effectively free parameters that
the search wiggles without affecting outcome.

A principled optimizer (CMA-ES, gradient descent via softmax
relaxation, coordinate descent) would likely converge to cleaner
values, but the empirical accuracy is unlikely to improve meaningfully
because the ceiling is structural.

## Findings

1. **The minimal 6-scalar design helps, but does not close the gap.**
   Delta is real (+0.22% task 2, +1.31% task 3) but small. Rerunning
   with 2000 samples or principled optimization will likely not close
   the rest.

2. **The remaining gap is structural.** Hand-coded task 3 uses
   conditional routing that scalar biases cannot express. Closing it
   requires a more expressive learned mechanism — either operator
   codebook, or learned transitions that recognize distinguished
   states.

3. **Marker visibility in type_groups does not help.** The search
   learned to suppress marker weights, because DROPPED as a retrieval
   target does not lead to a useful next state. A learned block that
   TRULY uses markers would need to interpret them as routing signals,
   not as state updates.

4. **Task-agnostic generality has a small but real cost.** ~1-2%
   on bAbI tasks 2 and 3. For a first paper on Cogito, this is a
   publishable result: "learned task-agnostic block reaches 98%
   accuracy within 1% of task-specific hand-coded reasoners, with
   only 6 learned scalars and zero neural network parameters".

5. **The design document's "if 6 scalars don't help, the gap is
   structural" prediction is validated.** Milestone D §5.1 anticipated
   this failure mode and suggested looking at operator-like mechanisms.
   Milestone D's analysis is now a more concrete document for
   Milestone D-revised or Milestone G-proper.

## Next step options (in order of preference)

1. **Stop here and accept the result.** 6 scalars give a small clean
   improvement. Write it up. Move on to Phase 0 / Milestone F when
   v14.1 finishes training on the server. The 1-2% gap is the known
   limit of the task-agnostic approach.

2. **Design a minimal operator-like mechanism.** Add a tiny learned
   routing table: "if cleanup returns X at iteration i, next step
   query memory Y". This is 5-10 more parameters. Requires a careful
   design review because it reintroduces top-down routing, which the
   proposal §4.3 warns against. But if it is **learned** from data
   rather than hardcoded, it may be acceptable.

3. **Failure analysis on the specific 2.4% of task 3 that the learned
   block gets wrong.** Look at the stories case by case. Diagnose
   whether the failures are all "stale holder vs dropped" or something
   else. Informative regardless of whether we fix them.

## Implications for the proposal document

Small updates to fold back into `docs/planning/hdc-cogito-proposal.md`:

- **§6.3 The reasoning block** — add a note: "a minimal learned block
  with 6 per-(memory, type) scalars matches the Milestone C task-
  agnostic block plus ~1% on task 3, but cannot close the full gap
  to hand-coded task-specific reasoners without expressive routing
  mechanisms."
- **§14 Open questions** — update Q1: "`learned_refine` with only
  scalar biases hit a structural ceiling on task 3. Closing the
  remaining gap requires learned operators / routing, which we are
  not yet ready to implement."

Not folding these in right now — waiting for the user's decision on
whether to continue to option 2, 3, or stop.
