# Milestone D — Design Review for the Learned Reasoning Block

**Status:** Design sketch for review, no code yet
**Date:** 2026-04-11
**Parent:** `docs/planning/hdc-cogito-proposal.md` §12 Milestone D
**Depends on:** Milestones A, B.1, B.2, C, C.1 (all DONE 2026-04-11)

## 0. Purpose and Scope

This document is a **paper design**, not an implementation. It
answers three questions before any code is written:

1. **What problem does a learned reasoning block solve that the
   Milestone C task-agnostic block does not?**
2. **What is the smallest learnable addition to the Milestone C
   block that could plausibly help, and why would it help?**
3. **What is the training setup — data, loss, differentiability —
   and what are its failure modes?**

After this design passes review, it gets implemented as Milestone G.
If review uncovers a fatal problem, we revise this document first.

A note on sources: the design below is derived from my working
knowledge of VSA/HRR and from the empirical findings of Milestones
A–C.1. I did not re-read Plate (1995) or Gayler (2003) in this
session. Before implementation, the user should re-verify any
specific HRR claim below against those sources. **Any load-bearing
claim is marked with [verify].**

---

## 1. What Does Milestone C Not Do?

The Milestone C task-agnostic block already:

- Solves bAbI task 1 (single hop) at 100% from D = 8.
- Solves bAbI task 2 (2-hop chain) at **98.65%** at D = 1024.
- Solves bAbI task 3 (branching with drops) at **98.13%** at D = 1024.
- Solves synthetic nested-container chains up to depth 20 at D = 2048
  with 100% accuracy, using exactly `depth` iterations per query.
- Handles adaptive compute for free via type-match halting.

The gaps are:

| Task | Hand-coded (Milestone B) | Task-agnostic (Milestone C) | Gap |
|-|-|-|-|
| Task 2 | 99.06% | 98.65% | 0.4% |
| Task 3 | 99.33% | 98.13% | 1.2% |

The gap is real but small. It comes from cases where two cleanup
retrievals have close confidence and the block picks the wrong one.
Hand-coded reasoners know which memory to consult first and avoid
this entirely.

### What the gap is NOT

- It is not a scaling problem. Larger D does not close it
  (C.1 sweep confirmed).
- It is not a missing primitive. Adding more Unbind / Bind /
  Permute operations does not add expressiveness beyond what the
  block already does.
- It is not an iteration budget problem. `max_iter = 5` is plenty
  for both tasks.

### What the gap IS

The block's failure mode on task 3 is consistent: when a memory's
signal is borderline, the block picks whichever retrieval happens
to have slightly higher cleanup cosine. Without learned knowledge
of **which memory to trust in which context**, the block has no
way to break ties systematically.

A **learned scoring bias** could break ties in the right direction,
based on training examples. That is the minimum thing a learned
block could do that Milestone C cannot.

---

## 2. Design Space (Ranked by Simplicity)

The design rule from the proposal (§13): **do not replace HDC
primitives with "something more expressive".** Whatever we learn
must stay inside the HDC algebra — vectors, Bind / Bundle / Permute,
cleanup. No free-form MLPs over activations.

Within that constraint, five candidate learnable mechanisms, from
simplest to most ambitious:

### 2.1 Learned per-memory / per-type scoring bias

Two small learnable scalar tables:

```
trust[memory_name]     ∈ ℝ   — one scalar per memory bank
affinity[memory, type] ∈ ℝ   — one scalar per (memory, type) pair
```

At cleanup time:

```
scored = raw_cosine * trust[mem] * affinity[mem, type]
```

For task 2 / task 3 there are 3 memories × 2 types = 6 affinity
parameters, plus 3 trust scalars = **9 learned scalars total.**

**Why this could help.** If training learns that "when looking for a
place, dropped_at is more trustworthy than holder_of's noise-level
place projection", then tie-breaking in ambiguous cases goes the
right way. It is the minimum signal that distinguishes "which
memory to consult when" from "brute-force try everything".

**Why this is still pure HDC.** The block does the same Unbind +
cleanup it already does. The scalar biases only affect the final
ranking. No MLP, no non-HDC operation.

**Parameters:** tiny — 9 scalars for current tasks, ~50 for a
Cogito-scale system with many memories and types.

### 2.2 Learned refinement vector

A single persistent vector `refine_v` bound into the state at the
start of each iteration:

```
h_new = bind(h, refine_v)
```

`refine_v` is a single HDC vector of D dims. It rotates the state
in a learned direction before the next round of cleanup. Over
training, it can learn to make the state "more retrievable" against
likely-next-step memories.

**Why this could help.** Some failures might come from the state
vector carrying noise from previous cleanup rounds. A learned
rotation could wash it out and leave only the "still relevant"
directions. This is what Plate's HRR cleanup is designed to do in
more elaborate form. [verify against Plate ch. 5]

**Problem.** A single refine_v is context-free — it rotates the
state the same way regardless of what stage of reasoning we are in.
Not obviously helpful. Better as a per-iteration-index lookup, but
that adds parameters.

### 2.3 Learned per-iteration refinement

Extend 2.2 to a small bank of K refinement vectors, one per
iteration index:

```
h_new = bind(h, refine_v[iteration_index])
```

K = 5 vectors of D = 2048 dims = 10,240 scalars. Still small.

**Why this could help.** Different iterations do different work
(hop 1 is object → holder, hop 2 is holder → place). A per-iteration
vector can specialize. The block learns "at iteration 1, rotate
toward holder retrieval; at iteration 2, rotate toward place
retrieval".

**Problem.** Introduces an assumption that each iteration has a
dedicated "job", which is true for fixed-depth tasks but not for
adaptive-depth queries. The vector at iteration 1 must generalize
across tasks of different total depths.

### 2.4 Learned attention over memories

Replace argmax over memories with a soft attention:

```
weights = softmax([learned_score(h, mem) for mem in memories])
combined_unbind = sum(weights[m] * unbind(memories[m], h) for m)
clean_name, conf = cleanup(combined_unbind, all_candidates)
```

Where `learned_score(h, mem)` is a small learned function. To stay
HDC-native, it can be a learned bind against a per-memory "key"
vector followed by cosine against h:

```
learned_score(h, mem) = cosine(h, bind(memory_keys[mem], query_context))
```

**Why this could help.** Lets the block learn "when in state X,
memory Y is relevant". More expressive than 2.1.

**Problem.** The soft combination introduces interference between
memories. If two memories return different concepts, their sum is
neither — it's a mixture. Cleanup might pick a third thing
entirely. Risk of worse behavior than 2.1.

### 2.5 Learned operator codebook (original proposal §5)

A small codebook of K "operator" vectors. At each iteration, pick
the best operator by similarity to state and bind it into h:

```
best_op = argmax_i cosine(h, op_codebook[i])
h_new = bind(h, op_codebook[best_op])
```

**Why this was in the original proposal.** Operators are the
"verbs" of the internal language — they learn to select reasoning
moves. This is the Plate-style symbolic-over-distributed approach.

**Why Milestone C findings argue against it.** The task-agnostic
block achieves 98.65% / 98.13% without any operator codebook. The
simpler rule "argmax over raw cleanup" discovers the right chain
without explicit operators. Adding operators might just reintroduce
the "top-down symbolic" trap the proposal §4.3 warns against.

**Verdict:** defer. If 2.1 closes the gap, we do not need 2.5.
If 2.1 does not help, rethink whether operators add value.

---

## 3. Recommended Minimal Design

**Start with 2.1 alone.** Nine learnable scalars on top of the
Milestone C block. Train on bAbI tasks 2 and 3. Measure:

- Does the learned block beat the Milestone C baseline on task 3
  (the weaker of the two)?
- Does it preserve Milestone C's performance on task 2?
- Does it generalize to held-out stories from the same task
  distribution?
- Does it transfer to C.1 depth-controlled chains without retraining?

If 2.1 alone closes the gap on tasks 2 and 3: stop, declare success,
move to Milestone E/F/G with the simpler design.

If 2.1 helps but does not fully close the gap: add 2.2 or 2.3
(learned refinement vector) as a second experiment.

If 2.1 does not help at all: the gap is not about memory trust,
it is about representation — and we need to rethink what the block
is doing, not just add parameters.

The rationale for starting minimal:

1. **Every extra parameter is a risk.** Adding parameters to HDC
   is less well-understood than adding parameters to transformers.
   We want to know exactly which parameter is doing the work.
2. **9 scalars train fast.** Gradient descent on a 9-dim loss
   surface is trivially stable. We can run hundreds of experiments
   in minutes.
3. **If the minimal thing works, the design document for Cogito v1
   stays simple.** A 9-parameter block is a sellable story: "we
   added nine scalars and the block beats hand-coded". Parsimony
   matters for the paper.
4. **If it does not work, we learn something structural about where
   the gap actually lives**, which redirects the next design iteration.

---

## 4. Training Setup

### 4.1 Data

- **Task 2 training set:** ~500 stories, 800–1000 queries. Same
  generator as Milestone B.1.
- **Task 3 training set:** ~500 stories, 800–1000 queries. Same
  generator as Milestone B.2.
- **Held-out validation:** another 200 stories per task, fresh seed.
- **Generalization probe:** C.1 depth-5 chains at D = 1024 — does
  the learned block break anything when applied to a different task
  type without retraining?

All synthetic, all cheap to generate and re-generate.

### 4.2 Forward pass (differentiable version of Milestone C block)

Current Milestone C step:

```python
best = argmax over (memory, type) of cleanup_confidence
```

Differentiable version:

```python
all_scores = [
    trust[m] * affinity[m, t] * cleanup_cosine(m, t, h)
    for m in memories
    for t in types
]
# Softmax over all (m, t) pairs
weights = softmax(all_scores / temperature)
# Expected next state: weighted mixture of retrieved concepts
h_new = sum(weights[i] * retrieved_concept_vector[i] for i in ...)
```

During training: use softmax and gradient descent.
During inference: use argmax (straight-through at the final step).

**Temperature schedule:** start high (soft, gradients flow through
all paths) and anneal to low (hard, matches inference behavior).

### 4.3 Loss

Primary loss: **cross-entropy between the block's final retrieved
concept and the true answer.** Specifically:

```python
final_cosines = cleanup_cosines_against_all_candidates(h_final)
loss = cross_entropy(final_cosines, true_answer_index)
```

This is a standard classification loss. No VSA fidelity loss, no
halting loss, no self-memory loss — keep the first training run
minimal. Add auxiliary losses only if the main loss does not
converge.

### 4.4 Optimizer

Plain Adam, learning rate 1e-2. Nine scalars is not a hard
optimization problem — we do not need schedulers, warmup, or fancy
techniques. A couple hundred gradient steps should be enough.

### 4.5 Compute budget

All on CPU. Training should complete in seconds to minutes. No GPU
needed. This is the main argument for starting minimal: the
experiment is cheap enough that if it fails, we have lost an hour,
not a week.

---

## 5. Failure Modes and Detection

### 5.1 "Training does not converge"

**What it looks like:** loss stays high, accuracy at Milestone C
baseline, trained scalars drift randomly.

**Diagnosis:** the 9-scalar scoring bias is insufficient to affect
which retrieval wins. The gap is not about memory trust — it is
about representation or something else.

**Action:** inspect the loss landscape. If it is flat, add a
refinement vector (2.2). If it is non-flat but does not improve
accuracy, the training data labels or the differentiable forward
pass is wrong.

### 5.2 "Training overfits task 2 but breaks task 3 (or vice versa)"

**What it looks like:** task 2 accuracy jumps to 99.5%, task 3
drops to 95%.

**Diagnosis:** the block learned a task-2-specific tie-breaking
strategy that is wrong for task 3. This is an interference
problem — the per-memory scoring biases cannot satisfy both tasks
simultaneously.

**Action:** train on a mixture of task 2 and task 3 stories from
the start, not sequentially. Also examine whether per-(memory, type)
affinity is more important than per-memory trust. If so, expand
the parameterization.

### 5.3 "Training converges but validation is at chance"

**What it looks like:** training accuracy 99%, validation 50%.

**Diagnosis:** the block is memorizing training stories via the
scoring biases, not learning a general rule. Unlikely with only
9 parameters, but possible if the evaluation metric is wrong.

**Action:** check whether training and validation stories are
drawn from the same distribution. Check whether the block's
`concepts` codebook is being re-initialized between stories.

### 5.4 "Task-specific beats learned block"

**What it looks like:** after training, learned block gets 98.8%
on task 2 and 98.5% on task 3. Better than Milestone C but still
below hand-coded 99.1% / 99.3%.

**Diagnosis:** the learned block captured most of the generalizable
tie-breaking knowledge, but task-specific hand-coding still beats
it because the hand-coded reasoner encodes task structure the block
does not see. This is expected and fine.

**Action:** accept. Declare success if learned block is within 0.5%
of hand-coded on both tasks. Go to Milestone E/F.

### 5.5 "C.1 regression"

**What it looks like:** after training on tasks 2 and 3, the block
gets worse on C.1 depth-controlled chains (was 100% at depth 5, now
90%).

**Diagnosis:** the learned scoring biases for tasks 2 and 3 are
pulling the block away from the neutral Milestone C behavior that
C.1 relies on.

**Action:** add C.1 stories to training, or regularize toward
Milestone C's unbiased behavior (add a penalty on learned scalars
deviating from 1.0).

---

## 6. Success Criteria

**Minimum viable result for Milestone D → G:**

- Learned block matches Milestone C on task 2 (≥ 98.5%)
- Learned block beats Milestone C on task 3 (≥ 99.0%)
- Learned block does not regress on Milestone C.1 (≥ 98% at
  depth 5, D = 2048)
- Learned parameters are interpretable (we can print the 9 scalars
  and reason about what they encode)

**Stretch result:**

- Learned block matches hand-coded on both tasks (≥ 99.0% on task 2,
  ≥ 99.3% on task 3)
- Learned block improves C.1 deeper than depth 5 (extended depth
  reach by learning something about chain traversal)
- Learned parameters reveal a structural finding (e.g. "training
  learned that holder_of's affinity for places is near zero, which
  matches the intuition that holder_of stores people, not places")

**Failure criteria (kill switch):**

- Learned block is worse than Milestone C on any task → design
  is flawed, revisit.
- Training is unstable across seeds → optimization problem, revisit
  temperature schedule and loss shape.

---

## 7. Open Questions for the Review

Things I am not confident about and would like the user (or a fresh
reading of Plate / Gayler) to verify before implementation.

1. **Is cleanup differentiability the right move?** Soft cleanup
   via softmax is the standard trick, but VSA literature may have
   a better-grounded differentiable cleanup. [verify against Plate
   ch. 5 or later Eliasmith SPA writing]

2. **Should per-memory trust scalars be in log-space?** Standard
   practice for scaling biases. Avoids negative values. Trivial to
   change later.

3. **Does the temperature schedule matter?** In transformer
   attention, annealing temperature is empirically load-bearing.
   For 9 scalars it may not matter.

4. **Can we skip the differentiable forward pass entirely?** With
   only 9 scalars, random search or black-box optimization (e.g.
   CMA-ES) might work as well as gradient descent and avoid the
   softmax-over-argmax awkwardness. Worth considering.

5. **What is the right temperature for inference?** Argmax is the
   obvious answer, but a slightly-soft inference might be more
   robust to cleanup noise.

6. **Is the 9-scalar design actually the simplest?** Could the gap
   be closed by a single learned scalar per memory (3 parameters,
   not 9)?

7. **Does the learned block still satisfy visited-exclusion and
   type-match halting?** By construction yes — we are only adding
   scoring biases, not changing the halting rule or candidate set.
   But worth explicitly checking in the implementation.

---

## 8. Deliverables for Milestone D

This document is the primary deliverable. Additional artifacts to
produce before Milestone G implementation:

- **Review from the user** — explicit yes / no on the minimal 2.1
  design and the training setup.
- **Verified HRR claims** — the user or a fresh Claude session
  should re-read Plate ch. 5 specifically for "differentiable
  cleanup" practices and cross-check the `[verify]` markers above.
- **A pseudocode sketch of the training loop** — not the final
  code, but enough to confirm the forward-pass differentiability
  is coherent before we commit to an implementation.

If the review signs off, Milestone G implementation is a short
focused burst of work on top of the existing `run_milestone_C.py`
codebase. No new files, no architectural change — just a new class
`LearnedReasoningBlock(ReasoningBlock)` that adds the scalars and
training logic.

---

## 9. What This Design Does NOT Address

Intentional omissions — deferred to later milestones:

- **Training against v14.1 embeddings.** This is Milestone F's
  job. If v14.1 embeddings are VSA-compatible, we retrain the
  learned block on top of them. If not, we handle that in F.
- **Natural language parsing.** We still assume clean event tuples
  from a generator. Closing this gap is a separate research
  direction (parser or end-to-end training from text).
- **Self-memory and Rung 4.** Deferred to Milestone H.
- **Open-ended queries.** The learned block here is still for
  type-bounded queries. Open-ended queries require Rung 1 proper,
  which is Milestone H.
- **Operator codebook (candidate 2.5).** Deferred until we know
  whether 2.1 is enough.

---

## 10. Summary Table

| Question | Answer |
|-|-|
| Is a learned block justified by the data? | Yes — 1.2% gap on task 3, 0.4% on task 2, both could plausibly close. |
| What is the minimal learnable addition? | 9 scalars: 3 per-memory trust + 6 per-(memory, type) affinity. |
| What changes in the forward pass? | Softmax over (memory, type) with temperature during training; argmax at inference. |
| What is the loss? | Cross-entropy between retrieved concept and true answer. |
| What is the training data? | bAbI task 2 and task 3 synthetic stories, same generators as Milestones B.1 / B.2. |
| What is the success criterion? | Match hand-coded within 0.5% on both tasks without regressing C.1. |
| What is the failure mode we care about most? | Training does not converge — would mean the gap is not about memory trust and we need to rethink. |
| What is the compute budget? | CPU, minutes. Cheap enough to rerun often. |
| What does this design NOT address? | v14.1 integration, natural language, self-memory, open-ended queries, operator codebook. |
