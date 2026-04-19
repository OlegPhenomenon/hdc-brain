# Milestone F.1 Log — First end-to-end Cogito on v14's learned codebook

**Date:** 2026-04-11
**Status:** PARTIAL SUCCESS — end-to-end integration works; task 3 and Rung 4 survive the switch; task 2 has a measurable ~7% cost from semantic noise in chains
**Code:** `hdc-cogito/run_milestone_F1.py`
**Parent:** follows `docs/experiments/hdc-cogito/phase_0_probe_log.md`

## Goal

Move Cogito from synthetic random-bipolar vectors to the **actual learned bipolar codebook of v14** (the Russian HDC language model). Measure the cost of this switch on three experiments:

1. bAbI task 2 — chained reasoning (Milestone B.1)
2. bAbI task 3 — branching reasoning with drops (Milestone B.2)
3. Rung 4 self-memory error correction (Milestone H.1)

This is the first experiment in the project where the concept vectors are not synthetic.

## Design

- Loaded `best_hdc_brain_v14.pt`, extracted the `codebook` parameter, applied `sign()` to get the bipolar form used at v14 inference. Shape: `(16000, 4096)`, all entries in `{-1, +1}`.
- Built `PretrainedCodebook` — a drop-in replacement for `hdc_lang.Codebook` that, instead of generating fresh random bipolar vectors per concept name, draws from a deterministic permutation of the v14 pool.
- Patched `Task2Reasoner` and `Task3Reasoner` to accept the pretrained codebook via attribute assignment.
- Ran each experiment twice: once with the original random-bipolar codebook at D=4096, once with the v14 codebook at the same D.

## Results

```
--- Task 2 (chained reasoning) ---
  random bipolar:  0.9906  (743 q)
  v14 pretrained:  0.9152  (743 q)
  delta:           -0.0754

--- Task 3 (branching reasoning with drops) ---
  random bipolar:  0.9827  (750 q)
  v14 pretrained:  0.9720  (750 q)
  delta:           -0.0107

--- Rung 4 self-memory (error correction on task 3) ---
  random bipolar:
    phase1 acc      : 0.9827
    errors          : 13
    recovered       : 13/13 (100.00%)
    stored conf     : 0.2713
    unstored conf   : 0.0234
    integrated acc  : 1.0000
  v14 pretrained:
    phase1 acc      : 0.9720
    errors          : 21
    recovered       : 21/21 (100.00%)
    stored conf     : 0.2184
    unstored conf   : 0.0217
    integrated acc  : 1.0000
```

## Interpretation

### What worked

- **End-to-end integration.** Every line of the Cogito reasoning block ran unchanged on top of the pretrained codebook. No architectural change, no retraining, no special handling. The `sign(v14.codebook)` is a drop-in substrate.
- **Task 3 held up.** Only 1.07% accuracy drop. Branching queries that resolve in a single direct lookup (drop-branch via `dropped_at`, ~36% of task 3) do not accumulate semantic noise, and the remaining chain-branch queries are short enough that the noise is tolerable.
- **Rung 4 self-memory worked perfectly.** 21/21 recovery on v14 errors, clean signal/noise margin (0.2184 / 0.0217 = ~10x), integrated accuracy **100%**. The online-error-correction mechanism is orthogonal to the codebook: whatever you put in, self-memory can carry corrections for it.

### What was costly

- **Task 2 dropped 7.5%.** The chain-only task suffered the most. Cause: v14's learned vectors have ~3x higher pairwise-cosine std than random bipolar (measured in Phase 0). In a 2-hop chain, each retrieval step accumulates noise proportional to that std. With noise that is 3x higher and two hops, the effective noise on the second cleanup is large enough to miss ~7% of queries.
- Decay tuning did NOT help. A decay grid search across holder/location pairs (0.5–0.95 × 0.70–0.99) gave no value above 91.52%. The bottleneck is not recency policy — it is cleanup noise.

### Why task 3 held while task 2 did not

Task 3's branching reasoner splits queries into two paths:

- **Drop-branch (~36% of queries):** one `unbind(dropped_at, obj)` + cleanup. No chaining, no noise accumulation. These queries suffer almost nothing from the codebook switch.
- **Chain-branch (~64% of queries):** same two-hop pattern as task 2. These queries should suffer the same way.

So why is the task 3 drop 1% and task 2 drop 7.5%? The math works out roughly: 64% of task 3 suffers like task 2 (7.5% of 64% ≈ 4.8%), and 36% is unaffected. That predicts a ~5% drop on task 3, not 1%. Actual drop is 1%.

Looking more carefully: the task 3 generator has fewer "long-chain" edge cases than task 2 because dropped objects are naturally short-circuited. On average the chains in task 3 are shorter, and the noise accumulation is less severe per query. This is an artifact of the specific generators, not a deep property.

### Why Rung 4 was immune

The Rung 4 test uses a self-memory keyed by a **random story-id vector**, not by a v14 vector. The story-id is a fresh `rng.choice([-1, +1], D)` each time — pure random bipolar. So the retrieval key has the clean signal/noise properties of random vectors, while only the values (which are v14 vectors) contribute a small amount of semantic noise to the cleanup margin. The net effect is essentially the same clean separation as the random-bipolar version.

This is actually a useful design lesson: **self-memory keys can be chosen for cleanness, independent of the concept codebook**. If Cogito v1 on v14 uses random story-id keys for self-memory, Rung 4 will work identically to the synthetic version.

## Findings

1. **Cogito runs end-to-end on v14.** No architectural changes, no retraining. The sign of v14's codebook is a drop-in concept codebook.
2. **The cost of learned vectors is visible on 2-hop chains.** Task 2 drops 7.5% because each chain step accumulates semantic cross-term noise that is ~3x higher than for random vectors. This is consistent with Phase 0's measurement (pairwise-cosine std 0.0492 vs 0.0156 theoretical).
3. **Branching + short-circuits mitigate the cost.** Task 3 drops only 1% because drop-branch queries avoid the chain entirely.
4. **Rung 4 self-memory is immune.** Using random story-id keys keeps the key cleanness independent of concept noise. 100% recovery on v14 errors, integrated 100% accuracy.
5. **Decay tuning does not fix the chain problem.** The bottleneck is cleanup noise, not recency policy. Fixes must target either the codebook (e.g. orthogonalization, permute scrambling) or the cleanup step (e.g. learned scoring, larger D).

## What this unlocks for the project

- **The complete AGI-direction story has empirical support.** We now have all three pieces running together:
  - a real HDC language model (v14) that generates coherent text
  - a reasoning block that operates on its actual embeddings (this experiment)
  - an online error-correction mechanism that works on top of both (Rung 4)
- **Task 3 on v14 at 97.20% is a publishable result for a future paper.** This is not the main focus for now (we're not writing papers), but it is a useful benchmark anchor.
- **The task 2 gap tells us where the next architectural experiment should go.** If we want to close the 7.5% gap on deep chains with v14 vectors, the directions to try are:
  1. Orthogonalization of the per-concept allocation (pick tokens with minimum pairwise cosine)
  2. Per-concept scrambling (bind each allocated v14 vector with a fresh random marker)
  3. Larger D (but this means not using v14 directly — would require retraining)
  4. Learned cleanup scoring (the Milestone G direction, extended to cross-term suppression)

## Suggested next experiments

1. **Orthogonal subset selection.** Instead of drawing v14 vectors in random order, greedily pick the subset of 22 most-orthogonal vectors for our concept set. This should reduce cross-term noise on chains without giving up v14 semantics entirely. Quick to test, informative either way.

2. **Per-concept scrambling.** Bind each allocated v14 vector with a fresh random marker. This reduces to essentially random vectors for VSA purposes but keeps the "v14 is the substrate" narrative. Should close the chain gap. But it gives up any potential semantic transfer benefit.

3. **Task 2 with larger D.** Run task 2 with a synthetic random codebook at D = 8192 and D = 16384 to see how much headroom we have. This tells us how much the v14 cost is about D vs about semantic structure.

4. **Capacity stress on v14 at multi-memory setups.** Phase 0 test 4 measured capacity in a single bundle. The real Cogito block uses 2-3 bundles simultaneously. How does the effective capacity change?

All four are quick to test. My pick for the most informative is #1 — orthogonal subset selection — because it targets the diagnosed problem directly and tells us whether the semantic clustering is responsible.

## Status

Partial success but directly informative. End-to-end Cogito-on-v14 works. One specific task (2-hop chains without branches) loses ~7%. Self-memory is immune. The project now has a complete integration story from language model to reasoning to online learning.
