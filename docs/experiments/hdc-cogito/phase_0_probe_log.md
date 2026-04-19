# Phase 0 Probe Log — VSA primitives on v14 Russian codebook

**Date:** 2026-04-11
**Status:** GATE PASSED — VSA primitives work cleanly on v14's learned bipolar embeddings
**Code:** `hdc-cogito/run_phase0_v14_probe.py`
**Model:** `hdc-brain-v14/best_hdc_brain_v14.pt` (Russian pretraining, 75500 iterations, vocab 16000, dim 4096)

## Goal

This was the **single biggest open risk** in the project for multiple milestones:

> Do HDC/VSA primitives actually work when the concept vectors come from a real trained language model, rather than random bipolar vectors?

Every Cogito experiment up to this point used random bipolar vectors. This proved the architecture works in the abstract. It did not prove anything about real learned embeddings, which have semantic structure (similar words → similar vectors). That structure could in principle break VSA memory by increasing cross-term noise.

The test became possible when Oleg pointed out he already has v14 (the earlier Russian pretraining of the same HDC architecture) sitting locally. v14 generates coherent Russian text and has a fully trained bipolar codebook.

## Setup

1. Load `best_hdc_brain_v14.pt`.
2. Extract the `codebook` parameter — `(16000, 4096)` float32.
3. Apply `sign(codebook)` to get the bipolar form used at inference. All 16000 × 4096 entries verified to be in `{-1, +1}`.
4. Run four tests on these learned vectors using the same primitives from `hdc_lang.py`.

Also loaded `bpe_ru_16k.model` to peek at what the tokens are:

```
token      0: '<unk>'
token    500: '▁ста'
token   1000: '▁этом'
token   5000: '▁говорят'
token  10000: '▁нужда'
token  15000: '▁gentlemen'
```

BPE tokens — some are partial words, some are whole short words, and there are English loan-words in the tail of the vocabulary.

## Test 1 — Pairwise cosine distribution

2000 tokens sampled at random, 1,999,000 pairs.

| Metric | Learned v14 | Random bipolar (expected) |
|-|-|-|
| Mean cosine         | +0.0235 | 0 |
| Std cosine          | 0.0492  | 1/√4096 ≈ 0.0156 |
| Fraction \|cos\| > 0.1 | 6.11% | <0.01% |
| Fraction \|cos\| > 0.2 | 0.30% | ~0% |
| Max \|cos\|            | 0.7612 | ≈0.04 |

**Interpretation.** The learned codebook is **semantically clustered but still mostly orthogonal**. Its pairwise-cosine standard deviation is ~3x that of random bipolar, and ~6% of pairs have nontrivial similarity, but 94% are essentially orthogonal. The max of 0.76 is a near-duplicate pair (probably BPE fragments of the same word stem).

For VSA memory, what matters is the cross-term noise level when many bindings are bundled. A std of 0.05 means each unrelated cross-term contributes ~0.05 cosine noise, versus ~0.016 for random. Capacity will be lower, but the primitives still work.

## Test 2 — Bind round-trip

`bind(bind(a, b), b) == a` for 100 random pairs of learned vectors.

**100 / 100 passes, zero failures.**

The bipolar algebra is preserved exactly by `sign(codebook)`. Nothing weird happens.

## Test 3 — Holographic bundle retrieval (the crucial one)

For each run: pick 20 random tokens as "keys" and 20 other random tokens as "values". Build one bundle vector `bundle = Σ bind(key_i, value_i)`. For each key, unbind and clean up against **the entire 16000-token vocabulary**. Record whether the correct value token wins.

20 independent runs, 400 total retrievals.

```
overall recovery: 400/400 = 100.00%
per-run recovery mean: 100.00%
per-run recovery std : 0.00%
avg cosine on CORRECT retrievals: 0.2268
```

**Every single retrieval hit the right token.** Out of 16000 possible words, cleanup correctly identified the right one every time. Average retrieval cosine of 0.2268 is well above the noise floor from Test 1 (0.05 std). Clean signal-to-noise margin of ~5x.

**This is the single most important result in the project so far.** It means that v14's learned embeddings are **drop-in compatible** with the Cogito reasoning primitives. We do not need to retrain the codebook. We do not need to cluster or project or re-bipolarize. The `sign()` of the existing trained vectors is all we need.

## Test 4 — Capacity curve

How many key-value pairs can we pack into a single bundle before recovery collapses?

| Pairs in bundle | Recovery rate |
|-|-|
| 5   | 100.00% |
| 10  | 100.00% |
| 20  | 100.00% |
| 40  | 72.50%  |
| 80  | 0.42%   |
| 160 | 0.00%   |
| 320 | 0.00%   |

**Empirical capacity: ~30-50 pairs per bundle at D=4096.**

For comparison, the theoretical capacity for random bipolar at this dimension is ~400 pairs. Learned vectors lose ~10x because their semantic clustering raises the noise floor. But 30-50 is still plenty for the kinds of multi-hop reasoning we do in tasks 2/3, which use at most 10-20 bindings per memory bank.

The collapse is sharp (between 40 and 80). This is consistent with the classical Kanerva capacity result: performance is flat below some threshold, then drops precipitously.

## Findings

1. **Learned embeddings are near-orthogonal.** Test 1 confirms that v14's semantic structure is mild compared to the orthogonality budget of a 4096-dim bipolar space. VSA operations work in this regime.

2. **Bind/Unbind is exact on learned vectors.** Test 2 verifies the bipolar algebra survives the `sign()` projection of trained floats. No surprises.

3. **Holographic memory retrieves 100% at reasonable load.** Test 3 is the critical result: 20 bindings in one bundle, 100% retrieval across the full 16000-token vocabulary. This is the functional property Cogito needs.

4. **Empirical capacity is ~30-50 pairs at D=4096.** Test 4 quantifies the headroom. Well within the needs of bAbI-class tasks; probably adequate for longer multi-step reasoning; would need attention for very complex tasks with many facts.

5. **The single biggest open risk in the project is now resolved positively.** Phase 0 was the gating question of "can Cogito attach to v14 at all". The answer is yes, and the attachment mechanism is trivial — apply `sign()` to the trained codebook and use it as the Cogito concept codebook.

## What this unlocks

- **End-to-end Cogito-on-v14 experiments are now possible.** We can run the reasoning block from Milestone C on stories built with v14's actual Russian tokens, and the block will do chained/branching reasoning over real learned embeddings.
- **The bAbI-style task generator can be rewritten in Russian** using tokens from the v14 vocabulary. This is a natural next step — instead of inventing English names "Mary", "apple", "office", we use v14's "Маша", "яблоко", "офис" (whatever BPE tokens match).
- **Rung 4 self-memory carries over.** The same self-memory experiment from Milestone H.1 will work on top of v14's embeddings — the memory mechanism is orthogonal to the codebook choice.
- **The learned reasoning block (Milestone G.1) also carries over.** The 6 scalars are per-memory trust, not per-codebook, so they work independently of the codebook semantics.

## Implications for the proposal

Small but important updates that should be folded back into `docs/planning/hdc-cogito-proposal.md`:

- **§3 Hypotheses** — H1 (structural representation) gets a validation mark: "Confirmed on v14 Russian embeddings: 100% holographic retrieval at 20-pair bundle capacity, empirical capacity ~30-50 pairs."
- **§8.2 Phase 0** — status update: "DONE via Phase 0 probe on v14 Russian. Codebook is directly usable; no retraining required. See `docs/experiments/hdc-cogito/phase_0_probe_log.md`."
- **§11 Risks** — "v14 embeddings don't cluster into clean concepts" can be downgraded or struck through. The clusters are mild enough that VSA capacity is only reduced by ~10x, not broken.
- **§14 Open questions** — update Q5 (biggest open risk) to: "Answered: v14 embeddings ARE VSA-compatible. Capacity is the new concern, not compatibility."

Not editing proposal immediately — will do in one consolidation pass after a couple more experiments.

## Next steps

1. **Build a Russian version of tasks 2/3** using v14's actual BPE tokens, run the Cogito reasoning block end-to-end. This is the first true Cogito + v14 integration experiment.
2. **Test Rung 4 self-memory on v14 tokens.** Does the Milestone H.1 result survive the move from random vectors to learned ones?
3. **Test capacity limits in multi-memory setups** — not a single bundle but the 3-4 memory banks the reasoning block actually uses. We need to know the REAL capacity under the full Cogito load.
4. **Explore semantic transfer.** Can we use v14's pretrained similarity to help with tasks? E.g., "Маша" and "Миша" probably have similar vectors; does that help or hurt chained reasoning?

## Status

**Phase 0: PASSED.** v14 is a viable substrate for Cogito. No retraining needed. The architecture plugs in directly. This closes the biggest open question that has been blocking the project for multiple milestones.
