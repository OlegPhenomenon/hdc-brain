"""
Phase 0 Probe — Do VSA primitives work on v14's LEARNED bipolar codebook?

Up to this point, every Cogito experiment has used random bipolar
vectors: sampled from {-1, +1} with no structure. Those experiments
proved the ARCHITECTURE works. What they did not prove is that the
architecture still works when the concept vectors come from a real
trained language model rather than a synthetic codebook.

v14 is the earlier version of Oleg's HDC language model, trained on
Russian, produces coherent Russian text, and uses a straight-through
bipolar codebook: the `codebook` parameter is trained as float, then
signed at each forward pass. So the effective vectors at inference
are `sign(codebook)` — literal bipolar vectors in {-1, +1}^4096.

These trained bipolar vectors have STRUCTURE. Unlike random vectors
they should cluster semantically: tokens that appear in similar
contexts get similar vectors. The question is whether VSA primitives
(Bind, Unbind, Bundle, cleanup) survive this structure.

The tests:

  1. Pairwise-cosine distribution. For random bipolar vectors the
     mean pairwise cosine is 0 and the std is 1/sqrt(D) ≈ 0.016.
     For learned bipolar vectors we expect a wider distribution —
     that is what SEMANTIC STRUCTURE looks like. The question is HOW
     wide. If pairs of unrelated tokens still have cosine near 0,
     VSA interference stays manageable. If most pairs have cosine
     > 0.2, VSA memories will be noisy.

  2. Bind round-trip. In bipolar math, `bind(bind(a, b), b) == a`
     unconditionally. This should hold trivially. But we verify it
     numerically on the learned vectors — a sanity check that the
     sign operation really does what we think.

  3. Holographic storage retrieval. This is the real test. Take 20
     random tokens as "keys", 20 other random tokens as "values".
     Build a bundle `sum Bind(key_i, value_i)`. For each key, unbind
     and clean up against the full vocabulary. Does the right value
     come back? How many of 20 are recovered cleanly?

     If test 3 succeeds, v14's learned embeddings are drop-in
     compatible with the Cogito reasoning block. We can literally
     use them as the concept codebook.

     If test 3 fails, the learned embeddings have enough semantic
     correlation to interfere with VSA memory. In that case we
     either need a post-processing step (e.g., random permutation,
     re-rotation) or we accept that Cogito needs its own codebook
     seeded from but not equal to v14's.
"""

from __future__ import annotations

import random

import numpy as np
import torch


CKPT_PATH = "/Users/oleghasjanov/Documents/learning/hoffman_swarm/hdc-brain-v14/best_hdc_brain_v14.pt"
TOKENIZER_PATH = "/Users/oleghasjanov/Documents/learning/hoffman_swarm/hdc-brain-v14/bpe_ru_16k.model"


def load_bipolar_codebook() -> tuple[np.ndarray, int, int]:
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    codebook = ckpt["model"]["codebook"].numpy()
    vocab_size, dim = codebook.shape
    # Apply the same straight-through sign that the model uses at inference.
    # Zero entries are pushed to +1 deterministically.
    bipolar = np.sign(codebook).astype(np.float32)
    bipolar[bipolar == 0] = 1.0
    return bipolar, vocab_size, dim


def load_tokenizer():
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    return sp


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_matrix(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Pairwise cosines between two sets of bipolar vectors."""
    rn = rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12)
    cn = cols / (np.linalg.norm(cols, axis=1, keepdims=True) + 1e-12)
    return rn @ cn.T


def test_1_pairwise_distribution(codebook: np.ndarray, num_samples: int = 2000):
    """Distribution of pairwise cosines between random tokens."""
    print("Test 1: pairwise cosine distribution between random tokens")
    rng = np.random.default_rng(7)
    idx = rng.choice(codebook.shape[0], size=num_samples, replace=False)
    sub = codebook[idx]
    # Compute lower triangle of pairwise cosines (excluding diagonal)
    sims = cosine_matrix(sub, sub)
    n = sims.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = sims[mask]

    expected_std = 1.0 / np.sqrt(codebook.shape[1])
    print(f"  samples: {num_samples} tokens, {len(vals)} pairs")
    print(f"  mean cosine    : {vals.mean():+.4f}")
    print(f"  std cosine     : {vals.std():.4f}")
    print(f"  expected std for random bipolar (1/sqrt(D)): {expected_std:.4f}")
    print(f"  fraction |cos| > 0.1 : {(np.abs(vals) > 0.1).mean():.4f}")
    print(f"  fraction |cos| > 0.2 : {(np.abs(vals) > 0.2).mean():.4f}")
    print(f"  max |cos|      : {np.abs(vals).max():.4f}")
    return vals


def test_2_bind_round_trip(codebook: np.ndarray, num_trials: int = 100):
    """bind(bind(a, b), b) should equal a for all learned pairs."""
    print()
    print("Test 2: bind round-trip on learned bipolar pairs")
    rng = np.random.default_rng(13)
    failures = 0
    for _ in range(num_trials):
        i, j = rng.choice(codebook.shape[0], size=2, replace=False)
        a = codebook[i]
        b = codebook[j]
        bound = a * b
        recovered = bound * b
        if not np.allclose(recovered, a):
            failures += 1
    print(f"  trials: {num_trials}")
    print(f"  failures: {failures}")
    print(f"  pass rate: {(num_trials - failures) / num_trials * 100:.2f}%")


def test_3_holographic_bundle(codebook: np.ndarray, num_pairs: int = 20, num_runs: int = 20):
    """
    Store `num_pairs` key->value bindings in a single bundle vector.
    Retrieve each value by unbinding with the corresponding key,
    then cleaning up against the full vocabulary.

    Run this `num_runs` times with different random key/value draws and
    report recovery statistics.
    """
    print()
    print(f"Test 3: holographic bundle retrieval on learned vectors")
    print(f"  storing {num_pairs} key->value pairs in one bundle")
    print(f"  cleanup against the full {codebook.shape[0]}-token vocabulary")
    print(f"  {num_runs} independent runs")
    print()

    rng = np.random.default_rng(31)
    total_tries = 0
    total_correct = 0
    recovery_rates = []
    cleanup_cosines_correct = []
    cleanup_cosines_winner = []

    for run in range(num_runs):
        # Pick 2 * num_pairs distinct token ids
        ids = rng.choice(codebook.shape[0], size=2 * num_pairs, replace=False)
        keys_idx = ids[:num_pairs]
        values_idx = ids[num_pairs:]
        keys = codebook[keys_idx]
        values = codebook[values_idx]

        # Build the bundle
        bundle = np.zeros(codebook.shape[1], dtype=np.float32)
        for k, v in zip(keys, values):
            bundle += k * v

        # For each key, unbind and cleanup
        run_correct = 0
        for i in range(num_pairs):
            retrieved = bundle * keys[i]
            # Cleanup: cosine against every token in vocab
            sims = codebook @ retrieved
            sims /= np.linalg.norm(retrieved) + 1e-12
            sims /= np.sqrt(codebook.shape[1])  # each bipolar row has norm sqrt(D)
            winner = int(np.argmax(sims))
            if winner == values_idx[i]:
                run_correct += 1
                cleanup_cosines_correct.append(float(sims[values_idx[i]]))
            cleanup_cosines_winner.append(float(sims[winner]))

        total_tries += num_pairs
        total_correct += run_correct
        recovery_rates.append(run_correct / num_pairs)

    overall = total_correct / total_tries
    print(f"  overall recovery: {total_correct}/{total_tries} = {overall * 100:.2f}%")
    print(f"  per-run recovery mean: {np.mean(recovery_rates) * 100:.2f}%")
    print(f"  per-run recovery std : {np.std(recovery_rates) * 100:.2f}%")
    if cleanup_cosines_correct:
        print(f"  avg cosine on CORRECT retrievals: {np.mean(cleanup_cosines_correct):.4f}")
    if cleanup_cosines_winner:
        print(f"  avg cosine on WINNER (any)       : {np.mean(cleanup_cosines_winner):.4f}")


def test_4_bundle_capacity_sweep(codebook: np.ndarray):
    """
    Sweep bundle size from 5 to 200, see how recovery drops with capacity.
    This is an empirical Kanerva capacity curve for the LEARNED vectors.
    """
    print()
    print("Test 4: capacity sweep — how many pairs can a single bundle hold?")
    print()
    print(f"  {'num_pairs':>12} {'recovery':>12}")
    rng = np.random.default_rng(101)
    for num_pairs in [5, 10, 20, 40, 80, 160, 320]:
        total = 0
        correct = 0
        runs = max(3, 60 // num_pairs) if num_pairs <= 60 else 3
        for _ in range(runs):
            ids = rng.choice(codebook.shape[0], size=2 * num_pairs, replace=False)
            keys_idx = ids[:num_pairs]
            values_idx = ids[num_pairs:]
            keys = codebook[keys_idx]
            values = codebook[values_idx]
            bundle = np.zeros(codebook.shape[1], dtype=np.float32)
            for k, v in zip(keys, values):
                bundle += k * v
            for i in range(num_pairs):
                retrieved = bundle * keys[i]
                sims = codebook @ retrieved
                winner = int(np.argmax(sims))
                total += 1
                if winner == values_idx[i]:
                    correct += 1
        rate = correct / max(total, 1)
        print(f"  {num_pairs:>12} {rate * 100:>11.2f}%")


def main():
    print("=" * 72)
    print("Phase 0 Probe — VSA primitives on v14 learned bipolar codebook")
    print("=" * 72)

    print()
    print("Loading v14 checkpoint...")
    codebook, vocab_size, dim = load_bipolar_codebook()
    print(f"  codebook shape: {codebook.shape}")
    print(f"  first token vector (first 16 dims): {codebook[0, :16]}")
    print(f"  all entries in {{-1, +1}}?: {np.all(np.isin(codebook, [-1.0, 1.0]))}")
    print()

    # Quick peek at what tokens mean
    try:
        sp = load_tokenizer()
        print("Sample token inventory:")
        for tid in [0, 1, 2, 100, 500, 1000, 5000, 10000, 15000]:
            if tid < vocab_size:
                piece = sp.id_to_piece(tid)
                print(f"  token {tid:>6}: {repr(piece)}")
        print()
    except Exception as e:
        print(f"  (tokenizer peek failed: {e})")
        print()

    test_1_pairwise_distribution(codebook)
    test_2_bind_round_trip(codebook)
    test_3_holographic_bundle(codebook, num_pairs=20, num_runs=20)
    test_4_bundle_capacity_sweep(codebook)

    print()
    print("=" * 72)
    print("What the tests tell us (interpret with care)")
    print("=" * 72)
    print(
        "\n  - If test 1 shows std ~ 1/sqrt(D) and few |cos| > 0.2 pairs,"
        "\n    then the learned codebook is nearly orthogonal and VSA memory"
        "\n    will behave similarly to random bipolar. Test 3 should succeed."
        "\n  - If test 1 shows many |cos| > 0.3 pairs, the codebook is"
        "\n    semantically clustered. VSA interference rises with cluster density."
        "\n    Test 3 recovery rate quantifies how severe this is."
        "\n  - Test 4 gives an empirical capacity curve: we learn how many"
        "\n    key-value pairs we can pack into one bundle before collapse."
        "\n    For random bipolar at D=4096, theory says ~400. For learned"
        "\n    vectors the number should be a little lower — we see by how much."
    )


if __name__ == "__main__":
    main()
