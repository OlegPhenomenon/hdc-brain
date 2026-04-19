"""
Milestone F.2 — Orthogonal subset selection on v14 codebook.

Milestone F.1 showed that using v14's learned codebook drop-in costs
7.5% accuracy on task 2 (chained reasoning). Diagnosis: v14 vectors
are semantically clustered (Phase 0 measured pairwise cosine std ~3x
higher than random bipolar), and chain reasoning amplifies this
cross-term noise across hops.

This milestone tests a simple fix: instead of allocating concept
vectors from a random permutation of the v14 pool, greedily pick a
subset of v14 vectors that are maximally orthogonal to each other.
The Cogito reasoning block sees only the 32 selected vectors and
never hits the noisy "neighboring concept" pairs.

If this closes the task 2 gap while still using real v14 vectors,
we have a clean way to reconcile "real learned embeddings" with
"deep-chain reasoning without semantic interference".
"""

from __future__ import annotations

import random
import time

import numpy as np

from hdc_lang import Codebook, cosine, bind, HDCMemory
from run_milestone_C import ReasoningBlock
from run_babi_task2 import Task2Reasoner, generate_task2
from run_babi_task3 import Task3Reasoner, generate_task3
from run_milestone_F1 import (
    load_v14_pool,
    PretrainedCodebook,
    make_t2_reasoner,
    make_t3_reasoner,
    eval_task2,
    eval_task3,
    eval_rung4,
)


def pick_orthogonal_subset(
    pool: np.ndarray,
    num: int,
    seed: int = 0,
    candidate_size: int = 2000,
) -> np.ndarray:
    """
    Greedy farthest-point selection: pick `num` vectors from `pool`
    that minimize the maximum pairwise cosine among the selected.

    To stay fast, we first sample `candidate_size` random tokens from
    the pool and run farthest-point selection inside that candidate set.
    Returns the indices into the original pool.
    """
    rng = np.random.default_rng(seed)
    cand_ids = rng.choice(pool.shape[0], size=candidate_size, replace=False)
    candidates = pool[cand_ids]

    # Normalize (bipolar norm is constant = sqrt(D), so cosine = dot / D)
    D = pool.shape[1]
    cos_mat = (candidates @ candidates.T) / float(D)

    # Start with an arbitrary candidate
    picked = [0]
    for _ in range(num - 1):
        # Max cosine from each candidate to the already-picked set
        max_to_picked = cos_mat[:, picked].max(axis=1)
        # Mask out already-picked
        for p in picked:
            max_to_picked[p] = np.inf
        # Pick the candidate with the smallest max-cosine — "farthest"
        next_idx = int(np.argmin(max_to_picked))
        picked.append(next_idx)

    selected = cand_ids[picked]
    return selected


class OrthogonalPretrainedCodebook:
    """
    Like PretrainedCodebook, but pre-selects a maximally-orthogonal
    subset of the pool as the allocation pool. Concepts are drawn in
    deterministic order from the subset.
    """

    def __init__(self, pool: np.ndarray, top_k: int = 32, seed: int = 0):
        subset_ids = pick_orthogonal_subset(pool, top_k, seed=seed)
        self.subset_pool = pool[subset_ids]
        self.subset_ids = subset_ids
        self.dim = pool.shape[1]
        self._next = 0
        self._vectors: dict[str, np.ndarray] = {}

    def add(self, name: str) -> np.ndarray:
        if name not in self._vectors:
            if self._next >= len(self.subset_pool):
                raise RuntimeError("orthogonal pool exhausted")
            self._vectors[name] = self.subset_pool[self._next]
            self._next += 1
        return self._vectors[name]

    def get(self, name: str) -> np.ndarray:
        return self._vectors[name]

    def __contains__(self, name: str) -> bool:
        return name in self._vectors

    def __len__(self) -> int:
        return len(self._vectors)

    def names(self) -> list[str]:
        return list(self._vectors.keys())

    def cleanup(self, noisy: np.ndarray, restrict_to=None) -> tuple[str, float]:
        candidates = list(restrict_to) if restrict_to is not None else list(self._vectors.keys())
        if not candidates:
            return "<empty>", 0.0
        best_name = candidates[0]
        best_sim = -float("inf")
        for name in candidates:
            sim = cosine(noisy, self._vectors[name])
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return best_name, best_sim


def measure_subset_pairwise(pool_subset: np.ndarray) -> dict:
    """Characterize how orthogonal a subset is."""
    D = pool_subset.shape[1]
    cos_mat = (pool_subset @ pool_subset.T) / float(D)
    n = cos_mat.shape[0]
    # Off-diagonal
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = cos_mat[mask]
    return {
        "n": n,
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "max_abs": float(np.abs(vals).max()),
        "frac_over_01": float((np.abs(vals) > 0.1).mean()),
        "frac_over_02": float((np.abs(vals) > 0.2).mean()),
    }


def main():
    print("=" * 72)
    print("Milestone F.2 — orthogonal subset selection on v14 codebook")
    print("=" * 72)

    print()
    print("Loading v14 pretrained codebook...")
    v14_pool = load_v14_pool()
    print(f"  shape: {v14_pool.shape}")

    DIM = v14_pool.shape[1]

    # Characterize subsets for comparison
    print()
    print("Pairwise cosine statistics on 32-vector subsets:")

    rng = np.random.default_rng(42)
    random_v14_ids = rng.choice(v14_pool.shape[0], size=32, replace=False)
    random_v14_subset = v14_pool[random_v14_ids]
    stats_random_v14 = measure_subset_pairwise(random_v14_subset)

    ortho_ids = pick_orthogonal_subset(v14_pool, 32, seed=42)
    ortho_subset = v14_pool[ortho_ids]
    stats_ortho = measure_subset_pairwise(ortho_subset)

    # Synthetic random bipolar for reference
    synth_rng = np.random.default_rng(7)
    synth_subset = synth_rng.choice([-1.0, 1.0], size=(32, DIM)).astype(np.float32)
    stats_synth = measure_subset_pairwise(synth_subset)

    def line(name: str, s: dict) -> str:
        return (f"  {name:<24s} mean={s['mean']:+.4f} "
                f"std={s['std']:.4f} "
                f"max|cos|={s['max_abs']:.4f} "
                f"%|>.1|={s['frac_over_01']:.3f}")

    print(line("random bipolar (synthetic)", stats_synth))
    print(line("v14 random subset", stats_random_v14))
    print(line("v14 orthogonal subset", stats_ortho))

    # Now run the actual tasks
    print()
    print("Running bAbI tasks...")
    rng_t2 = random.Random(42)
    stories_t2 = generate_task2(500, rng_t2)
    rng_t3 = random.Random(42)
    stories_t3 = generate_task3(500, rng_t3)

    def _randcb():
        return Codebook(DIM, seed=42)

    def _v14rand():
        return PretrainedCodebook(v14_pool, seed=42)

    def _v14ortho():
        return OrthogonalPretrainedCodebook(v14_pool, top_k=32, seed=42)

    # Task 2
    print()
    print("--- Task 2 ---")
    t0 = time.perf_counter()
    a_rand, n = eval_task2(stories_t2, _randcb(), DIM)
    print(f"  random bipolar     : {a_rand:.4f}  ({time.perf_counter()-t0:.1f}s)")
    t0 = time.perf_counter()
    a_v14r, _ = eval_task2(stories_t2, _v14rand(), DIM)
    print(f"  v14 random subset  : {a_v14r:.4f}  ({time.perf_counter()-t0:.1f}s)")
    t0 = time.perf_counter()
    a_v14o, _ = eval_task2(stories_t2, _v14ortho(), DIM)
    print(f"  v14 orthogonal     : {a_v14o:.4f}  ({time.perf_counter()-t0:.1f}s)")
    print(f"  ortho vs random v14 delta: {a_v14o - a_v14r:+.4f}")

    # Task 3
    print()
    print("--- Task 3 ---")
    t0 = time.perf_counter()
    a3_rand, n3 = eval_task3(stories_t3, _randcb(), DIM)
    print(f"  random bipolar     : {a3_rand:.4f}  ({time.perf_counter()-t0:.1f}s)")
    t0 = time.perf_counter()
    a3_v14r, _ = eval_task3(stories_t3, _v14rand(), DIM)
    print(f"  v14 random subset  : {a3_v14r:.4f}  ({time.perf_counter()-t0:.1f}s)")
    t0 = time.perf_counter()
    a3_v14o, _ = eval_task3(stories_t3, _v14ortho(), DIM)
    print(f"  v14 orthogonal     : {a3_v14o:.4f}  ({time.perf_counter()-t0:.1f}s)")
    print(f"  ortho vs random v14 delta: {a3_v14o - a3_v14r:+.4f}")

    # Summary
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Task 2:  random={a_rand:.4f}  v14-rand={a_v14r:.4f}  v14-ortho={a_v14o:.4f}")
    print(f"Task 3:  random={a3_rand:.4f}  v14-rand={a3_v14r:.4f}  v14-ortho={a3_v14o:.4f}")
    print()
    print("The gap we wanted to close (task 2, random bipolar -> v14):")
    print(f"  F.1 gap (random subset): {a_v14r - a_rand:+.4f}")
    print(f"  F.2 gap (orthogonal subset): {a_v14o - a_rand:+.4f}")
    if a_v14o > a_v14r:
        closed = (a_v14o - a_v14r) / (a_rand - a_v14r) * 100
        print(f"  Orthogonal selection closed {closed:.0f}% of the gap")


if __name__ == "__main__":
    main()
