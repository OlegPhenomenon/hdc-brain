"""
Milestone F.3 — Depth staircase on v14 orthogonal codebook.

Milestone C.1 proved that iteration count scales 1:1 with reasoning
depth on RANDOM bipolar vectors — a clean staircase from depth 1 to 20
with exactly `depth` iterations per query at D=2048.

This milestone re-runs the same depth test but with real v14 learned
vectors, using F.2's orthogonal subset selection to keep the concept
set clean. If the staircase still holds up to depth 20, we have
proven that deep multi-hop reasoning works end-to-end on a real
language model's embeddings — the full AGI-direction claim.

If the staircase falls short of depth 20 with v14, we learn exactly
where the capacity limit kicks in when the substrate is pretrained
rather than synthetic.
"""

from __future__ import annotations

import random
import time

import numpy as np

from hdc_lang import Codebook, HDCMemory, bind, cosine
from run_milestone_C import ReasoningBlock
from run_milestone_C1 import Chain, Story, generate_depth_stories, DepthEncoder
from run_milestone_F1 import load_v14_pool
from run_milestone_F2 import pick_orthogonal_subset


class PrecomputedOrthoCodebook:
    """Uses a pre-selected orthogonal subset pool with per-instance
    random permutation order. Cheap to instantiate — the expensive
    orthogonal selection is done once outside and passed in.
    """

    def __init__(self, subset_pool: np.ndarray, seed: int = 0):
        self.pool = subset_pool
        self.dim = subset_pool.shape[1]
        rng = np.random.default_rng(seed)
        self._perm = rng.permutation(len(subset_pool)).tolist()
        self._next = 0
        self._vectors: dict[str, np.ndarray] = {}

    def add(self, name: str) -> np.ndarray:
        if name not in self._vectors:
            if self._next >= len(self._perm):
                raise RuntimeError("orthogonal pool exhausted")
            tid = self._perm[self._next]
            self._next += 1
            self._vectors[name] = self.pool[tid]
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


class DepthEncoderWithPool(DepthEncoder):
    """DepthEncoder that uses a precomputed orthogonal pool instead of
    a fresh random codebook per instance.
    """

    def __init__(self, dim: int, decay: float, seed: int, ortho_pool: np.ndarray):
        # Reuse the base init minus the codebook creation
        self.dim = dim
        self.inside_of = HDCMemory(dim, decay=decay)
        self._items: set[str] = set()
        self._containers: set[str] = set()
        self._places: set[str] = set()
        self.concepts = PrecomputedOrthoCodebook(ortho_pool, seed=seed)


def evaluate_depth_v14(
    stories: list[Story],
    dim: int,
    max_iter: int,
    decay: float,
    conf_floor: float,
    ortho_pool: np.ndarray,
    seed_base: int = 42,
) -> dict:
    correct = 0
    total = 0
    iters_hist: dict[int, int] = {}

    for story_idx, story in enumerate(stories):
        enc = DepthEncoderWithPool(
            dim=dim, decay=decay, seed=seed_base + story_idx,
            ortho_pool=ortho_pool,
        )
        for chain in story.chains:
            enc.observe_chain(chain)

        block = ReasoningBlock(
            concepts=enc.concepts,
            memories={"inside_of": enc.inside_of},
            type_groups={
                "container": enc._containers,
                "place": enc._places,
            },
            conf_floor=conf_floor,
            max_iter=max_iter,
        )

        for leaf, expected in story.queries:
            pred, iters = block.solve(leaf, target_type="place")
            correct += int(pred == expected)
            total += 1
            iters_hist[iters] = iters_hist.get(iters, 0) + 1

    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "iters_hist": iters_hist,
    }


def main():
    print("=" * 72)
    print("Milestone F.3 — depth staircase on v14 orthogonal codebook")
    print("=" * 72)

    print()
    print("Loading v14 pool and precomputing orthogonal subset...")
    v14_pool = load_v14_pool()
    DIM = v14_pool.shape[1]

    t0 = time.perf_counter()
    ortho_ids = pick_orthogonal_subset(v14_pool, 80, seed=42, candidate_size=3000)
    ortho_pool = v14_pool[ortho_ids]
    print(f"  orthogonal subset size: {ortho_pool.shape}, "
          f"selection took {time.perf_counter() - t0:.1f}s")

    # Characterize the subset
    cos_mat = (ortho_pool @ ortho_pool.T) / float(DIM)
    mask = np.triu(np.ones_like(cos_mat, dtype=bool), k=1)
    vals = cos_mat[mask]
    print(f"  subset pairwise cos: mean={vals.mean():+.4f} "
          f"std={vals.std():.4f} max|cos|={np.abs(vals).max():.4f}")
    print()

    print("Running depth staircase at D=4096, decay=1.0 (pure sum):")
    print("Cogito concept codebook = v14 orthogonal subset")
    print()

    depths = [1, 2, 3, 5, 7, 10, 15, 20]
    max_iters = [1, 2, 3, 5, 10, 20, 25]

    # Build stories once per depth
    header = f"{'depth':>6}  " + "  ".join(f"iter={mi:>2}" for mi in max_iters)
    print(header)
    print("-" * len(header))

    results = {}
    for depth in depths:
        rng = random.Random(100 + depth)
        stories = generate_depth_stories(
            num_stories=150,
            depth=depth,
            rng=rng,
            chains_per_story=3,
        )
        row = f"{depth:>6}  "
        results[depth] = {}
        for max_iter in max_iters:
            r = evaluate_depth_v14(
                stories,
                dim=DIM,
                max_iter=max_iter,
                decay=1.0,
                conf_floor=0.03,
                ortho_pool=ortho_pool,
            )
            results[depth][max_iter] = r
            row += f"  {r['accuracy']:>6.3f}"
        print(row)

    print()
    print("Iteration histograms at max_iter=25:")
    for depth in depths:
        r = results[depth][25]
        # Only show the most-used iteration count for brevity
        hist = r["iters_hist"]
        # Top entry
        top_k = sorted(hist.items(), key=lambda kv: -kv[1])[:3]
        top_str = ", ".join(f"{k} iter: {v}" for k, v in top_k)
        print(f"  depth={depth:>2}: acc={r['accuracy']:.3f}  top-3: {top_str}")

    # Check staircase
    print()
    print("Staircase check:")
    clean = True
    for depth in depths:
        for mi in max_iters:
            if mi < depth and results[depth][mi]["accuracy"] > 0.2:
                clean = False
                print(f"  unexpected: depth={depth}, iter={mi}, acc={results[depth][mi]['accuracy']:.3f}")
            if mi >= depth and mi <= 25 and results[depth][mi]["accuracy"] < 0.8:
                clean = False
                print(f"  deficit: depth={depth}, iter={mi}, acc={results[depth][mi]['accuracy']:.3f}")
    if clean:
        print("  CLEAN staircase up to depth 20 on v14 orthogonal codebook.")
        print("  Deep multi-hop reasoning on real learned embeddings is VALIDATED.")
    else:
        print("  Staircase has deviations — see warnings above.")


if __name__ == "__main__":
    main()
