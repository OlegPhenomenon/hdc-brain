"""
Milestone F.1 — First end-to-end Cogito experiment on v14's learned codebook.

Replaces the reasoner's random bipolar Codebook with a selection of real
vectors from v14 (the Russian HDC language model). Runs three earlier
experiments on top of the learned codebook:

  1. bAbI task 2 (chained reasoning via holder_of -> location_of)
  2. bAbI task 3 (branching reasoning with DROPPED markers + dropped_at)
  3. Rung 4 self-memory error correction (from Milestone H.1)

Each test is also run with the original random bipolar codebook as a
baseline. The difference between the two runs measures the cost (or,
hopefully, non-cost) of switching from synthetic to real embeddings.

If the deltas are small, v14 is a drop-in substrate for Cogito and we
have the first true end-to-end integration: our reasoning layer on top
of Oleg's real HDC language model.

Dimensionality: 4096 (matches v14). Codebook vectors are drawn from
v14's 16000 pretrained bipolar vectors (i.e. sign(v14.codebook)).
"""

from __future__ import annotations

import random
import time
from typing import Optional

import numpy as np
import torch

from hdc_lang import Codebook, HDCMemory, bind, cosine
from run_milestone_C import ReasoningBlock
from run_babi_task2 import Task2Reasoner, generate_task2
from run_babi_task3 import Task3Reasoner, generate_task3


V14_CKPT = "/Users/oleghasjanov/Documents/learning/hoffman_swarm/hdc-brain-v14/best_hdc_brain_v14.pt"


def load_v14_pool() -> np.ndarray:
    ckpt = torch.load(V14_CKPT, map_location="cpu", weights_only=False)
    raw = ckpt["model"]["codebook"].numpy()
    pool = np.sign(raw).astype(np.float32)
    pool[pool == 0] = 1.0
    return pool


class PretrainedCodebook:
    """Codebook that draws from a pool of pretrained bipolar vectors
    instead of sampling fresh randomness.

    Drop-in replacement for hdc_lang.Codebook — same interface
    (add, get, cleanup, __contains__, __len__, names).

    Each new concept name gets the next token in a deterministic
    (seeded) permutation of the pool. Once assigned, the mapping is
    stable for the lifetime of this codebook.
    """

    def __init__(self, pool: np.ndarray, seed: int = 0):
        self.pool = pool
        self.dim = pool.shape[1]
        rng = np.random.default_rng(seed)
        self._perm = rng.permutation(pool.shape[0]).tolist()
        self._next = 0
        self._vectors: dict[str, np.ndarray] = {}

    def add(self, name: str) -> np.ndarray:
        if name not in self._vectors:
            if self._next >= len(self._perm):
                raise RuntimeError("pretrained pool exhausted")
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


def make_t2_reasoner(dim: int, codebook) -> Task2Reasoner:
    enc = Task2Reasoner(dim=dim, decay_holder=0.7, decay_location=0.8, seed=42)
    enc.concepts = codebook
    return enc


def make_t3_reasoner(dim: int, codebook) -> Task3Reasoner:
    enc = Task3Reasoner(dim=dim, decay_holder=0.7, decay_location=0.8, decay_drop=0.8, seed=42)
    enc.concepts = codebook
    # DROPPED marker was created from the original Codebook inside __init__,
    # re-resolve it against the new codebook so it points to a v14 vector.
    enc.dropped_marker = enc.concepts.add(enc.DROPPED_NAME)
    return enc


def eval_task2(stories, codebook, dim: int) -> tuple[float, int]:
    enc = make_t2_reasoner(dim, codebook)
    correct = 0
    total = 0
    for story in stories:
        enc.reset()
        for event in story.events:
            enc.observe(event)
        block = ReasoningBlock(
            concepts=enc.concepts,
            memories={"holder_of": enc.holder_of, "location_of": enc.location_of},
            type_groups={"person": enc._people, "place": enc._places},
            max_iter=5,
        )
        for obj, expected in story.queries:
            pred, _ = block.solve(obj, target_type="place")
            correct += int(pred == expected)
            total += 1
    return correct / max(total, 1), total


def eval_task3(stories, codebook, dim: int) -> tuple[float, int]:
    enc = make_t3_reasoner(dim, codebook)
    correct = 0
    total = 0
    for story in stories:
        enc.reset()
        for event in story.events:
            enc.observe(event)
        block = ReasoningBlock(
            concepts=enc.concepts,
            memories={
                "holder_of": enc.holder_of,
                "location_of": enc.location_of,
                "dropped_at": enc.dropped_at,
            },
            type_groups={"person": enc._people, "place": enc._places},
            max_iter=5,
        )
        for obj, expected in story.queries:
            pred, _ = block.solve(obj, target_type="place")
            correct += int(pred == expected)
            total += 1
    return correct / max(total, 1), total


def eval_rung4(stories, codebook, dim: int) -> dict:
    """Replicates Milestone H.1 but with a swappable codebook."""
    enc = make_t3_reasoner(dim, codebook)
    self_memory = HDCMemory(dim, decay=1.0)

    # Phase 1: normal reasoning
    results = []
    for story_idx, story in enumerate(stories):
        enc.reset()
        for event in story.events:
            enc.observe(event)
        block = ReasoningBlock(
            concepts=enc.concepts,
            memories={
                "holder_of": enc.holder_of,
                "location_of": enc.location_of,
                "dropped_at": enc.dropped_at,
            },
            type_groups={"person": enc._people, "place": enc._places},
            max_iter=5,
        )
        for obj, expected in story.queries:
            pred, _ = block.solve(obj, target_type="place")
            enc.concepts.add(obj)
            enc.concepts.add(expected)
            results.append({
                "story_idx": story_idx,
                "obj": obj,
                "predicted": pred,
                "true": expected,
                "correct": pred == expected,
            })

    correct_p1 = sum(r["correct"] for r in results)
    acc_p1 = correct_p1 / len(results)
    errors = [r for r in results if not r["correct"]]

    if not errors:
        return {
            "phase1_acc": acc_p1,
            "errors": 0,
            "recovered": 0,
            "recovery_rate": 1.0,
            "avg_stored_conf": 0.0,
            "avg_unstored_conf": 0.0,
            "integrated_acc": acc_p1,
        }

    # Phase 2: write corrections using story-id keys
    def story_vec(idx: int, d: int) -> np.ndarray:
        rng = np.random.default_rng(12345 + idx)
        return rng.choice([-1.0, 1.0], size=d).astype(np.float32)

    all_places = set(r["true"] for r in results) | enc._places
    for r in errors:
        svec = story_vec(r["story_idx"], dim)
        obj_vec = enc.concepts.get(r["obj"])
        true_vec = enc.concepts.get(r["true"])
        key = bind(svec, obj_vec)
        self_memory.write(bind(key, true_vec))

    # Phase 3: retrieval on the failures
    recovered = 0
    stored_confs = []
    for r in errors:
        svec = story_vec(r["story_idx"], dim)
        obj_vec = enc.concepts.get(r["obj"])
        key = bind(svec, obj_vec)
        retrieved = bind(self_memory.read(), key)
        name, conf = enc.concepts.cleanup(retrieved, restrict_to=all_places)
        stored_confs.append(conf)
        if name == r["true"]:
            recovered += 1

    # Phase 4: sanity on unstored keys
    unstored_confs = []
    for r in results[:200]:
        if not r["correct"]:
            continue
        svec = story_vec(r["story_idx"], dim)
        obj_vec = enc.concepts.get(r["obj"])
        key = bind(svec, obj_vec)
        retrieved = bind(self_memory.read(), key)
        _, conf = enc.concepts.cleanup(retrieved, restrict_to=all_places)
        unstored_confs.append(conf)

    # Phase 5: integrated accuracy with halfway threshold
    stored_mean = float(np.mean(stored_confs)) if stored_confs else 0.0
    unstored_mean = float(np.mean(unstored_confs)) if unstored_confs else 0.0
    threshold = (stored_mean + unstored_mean) / 2  # midpoint

    correct_integ = 0
    total_integ = 0
    for r in results:
        total_integ += 1
        svec = story_vec(r["story_idx"], dim)
        obj_vec = enc.concepts.get(r["obj"])
        key = bind(svec, obj_vec)
        retrieved = bind(self_memory.read(), key)
        name, conf = enc.concepts.cleanup(retrieved, restrict_to=all_places)
        if conf > threshold:
            final = name
        else:
            final = r["predicted"]
        if final == r["true"]:
            correct_integ += 1

    return {
        "phase1_acc": acc_p1,
        "errors": len(errors),
        "recovered": recovered,
        "recovery_rate": recovered / len(errors),
        "avg_stored_conf": stored_mean,
        "avg_unstored_conf": unstored_mean,
        "threshold_used": threshold,
        "integrated_acc": correct_integ / total_integ,
    }


def main():
    print("=" * 72)
    print("Milestone F.1 — end-to-end Cogito on v14 learned codebook")
    print("=" * 72)

    print()
    print("Loading v14 pretrained codebook...")
    v14_pool = load_v14_pool()
    print(f"  shape: {v14_pool.shape}")

    DIM = 4096

    rng2 = random.Random(42)
    stories_t2 = generate_task2(500, rng2)
    rng3 = random.Random(42)
    stories_t3 = generate_task3(500, rng3)

    # --------- Task 2 ------------
    print()
    print("--- Task 2 (chained reasoning) ---")
    t0 = time.perf_counter()
    acc_rand, n = eval_task2(stories_t2, Codebook(DIM, seed=42), DIM)
    dt_rand = time.perf_counter() - t0
    print(f"  random bipolar:  {acc_rand:.4f}  ({n} q, {dt_rand:.1f}s)")

    t0 = time.perf_counter()
    acc_v14, _ = eval_task2(stories_t2, PretrainedCodebook(v14_pool, seed=42), DIM)
    dt_v14 = time.perf_counter() - t0
    print(f"  v14 pretrained:  {acc_v14:.4f}  ({n} q, {dt_v14:.1f}s)")
    print(f"  delta:           {acc_v14 - acc_rand:+.4f}")

    # --------- Task 3 ------------
    print()
    print("--- Task 3 (branching reasoning with drops) ---")
    t0 = time.perf_counter()
    acc3_rand, n3 = eval_task3(stories_t3, Codebook(DIM, seed=42), DIM)
    dt3_rand = time.perf_counter() - t0
    print(f"  random bipolar:  {acc3_rand:.4f}  ({n3} q, {dt3_rand:.1f}s)")

    t0 = time.perf_counter()
    acc3_v14, _ = eval_task3(stories_t3, PretrainedCodebook(v14_pool, seed=42), DIM)
    dt3_v14 = time.perf_counter() - t0
    print(f"  v14 pretrained:  {acc3_v14:.4f}  ({n3} q, {dt3_v14:.1f}s)")
    print(f"  delta:           {acc3_v14 - acc3_rand:+.4f}")

    # --------- Rung 4 self-memory ------------
    print()
    print("--- Rung 4 self-memory (error correction) ---")
    r_rand = eval_rung4(stories_t3, Codebook(DIM, seed=42), DIM)
    print(f"  random bipolar:")
    print(f"    phase1 acc      : {r_rand['phase1_acc']:.4f}")
    print(f"    errors          : {r_rand['errors']}")
    print(f"    recovered       : {r_rand['recovered']}/{r_rand['errors']}"
          f" ({r_rand['recovery_rate']*100:.2f}%)")
    print(f"    stored conf     : {r_rand['avg_stored_conf']:.4f}")
    print(f"    unstored conf   : {r_rand['avg_unstored_conf']:.4f}")
    print(f"    integrated acc  : {r_rand['integrated_acc']:.4f}")

    r_v14 = eval_rung4(stories_t3, PretrainedCodebook(v14_pool, seed=42), DIM)
    print(f"  v14 pretrained:")
    print(f"    phase1 acc      : {r_v14['phase1_acc']:.4f}")
    print(f"    errors          : {r_v14['errors']}")
    print(f"    recovered       : {r_v14['recovered']}/{r_v14['errors']}"
          f" ({r_v14['recovery_rate']*100:.2f}%)")
    print(f"    stored conf     : {r_v14['avg_stored_conf']:.4f}")
    print(f"    unstored conf   : {r_v14['avg_unstored_conf']:.4f}")
    print(f"    integrated acc  : {r_v14['integrated_acc']:.4f}")

    # --------- Summary ------------
    print()
    print("=" * 72)
    print("Summary — cost of swapping random bipolar -> v14 pretrained")
    print("=" * 72)
    print(f"  Task 2 accuracy:      {acc_rand:.4f} -> {acc_v14:.4f}"
          f"  (delta {acc_v14-acc_rand:+.4f})")
    print(f"  Task 3 accuracy:      {acc3_rand:.4f} -> {acc3_v14:.4f}"
          f"  (delta {acc3_v14-acc3_rand:+.4f})")
    print(f"  Rung 4 integrated:    {r_rand['integrated_acc']:.4f}"
          f" -> {r_v14['integrated_acc']:.4f}"
          f"  (delta {r_v14['integrated_acc']-r_rand['integrated_acc']:+.4f})")


if __name__ == "__main__":
    main()
