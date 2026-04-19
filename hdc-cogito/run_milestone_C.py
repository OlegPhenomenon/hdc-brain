"""
Milestone C — first iterative reasoning block (no learned weights).

Goal: validate hypothesis H2 from the proposal (§3):

    "Variable-depth reasoning ... through a single reasoning block
     ... can solve tasks that would normally require stacking more
     layers. Iteration count scales with task difficulty, not with
     architecture size."

Approach: a task-agnostic HDC reasoning block that, at each step,
tries unbinding the current state against EVERY available memory
and picks the retrieval that gives the sharpest cleanup. It does
not know which memory is which. It just iterates, refining the
state, until the retrieval is of the target concept type.

If this block — using the SAME hdc_lang primitives, no learned
weights — matches the accuracy of the hand-coded Milestone B
reasoners on tasks 2 and 3, then we have direct evidence that:

  1. Chained reasoning is latent in the primitives (Milestone B
     already showed this), AND
  2. A general iteration rule can DISCOVER the right chain at
     query time without knowing the task.

The H2 test is a comparison between:

  - max_iter = 1  (single-pass — a degenerate "block" that only
                   does one Unbind)
  - max_iter = 5  (iterative — up to five refinement steps)

Task 2 every question needs at least 2 steps (chain). So max_iter=1
should collapse to near-zero accuracy, and iteration should lift it
to near-ceiling. That is H2 in action.
"""

from __future__ import annotations

import argparse
import random
import time

from hdc_lang import Codebook, HDCMemory
from run_babi_task2 import Task2Reasoner, generate_task2
from run_babi_task3 import Task3Reasoner, generate_task3


# ----- The reasoning block ------------------------------------------------

class ReasoningBlock:
    """
    Task-agnostic iterative VSA reasoning.

    State at step i is a single HDC vector h_i.  At each step:

      1. For every (memory, type-group) pair, compute
             name, conf = cleanup( unbind(memory, h_i), type_group )
      2. Score each candidate:
             scored = conf + target_bonus  (if type_group is target type)
                      conf                (otherwise)
      3. Pick the candidate with the highest scored confidence.
      4. If the picked candidate is of the target type, return it.
      5. Otherwise update h_{i+1} to the picked concept's vector and loop.

    Termination:
      - Target type reached
      - Confidence falls below conf_floor
      - Cycle detected (same concept revisited)
      - max_iter reached

    Nothing here knows "holder_of stores people" or "location_of stores places".
    The block discovers the chain by picking the sharpest retrieval.
    """

    def __init__(
        self,
        concepts: Codebook,
        memories: dict[str, HDCMemory],
        type_groups: dict[str, set[str]],
        target_type_bonus: float = 0.0,
        conf_floor: float = 0.08,
        max_iter: int = 5,
    ):
        # Note on target_type_bonus:
        # An early version of this block added a bonus to the scored
        # confidence whenever a retrieval matched the target type, to
        # encourage "stop as soon as you land on the answer". Empirically
        # that hurts: a non-zero bonus lets *noisy* place retrievals
        # from the wrong memory beat *legitimate* intermediate person
        # retrievals, causing premature commits on random places.
        # Grid search on tasks 2 and 3 shows bonus = 0.0 is optimal.
        # Kept as a parameter for future ablations.
        self.concepts = concepts
        self.memories = memories
        self.type_groups = type_groups
        self.target_type_bonus = target_type_bonus
        self.conf_floor = conf_floor
        self.max_iter = max_iter

    def _best_step(
        self,
        h,
        target_type: str,
        visited: set[str],
    ) -> tuple[str, str, float, float, str] | None:
        """
        Try every memory x type combination at the current state.
        Return (name, type_name, raw_conf, scored_conf, memory_name) for the best.

        `visited` is the set of concept names already on the current
        reasoning path. They are excluded from cleanup candidates so
        that the block cannot step backward along a symmetric binding
        (the HDC `bind` operation is its own inverse, so bind(a, b)
        looks identical to bind(b, a) when unbinding naively).
        """
        best = None
        for mem_name, mem in self.memories.items():
            noisy = mem.query(h)
            for type_name, names in self.type_groups.items():
                candidates = [n for n in names if n not in visited]
                if not candidates:
                    continue
                name, conf = self.concepts.cleanup(noisy, restrict_to=candidates)
                bonus = self.target_type_bonus if type_name == target_type else 0.0
                scored = conf + bonus
                if best is None or scored > best[3]:
                    best = (name, type_name, conf, scored, mem_name)
        return best

    def solve(self, query_name: str, target_type: str) -> tuple[str, int]:
        """
        Starting from the concept `query_name`, iterate until a concept of
        `target_type` is retrieved (or the termination conditions fire).

        Returns (answer_name, iterations_used).
        """
        if query_name not in self.concepts:
            return "<unknown>", 0
        h = self.concepts.get(query_name)
        visited = {query_name}
        last_name = query_name
        for i in range(self.max_iter):
            step = self._best_step(h, target_type, visited)
            iterations = i + 1
            if step is None:
                return last_name, iterations
            name, type_name, raw_conf, _, _ = step
            last_name = name
            if raw_conf < self.conf_floor:
                return name, iterations
            if type_name == target_type:
                return name, iterations
            if name not in self.concepts:
                return name, iterations
            visited.add(name)
            h = self.concepts.get(name)
        return last_name, self.max_iter


# ----- Adapters: use Task2Reasoner / Task3Reasoner as encoders ------------

def task2_block_eval(stories, dim: int, max_iter: int) -> dict:
    """Encode task 2 stories via Task2Reasoner, query via ReasoningBlock."""
    correct = 0
    total = 0
    iters_hist: dict[int, int] = {}
    for story in stories:
        enc = Task2Reasoner(
            dim=dim, decay_holder=0.7, decay_location=0.8, seed=42
        )
        for event in story.events:
            enc.observe(event)
        block = ReasoningBlock(
            concepts=enc.concepts,
            memories={
                "holder_of": enc.holder_of,
                "location_of": enc.location_of,
            },
            type_groups={"person": enc._people, "place": enc._places},
            max_iter=max_iter,
        )
        for obj, expected in story.queries:
            pred, iters = block.solve(obj, target_type="place")
            correct += int(pred == expected)
            total += 1
            iters_hist[iters] = iters_hist.get(iters, 0) + 1
    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "iters_hist": iters_hist,
    }


def task3_block_eval(stories, dim: int, max_iter: int) -> dict:
    """Encode task 3 stories via Task3Reasoner, query via ReasoningBlock."""
    correct = 0
    total = 0
    iters_hist: dict[int, int] = {}
    for story in stories:
        enc = Task3Reasoner(
            dim=dim,
            decay_holder=0.7,
            decay_location=0.8,
            decay_drop=0.8,
            seed=42,
        )
        for event in story.events:
            enc.observe(event)
        # Deliberate: we do NOT give the block the DROPPED marker in type_groups.
        # The block sees only 'person' and 'place' types. The DROPPED
        # write into holder_of then just acts as noise that weakens Mary's
        # signal in holder_of when the object has been dropped — and
        # meanwhile dropped_at gives a clean place binding. The block
        # picks dropped_at because its place retrieval is sharper.
        block = ReasoningBlock(
            concepts=enc.concepts,
            memories={
                "holder_of": enc.holder_of,
                "location_of": enc.location_of,
                "dropped_at": enc.dropped_at,
            },
            type_groups={"person": enc._people, "place": enc._places},
            max_iter=max_iter,
        )
        for obj, expected in story.queries:
            pred, iters = block.solve(obj, target_type="place")
            correct += int(pred == expected)
            total += 1
            iters_hist[iters] = iters_hist.get(iters, 0) + 1
    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "iters_hist": iters_hist,
    }


def fmt_hist(hist: dict[int, int], total: int) -> str:
    parts = [f"{k} iter: {v} ({v/total*100:.0f}%)" for k, v in sorted(hist.items())]
    return "  |  ".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--stories", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng2 = random.Random(args.seed)
    stories2 = generate_task2(args.stories, rng2)
    rng3 = random.Random(args.seed)
    stories3 = generate_task3(args.stories, rng3)
    nq2 = sum(len(s.queries) for s in stories2)
    nq3 = sum(len(s.queries) for s in stories3)

    print("Milestone C: task-agnostic iterative reasoning block")
    print(f"Task 2: {len(stories2)} stories, {nq2} questions (chained, no drops)")
    print(f"Task 3: {len(stories3)} stories, {nq3} questions (with drops)")
    print(f"D = {args.dim}")
    print()
    print("Hypothesis H2: iteration replaces depth.")
    print("Test: compare max_iter=1 (single-pass) against max_iter=2,3,5 (iterative).")
    print("If iteration matters, max_iter=1 should collapse on task 2 (no direct path).")
    print()

    print(f"{'task':>8}  {'max_iter':>10}  {'accuracy':>10}  iteration distribution")
    print("-" * 78)

    for max_iter in [1, 2, 3, 5]:
        t0 = time.perf_counter()
        r2 = task2_block_eval(stories2, dim=args.dim, max_iter=max_iter)
        t2 = time.perf_counter() - t0
        print(
            f"{'task 2':>8}  {max_iter:>10}  {r2['accuracy']:>10.4f}  "
            f"{fmt_hist(r2['iters_hist'], r2['total'])}   [{t2*1000:.0f} ms]"
        )

    print()
    for max_iter in [1, 2, 3, 5]:
        t0 = time.perf_counter()
        r3 = task3_block_eval(stories3, dim=args.dim, max_iter=max_iter)
        t3 = time.perf_counter() - t0
        print(
            f"{'task 3':>8}  {max_iter:>10}  {r3['accuracy']:>10.4f}  "
            f"{fmt_hist(r3['iters_hist'], r3['total'])}   [{t3*1000:.0f} ms]"
        )

    print()
    print("Comparison against Milestone B reference (hand-coded task-specific):")
    print("  task 2 reference: ~99.1%  at D=1024 (Milestone B.1)")
    print("  task 3 reference: ~99.3%  at D=1024 (Milestone B.2)")
    print()
    print("If the block at max_iter=5 matches these numbers, we have shown")
    print("iteration can discover the chain that Milestone B hand-coded.")


if __name__ == "__main__":
    main()
