"""
Milestone C.1 — depth stress test for iterative reasoning.

The question this milestone answers:

    Does iteration count genuinely scale with problem depth, or does
    it saturate at 2 hops (as in bAbI tasks 2 and 3)?

If iteration saturates at 2 even for deeper problems, the proposal's
"variable depth up to ~30 iterations" story is over-engineered and
Milestone D must shrink. If iteration scales cleanly with depth, the
story holds and we can start designing the learned reasoning block
with confidence that depth will matter.

Task construction
-----------------

Nested containers:

    put(apple, box)
    put(box, bag)
    put(bag, kitchen)
    Where is the apple?   -> kitchen

The chain depth is controlled by construction. Depth D means:
D unbind operations are required to follow the chain from the leaf
item to the root place. No shortcut exists — the `inside_of` memory
stores only adjacent-level bindings.

For each depth D in {1, 2, 3, 4, 5}, we run the exact same
`ReasoningBlock` from Milestone C with `max_iter` set to various
values from 1 to 7. We expect a staircase:

    max_iter < depth  -> accuracy collapses
    max_iter >= depth -> accuracy saturates near ceiling

If the staircase is clean, iteration count is genuinely the knob
that unlocks deeper reasoning, and "compute scales with difficulty"
is empirically supported.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

from hdc_lang import Codebook, HDCMemory, bind
from run_milestone_C import ReasoningBlock


# ----- Data ---------------------------------------------------------------

@dataclass
class Chain:
    # elements[0] = leaf item, elements[-1] = root place, others = containers
    elements: list[str] = field(default_factory=list)


@dataclass
class Story:
    chains: list[Chain] = field(default_factory=list)
    queries: list[tuple[str, str]] = field(default_factory=list)  # (leaf, root_place)


def generate_depth_stories(
    num_stories: int,
    depth: int,
    rng: random.Random,
    chains_per_story: int = 3,
    items_pool_size: int = 80,
    containers_pool_size: int = 300,
    places_pool_size: int = 15,
) -> list[Story]:
    """
    Generate stories where every chain has exactly `depth` hops
    (i.e. `depth + 1` elements). Multiple chains per story provide
    distractor noise in the shared HDC memory.

    Each chain uses fresh container names to avoid accidental
    interference between chains.
    """
    items_pool = [f"item_{i}" for i in range(items_pool_size)]
    containers_pool = [f"cont_{i}" for i in range(containers_pool_size)]
    places_pool = [f"place_{i}" for i in range(places_pool_size)]

    stories: list[Story] = []
    for _ in range(num_stories):
        story_items = rng.sample(items_pool, chains_per_story)
        # Containers needed: each chain has (depth - 1) intermediate containers.
        # Depth 1 means no intermediate containers (item → place directly).
        intermediate_count = max(depth - 1, 0)
        containers_needed = chains_per_story * intermediate_count
        if containers_needed > 0:
            story_containers = rng.sample(containers_pool, containers_needed)
        else:
            story_containers = []

        chains: list[Chain] = []
        idx = 0
        for item in story_items:
            elements = [item]
            for _ in range(intermediate_count):
                elements.append(story_containers[idx])
                idx += 1
            elements.append(rng.choice(places_pool))
            chains.append(Chain(elements=elements))

        # Query about a uniformly random chain so that write order
        # does not bias query difficulty.
        target = rng.choice(chains)
        queries = [(target.elements[0], target.elements[-1])]

        stories.append(Story(chains=chains, queries=queries))
    return stories


# ----- Encoder ------------------------------------------------------------

class DepthEncoder:
    """
    Encodes nested-container facts into a single HDC memory.

    For each adjacent pair (inner, outer) in a chain, writes
    `bind(inner, outer)` into `inside_of`. No decay by default —
    all chain edges get equal weight, so the difficulty of a query
    does not depend on write order.
    """

    def __init__(self, dim: int, decay: float, seed: int):
        self.dim = dim
        self.concepts = Codebook(dim, seed=seed)
        self.inside_of = HDCMemory(dim, decay=decay)
        self._items: set[str] = set()
        self._containers: set[str] = set()
        self._places: set[str] = set()

    def reset(self) -> None:
        self.inside_of.reset()

    def observe_chain(self, chain: Chain) -> None:
        elts = chain.elements
        n = len(elts)
        for i in range(n - 1):
            inner = elts[i]
            outer = elts[i + 1]
            if i == 0:
                self._items.add(inner)
            else:
                self._containers.add(inner)
            if i == n - 2:
                self._places.add(outer)
            else:
                self._containers.add(outer)
            self.inside_of.write(
                bind(self.concepts.add(inner), self.concepts.add(outer))
            )


# ----- Evaluation ---------------------------------------------------------

def evaluate_depth(
    stories: list[Story],
    dim: int,
    max_iter: int,
    decay: float,
    conf_floor: float,
) -> dict:
    correct = 0
    total = 0
    iters_hist: dict[int, int] = {}

    for story in stories:
        enc = DepthEncoder(dim=dim, decay=decay, seed=42)
        for chain in story.chains:
            enc.observe_chain(chain)

        # IMPORTANT: type_groups does NOT include "item". Items are only
        # start-states; including them as valid retrieval targets would
        # let the block step *backward* (e.g. from box back to apple),
        # defeating the point of depth measurement.
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
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--stories", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decay", type=float, default=1.0,
                   help="decay for inside_of memory (1.0 = pure sum, unbiased)")
    p.add_argument("--conf-floor", type=float, default=0.03)
    p.add_argument("--chains-per-story", type=int, default=3)
    args = p.parse_args()

    print("Milestone C.1: depth stress test for iterative reasoning")
    print(f"D = {args.dim}, decay = {args.decay}, conf_floor = {args.conf_floor}")
    print(f"Chains per story: {args.chains_per_story}")
    print(f"Stories per depth: {args.stories}")
    print()
    print("Task: nested containers. A chain of depth D needs D unbinds to reach a place.")
    print("Expected: staircase — accuracy collapses when max_iter < depth, saturates when max_iter >= depth.")
    print()

    depths = [1, 2, 3, 4, 5]
    max_iters = [1, 2, 3, 4, 5, 7]

    header = f"{'depth':>6}  " + "  ".join(f"iter={mi:>2}" for mi in max_iters)
    print(header)
    print("-" * len(header))

    results = {}
    for depth in depths:
        rng = random.Random(args.seed + depth)
        stories = generate_depth_stories(
            num_stories=args.stories,
            depth=depth,
            rng=rng,
            chains_per_story=args.chains_per_story,
        )
        row = f"{depth:>6}  "
        results[depth] = {}
        for max_iter in max_iters:
            r = evaluate_depth(
                stories,
                dim=args.dim,
                max_iter=max_iter,
                decay=args.decay,
                conf_floor=args.conf_floor,
            )
            results[depth][max_iter] = r
            row += f"  {r['accuracy']:>6.3f}"
        print(row)

    print()
    print("Iteration-use histograms at max_iter=7 (ample budget):")
    for depth in depths:
        r = results[depth][7]
        hist_str = ", ".join(
            f"{k} iter: {v}" for k, v in sorted(r["iters_hist"].items())
        )
        print(f"  depth={depth}: acc={r['accuracy']:.3f}  |  {hist_str}")

    print()
    print("Interpretation check:")
    clean_staircase = True
    for depth in depths:
        # When max_iter < depth, accuracy should be near 0
        for mi in max_iters:
            if mi >= depth:
                continue
            if results[depth][mi]["accuracy"] > 0.2:
                clean_staircase = False
                print(f"  WARN: depth={depth}, max_iter={mi}, acc={results[depth][mi]['accuracy']:.3f} (expected near 0)")
        # When max_iter >= depth, accuracy should be high
        for mi in max_iters:
            if mi < depth:
                continue
            if results[depth][mi]["accuracy"] < 0.8:
                clean_staircase = False
                print(f"  WARN: depth={depth}, max_iter={mi}, acc={results[depth][mi]['accuracy']:.3f} (expected > 0.8)")
    if clean_staircase:
        print("  Clean staircase: max_iter < depth collapses, max_iter >= depth saturates.")
        print("  H2 (iteration scales with difficulty) is empirically supported at depths 1-5.")


if __name__ == "__main__":
    main()
