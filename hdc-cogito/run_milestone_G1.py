"""
Milestone G.1 — First learned reasoning block, minimal version.

Adds 6 learnable scalars to the Milestone C block. Each scalar is the
trust weight for a specific (memory, concept-type) combination. The
block otherwise behaves exactly like Milestone C: same iteration loop,
same visited-exclusion, same type-match halting, same four VSA
primitives.

The point of this milestone is to test whether the 1–2% gap between
the Milestone C task-agnostic block and the hand-coded Milestone B
reasoners is explained by missing "which memory to trust for which
answer type" knowledge. If six scalars close the gap, the answer is
yes and we have a very small, very interpretable learned block. If
not, the gap is about something deeper and the design needs to
revisit.

Training method: random search. Nine scalars (wait, six — we only
have 2 types and up to 3 memories) is tiny, and the forward pass is
non-trivial to differentiate, so we just sample 500 random weight
assignments from a reasonable range and keep the one with highest
validation accuracy. No PyTorch, no autograd. Standard
train / validation / test splits to avoid overfitting.
"""

from __future__ import annotations

import random
import time

from hdc_lang import Codebook, HDCMemory
from run_milestone_C import ReasoningBlock
from run_babi_task2 import Task2Reasoner, generate_task2
from run_babi_task3 import Task3Reasoner, generate_task3


# ----- The learned block --------------------------------------------------

class LearnedReasoningBlock(ReasoningBlock):
    """
    Milestone C block, plus a per-(memory, type) weight on each retrieval.

    Every structural property of the Milestone C block is preserved:
      - Visited-exclusion in cleanup (Milestone C.1 correctness requirement)
      - Type-match halting (target type reached -> stop)
      - Iterative refinement until target type, cycle, or max_iter
      - No learned neural network inside — only scalar biases on scores

    `weights[(memory_name, type_name)]` multiplies the raw cleanup cosine
    for retrievals of that combination. A weight of 1.0 reproduces
    Milestone C behavior exactly.
    """

    def __init__(self, *, weights: dict[tuple[str, str], float], **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def _best_step(self, h, target_type, visited):
        best = None
        for mem_name, mem in self.memories.items():
            noisy = mem.query(h)
            for type_name, names in self.type_groups.items():
                candidates = [n for n in names if n not in visited]
                if not candidates:
                    continue
                name, conf = self.concepts.cleanup(noisy, restrict_to=candidates)
                weight = self.weights.get((mem_name, type_name), 1.0)
                scored = conf * weight
                if best is None or scored > best[3]:
                    best = (name, type_name, conf, scored, mem_name)
        return best


# ----- Precomputation of encoded stories ---------------------------------

def precompute_task2(stories, dim: int = 1024):
    """Encode each task-2 story once into a reasoner; reuse across eval runs."""
    encoded = []
    for story in stories:
        enc = Task2Reasoner(dim=dim, decay_holder=0.7, decay_location=0.8, seed=42)
        for event in story.events:
            enc.observe(event)
        encoded.append((enc, story.queries))
    return encoded


def precompute_task3(stories, dim: int = 1024):
    encoded = []
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
        encoded.append((enc, story.queries))
    return encoded


# ----- Evaluation -------------------------------------------------------

def eval_task2(encoded, weights, max_iter: int = 5) -> float:
    correct = 0
    total = 0
    for enc, queries in encoded:
        block = LearnedReasoningBlock(
            concepts=enc.concepts,
            memories={
                "holder_of": enc.holder_of,
                "location_of": enc.location_of,
            },
            type_groups={"person": enc._people, "place": enc._places},
            weights=weights,
            max_iter=max_iter,
        )
        for obj, expected in queries:
            pred, _ = block.solve(obj, target_type="place")
            correct += int(pred == expected)
            total += 1
    return correct / max(total, 1)


def eval_task3(encoded, weights, max_iter: int = 5) -> float:
    correct = 0
    total = 0
    for enc, queries in encoded:
        block = LearnedReasoningBlock(
            concepts=enc.concepts,
            memories={
                "holder_of": enc.holder_of,
                "location_of": enc.location_of,
                "dropped_at": enc.dropped_at,
            },
            type_groups={"person": enc._people, "place": enc._places},
            weights=weights,
            max_iter=max_iter,
        )
        for obj, expected in queries:
            pred, _ = block.solve(obj, target_type="place")
            correct += int(pred == expected)
            total += 1
    return correct / max(total, 1)


# ----- Random search over weights ---------------------------------------

WEIGHT_KEYS = [
    ("holder_of", "person"),
    ("holder_of", "place"),
    ("location_of", "person"),
    ("location_of", "place"),
    ("dropped_at", "person"),
    ("dropped_at", "place"),
]

BASELINE_WEIGHTS = {k: 1.0 for k in WEIGHT_KEYS}


def sample_random_weights(rng: random.Random, low: float = 0.3, high: float = 2.5) -> dict:
    return {k: rng.uniform(low, high) for k in WEIGHT_KEYS}


def format_weights(weights: dict) -> str:
    lines = []
    for k in WEIGHT_KEYS:
        lines.append(f"    {k[0]:>12s} x {k[1]:>6s} : {weights[k]:.3f}")
    return "\n".join(lines)


def main():
    DIM = 1024

    print("=" * 72)
    print("Milestone G.1 — learned reasoning block (6 scalars, random search)")
    print("=" * 72)

    # ---- Data splits --------------------------------------------------
    print()
    print("Generating train / validation / test splits...")
    train_t2 = generate_task2(500, random.Random(42))
    train_t3 = generate_task3(500, random.Random(42))
    val_t2 = generate_task2(200, random.Random(100))
    val_t3 = generate_task3(200, random.Random(100))
    test_t2 = generate_task2(300, random.Random(200))
    test_t3 = generate_task3(300, random.Random(200))

    print(f"  train : task2 {sum(len(s.queries) for s in train_t2)} q,"
          f" task3 {sum(len(s.queries) for s in train_t3)} q")
    print(f"  val   : task2 {sum(len(s.queries) for s in val_t2)} q,"
          f" task3 {sum(len(s.queries) for s in val_t3)} q")
    print(f"  test  : task2 {sum(len(s.queries) for s in test_t2)} q,"
          f" task3 {sum(len(s.queries) for s in test_t3)} q")

    # ---- Precompute encoded memories (cheap trick: do it once) -------
    print()
    print("Encoding stories once (so random search only runs the query step)...")
    t0 = time.perf_counter()
    val_enc_t2 = precompute_task2(val_t2, dim=DIM)
    val_enc_t3 = precompute_task3(val_t3, dim=DIM)
    test_enc_t2 = precompute_task2(test_t2, dim=DIM)
    test_enc_t3 = precompute_task3(test_t3, dim=DIM)
    print(f"  encoding took {time.perf_counter() - t0:.1f}s")

    # ---- Baseline --------------------------------------------------
    print()
    print("Baseline (all weights = 1.0, equivalent to Milestone C block):")
    base_val_t2 = eval_task2(val_enc_t2, BASELINE_WEIGHTS)
    base_val_t3 = eval_task3(val_enc_t3, BASELINE_WEIGHTS)
    print(f"  validation task 2: {base_val_t2:.4f}")
    print(f"  validation task 3: {base_val_t3:.4f}")
    print(f"  combined         : {(base_val_t2 + base_val_t3) / 2:.4f}")

    # ---- Random search --------------------------------------------
    print()
    print("Running random search (500 samples over 6 weights)...")
    NUM_SAMPLES = 500
    best_score = (base_val_t2 + base_val_t3) / 2
    best_weights = BASELINE_WEIGHTS.copy()
    best_t2 = base_val_t2
    best_t3 = base_val_t3

    rng = random.Random(7)
    t0 = time.perf_counter()
    for i in range(NUM_SAMPLES):
        w = sample_random_weights(rng)
        acc_t2 = eval_task2(val_enc_t2, w)
        acc_t3 = eval_task3(val_enc_t3, w)
        score = (acc_t2 + acc_t3) / 2
        if score > best_score:
            best_score = score
            best_weights = w
            best_t2 = acc_t2
            best_t3 = acc_t3
            print(f"  sample {i:>3d}: new best combined={score:.4f}"
                  f"  (t2={acc_t2:.4f}, t3={acc_t3:.4f})")
    dt = time.perf_counter() - t0
    print(f"  search done in {dt:.1f}s")

    # ---- Test evaluation ------------------------------------------
    print()
    print("Test evaluation (held-out stories, never seen during search):")
    print()
    print("Best weights found:")
    print(format_weights(best_weights))
    print()

    base_test_t2 = eval_task2(test_enc_t2, BASELINE_WEIGHTS)
    base_test_t3 = eval_task3(test_enc_t3, BASELINE_WEIGHTS)
    learned_test_t2 = eval_task2(test_enc_t2, best_weights)
    learned_test_t3 = eval_task3(test_enc_t3, best_weights)

    print(f"{'system':<24}{'task 2':>10}{'task 3':>10}")
    print("-" * 44)
    print(f"{'Milestone C baseline':<24}{base_test_t2:>10.4f}{base_test_t3:>10.4f}")
    print(f"{'Learned (6 scalars)':<24}{learned_test_t2:>10.4f}{learned_test_t3:>10.4f}")
    print(f"{'Hand-coded reference':<24}{0.9906:>10.4f}{0.9933:>10.4f}")
    print()

    delta_t2 = learned_test_t2 - base_test_t2
    delta_t3 = learned_test_t3 - base_test_t3
    print(f"Delta vs baseline:   task 2 {delta_t2:+.4f}   task 3 {delta_t3:+.4f}")

    # ---- Interpretation --------------------------------------------
    print()
    print("Interpretation of learned weights (highest = memory most trusted for that type):")
    print()
    for type_name in ["person", "place"]:
        print(f"  for target type = {type_name}:")
        ranked = sorted(
            [(m, t, best_weights[(m, t)]) for (m, t) in WEIGHT_KEYS if t == type_name],
            key=lambda x: -x[2],
        )
        for m, t, w in ranked:
            print(f"    {m:>14s} : {w:.3f}")


if __name__ == "__main__":
    main()
