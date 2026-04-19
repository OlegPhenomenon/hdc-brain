"""
Phase -1: pure VSA baseline on bAbI task 1 (Single Supporting Fact).

The experiment:
  1. Generate synthetic bAbI-task-1-style stories ("X moved to Y" sentences)
  2. For each story, encode each sentence as Bind(person, location)
  3. Bundle into a time-decayed HDC memory
  4. For each query "Where is X?", compute Unbind(memory, X) and clean up
     against the location codebook

Success criterion: > 95% accuracy with zero training and zero learned weights.

This is the go/no-go gate for HDC-Cogito. If this fails, there is a bug
in the VSA primitives, not a conceptual problem — Plate 1995 proved this
approach works.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

from hdc_lang import Codebook, HDCMemory, bind


# ----- Data ----------------------------------------------------------------

@dataclass
class Story:
    # Each sentence is (person, location)
    sentences: list[tuple[str, str]] = field(default_factory=list)
    # Each query: (person, expected_location, index_of_last_relevant_sentence)
    queries: list[tuple[str, str, int]] = field(default_factory=list)


def generate_synthetic_task1(num_stories: int, rng: random.Random) -> list[Story]:
    """
    Generate bAbI task 1 style stories.

    Each story has 2-6 move events. Queries ask about the current (most recent)
    location of each person who appears in the story.
    """
    people = ["Mary", "John", "Sandra", "Daniel", "Bob", "Alice", "Carol", "Frank"]
    places = ["bathroom", "kitchen", "hallway", "garden", "bedroom", "office", "cellar", "attic"]

    stories: list[Story] = []
    for _ in range(num_stories):
        k = rng.randint(2, 6)
        sentences: list[tuple[str, str]] = []
        for _ in range(k):
            person = rng.choice(people)
            place = rng.choice(places)
            sentences.append((person, place))

        # Build queries: for each person that appeared, ask where they are now
        queries: list[tuple[str, str, int]] = []
        seen: dict[str, tuple[str, int]] = {}
        for i, (p, l) in enumerate(sentences):
            seen[p] = (l, i)
        for person, (expected, last_idx) in seen.items():
            queries.append((person, expected, last_idx))

        stories.append(Story(sentences=sentences, queries=queries))
    return stories


# ----- VSA Reasoner --------------------------------------------------------

class VSAReasoner:
    """
    Phase -1 reasoner: pure Plate-style HRR, no training.

    Encoding: each "X is at Y" sentence becomes Bind(X_vec, Y_vec),
    written to a decaying memory. Query "Where is X?" is answered by
    Unbind(memory, X_vec) followed by nearest-neighbor cleanup against
    the location codebook.

    Note for Phase -1: we're using person vectors as de-facto "role" vectors.
    In later phases we will separate role vocabulary (agent, location, ...)
    from content vocabulary. For a single-role task like bAbI-1, this
    collapse is legitimate and lets us validate the primitives in isolation.
    """

    def __init__(self, dim: int, decay: float, seed: int):
        self.dim = dim
        self.decay = decay
        self.people = Codebook(dim, seed=seed)
        self.places = Codebook(dim, seed=seed + 10_000)
        self.memory = HDCMemory(dim, decay=decay)

    def encode_up_to(self, story: Story, up_to_index: int) -> None:
        """Reset memory and encode sentences 0..up_to_index (inclusive)."""
        self.memory.reset()
        for i in range(up_to_index + 1):
            person_name, place_name = story.sentences[i]
            person_vec = self.people.add(person_name)
            place_vec = self.places.add(place_name)
            self.memory.write(bind(person_vec, place_vec))

    def where_is(self, person_name: str) -> tuple[str, float]:
        """Answer 'Where is <person>?' via Unbind + cleanup."""
        if person_name not in self.people:
            return "<unknown>", 0.0
        person_vec = self.people.get(person_name)
        noisy_location = self.memory.query(person_vec)
        return self.places.cleanup(noisy_location)


# ----- Evaluation ----------------------------------------------------------

def evaluate(
    stories: list[Story],
    dim: int,
    decay: float,
    seed: int,
) -> dict:
    reasoner = VSAReasoner(dim=dim, decay=decay, seed=seed)

    correct = 0
    total = 0
    confidences_correct: list[float] = []
    confidences_wrong: list[float] = []
    details: list[dict] = []

    for story_idx, story in enumerate(stories):
        for person, expected, after_idx in story.queries:
            reasoner.encode_up_to(story, up_to_index=after_idx)
            predicted, confidence = reasoner.where_is(person)
            is_correct = predicted == expected
            correct += int(is_correct)
            total += 1
            (confidences_correct if is_correct else confidences_wrong).append(confidence)
            details.append({
                "story": story_idx,
                "person": person,
                "expected": expected,
                "predicted": predicted,
                "confidence": round(confidence, 4),
                "correct": is_correct,
            })

    accuracy = correct / max(total, 1)
    avg_conf_correct = sum(confidences_correct) / max(len(confidences_correct), 1)
    avg_conf_wrong = sum(confidences_wrong) / max(len(confidences_wrong), 1)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_confidence_when_correct": avg_conf_correct,
        "avg_confidence_when_wrong": avg_conf_wrong,
        "details": details,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=4096, help="HDC space dimensionality")
    p.add_argument("--decay", type=float, default=0.5, help="memory decay per write (<1 = recency bias)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stories", type=int, default=200, help="number of synthetic stories")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--sweep-dim", action="store_true", help="run with dim in {512, 1024, 2048, 4096}")
    args = p.parse_args()

    rng = random.Random(args.seed)
    stories = generate_synthetic_task1(args.stories, rng)
    n_questions = sum(len(s.queries) for s in stories)
    print(f"Phase -1: pure VSA baseline on synthetic bAbI task 1")
    print(f"Stories: {len(stories)}, questions: {n_questions}")

    dims_to_try = [512, 1024, 2048, 4096] if args.sweep_dim else [args.dim]

    for dim in dims_to_try:
        print()
        print(f"--- dim={dim}, decay={args.decay} ---")
        results = evaluate(stories, dim=dim, decay=args.decay, seed=args.seed)
        print(f"Accuracy: {results['accuracy']:.4f}  ({results['correct']}/{results['total']})")
        print(f"Avg confidence when correct: {results['avg_confidence_when_correct']:.4f}")
        if results["total"] - results["correct"] > 0:
            print(f"Avg confidence when wrong:   {results['avg_confidence_when_wrong']:.4f}")

        if args.verbose:
            print("\nSample predictions:")
            for row in results["details"][:15]:
                mark = "OK" if row["correct"] else "XX"
                print(
                    f"  [{mark}] Where is {row['person']}? -> {row['predicted']} "
                    f"(expected {row['expected']}, conf {row['confidence']})"
                )

    print()
    threshold = 0.95
    final = evaluate(stories, dim=dims_to_try[-1], decay=args.decay, seed=args.seed)
    if final["accuracy"] >= threshold:
        print(f"GATE PASSED: accuracy {final['accuracy']:.4f} >= {threshold}")
        print("VSA language works. Phase -1 green-lit. Proceed to Phase 0/1.")
    else:
        print(f"GATE FAILED: accuracy {final['accuracy']:.4f} < {threshold}")
        print("Something is wrong with the primitives. DO NOT proceed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
