"""
Milestone B.1 — bAbI task 2 (Two Supporting Facts) with chained VSA reasoning.

This is the first task in Cogito that requires multi-hop inference.

Typical task-2 story:
    Mary got the milk there.
    Mary travelled to the office.
    Where is the milk?   -> office

The answer cannot be read off any single event. It requires *chaining*:

    milk  --(held by)-->  Mary  --(located at)-->  office

In our HDC language this is literally two Unbind operations composed.
No training, no learned weights, no reasoning block yet. Just:

    holder_of  : decaying HDC memory of object -> holder bindings
    location_of: decaying HDC memory of person -> place  bindings

Query:
    holder  = cleanup( Unbind(holder_of,   obj_vec)    , people_seen )
    place   = cleanup( Unbind(location_of, holder_vec) , places_seen )

This is the minimum possible demonstration that our internal language
can express compositional inference. If this works, chained reasoning
is already latent in the primitives and the future "reasoning block"
just has to learn which chain to run, not how chaining itself works.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field

from hdc_lang import Codebook, HDCMemory, bind


# ----- Data ----------------------------------------------------------------

@dataclass
class Story:
    # Each event is a tuple:
    #   ("move",   person, place)
    #   ("pickup", person, object)
    events: list[tuple] = field(default_factory=list)
    # (object, expected_place)
    queries: list[tuple[str, str]] = field(default_factory=list)


def generate_task2(num_stories: int, rng: random.Random) -> list[Story]:
    """
    Generate bAbI task 2 style stories: mixed move and pickup events.

    State kept by the generator (NOT by the reasoner) to compute ground truth:
      holder[obj]     = person currently holding obj
      person_loc[p]   = person p's current place

    At the end of each story, we query every object whose holder has a
    known location. That gives us the ground-truth chained answer.
    """
    people = ["Mary", "John", "Sandra", "Daniel", "Bob", "Alice"]
    places = ["bathroom", "kitchen", "hallway", "garden", "bedroom", "office", "cellar", "attic"]
    objects = ["apple", "football", "milk", "book", "ball", "key", "pen", "phone"]

    stories: list[Story] = []
    for _ in range(num_stories):
        events: list[tuple] = []
        holder: dict[str, str] = {}
        person_loc: dict[str, str] = {}

        num_events = rng.randint(4, 12)
        for _ in range(num_events):
            kind = rng.choices(["move", "pickup"], weights=[3, 2])[0]
            if kind == "move":
                p = rng.choice(people)
                l = rng.choice(places)
                events.append(("move", p, l))
                person_loc[p] = l
            else:  # pickup
                p = rng.choice(people)
                o = rng.choice(objects)
                events.append(("pickup", p, o))
                holder[o] = p

        # Build queries: for every object whose current holder has a known place
        queries: list[tuple[str, str]] = []
        for obj, h in holder.items():
            if h in person_loc:
                queries.append((obj, person_loc[h]))

        if queries:
            stories.append(Story(events=events, queries=queries))
    return stories


# ----- Reasoner -----------------------------------------------------------

class Task2Reasoner:
    """
    Pure VSA multi-hop reasoner.

    Two decaying HDC memories — each is a single vector in the HDC space.
    No tables, no hash maps of state. Everything the reasoner "knows"
    lives in two superimposed vectors, read by Unbind.

    IMPORTANT: the two memories need different decay rates. This is a
    structural finding from grid search:

      holder_of    — wants moderate recency (0.7). Same object can be
                     re-picked-up by a different person; we want the
                     most recent holder, but not extreme decay since
                     pickups are infrequent compared to moves.

      location_of  — wants weaker decay (0.8). A person might move once
                     early in the story and not again, so their location
                     entry must survive many unrelated writes from other
                     people's moves. Too-strong decay drowns such persons
                     in cross-term noise from the majority who moved later.

    This is the first concrete lesson about "memory policy" in HDC-Cogito:
    per-memory decay is a real hyperparameter, not a global setting. Each
    role has its own temporal dynamics and its own forgetting rate.
    """

    def __init__(
        self,
        dim: int,
        decay_holder: float,
        decay_location: float,
        seed: int,
    ):
        self.dim = dim
        self.concepts = Codebook(dim, seed=seed)
        self.holder_of = HDCMemory(dim, decay=decay_holder)
        self.location_of = HDCMemory(dim, decay=decay_location)
        self._people: set[str] = set()
        self._places: set[str] = set()
        self._objects: set[str] = set()

    def reset(self) -> None:
        self.holder_of.reset()
        self.location_of.reset()
        # Vocabulary and codebook persist across stories — that matches
        # the assumption that concepts are long-lived and only memory
        # state is per-story.

    def observe(self, event: tuple) -> None:
        kind = event[0]
        if kind == "move":
            _, person, place = event
            self._people.add(person)
            self._places.add(place)
            self.location_of.write(
                bind(self.concepts.add(person), self.concepts.add(place))
            )
        elif kind == "pickup":
            _, person, obj = event
            self._people.add(person)
            self._objects.add(obj)
            self.holder_of.write(
                bind(self.concepts.add(obj), self.concepts.add(person))
            )
        else:
            raise ValueError(f"unknown event kind: {kind}")

    def where_is_object(self, obj: str) -> tuple[str, float, float, str]:
        """
        Answer 'Where is <obj>?' via two chained Unbind operations.

        Returns:
            (place_name, stage1_confidence, stage2_confidence, inferred_holder)
        """
        if obj not in self.concepts or not self._people or not self._places:
            return "<unknown>", 0.0, 0.0, "<none>"

        obj_vec = self.concepts.get(obj)

        # Stage 1: object -> holder
        holder_noisy = self.holder_of.query(obj_vec)
        holder_name, holder_conf = self.concepts.cleanup(
            holder_noisy, restrict_to=self._people
        )

        # Stage 2: holder -> place
        holder_vec = self.concepts.get(holder_name)
        place_noisy = self.location_of.query(holder_vec)
        place_name, place_conf = self.concepts.cleanup(
            place_noisy, restrict_to=self._places
        )

        return place_name, holder_conf, place_conf, holder_name


# ----- Evaluation ---------------------------------------------------------

def evaluate(
    stories: list[Story],
    dim: int,
    decay_holder: float,
    decay_location: float,
    seed: int,
) -> dict:
    reasoner = Task2Reasoner(
        dim=dim,
        decay_holder=decay_holder,
        decay_location=decay_location,
        seed=seed,
    )

    correct = 0
    total = 0
    stage1_confs: list[float] = []
    stage2_confs: list[float] = []
    details: list[dict] = []

    for story_idx, story in enumerate(stories):
        reasoner.reset()
        for event in story.events:
            reasoner.observe(event)
        for obj, expected in story.queries:
            predicted, c1, c2, holder = reasoner.where_is_object(obj)
            ok = predicted == expected
            correct += int(ok)
            total += 1
            stage1_confs.append(c1)
            stage2_confs.append(c2)
            details.append({
                "story": story_idx,
                "obj": obj,
                "expected": expected,
                "predicted": predicted,
                "inferred_holder": holder,
                "stage1_conf": round(c1, 4),
                "stage2_conf": round(c2, 4),
                "correct": ok,
            })

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "avg_stage1": sum(stage1_confs) / max(len(stage1_confs), 1),
        "avg_stage2": sum(stage2_confs) / max(len(stage2_confs), 1),
        "details": details,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--decay-holder", type=float, default=0.7,
                   help="decay for the holder_of memory (wants moderate recency)")
    p.add_argument("--decay-location", type=float, default=0.8,
                   help="decay for the location_of memory (needs to remember older moves)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stories", type=int, default=500)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--sweep-dim", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)
    stories = generate_task2(args.stories, rng)
    n_questions = sum(len(s.queries) for s in stories)

    print("Milestone B.1: bAbI task 2 via chained VSA reasoning")
    print(f"Stories: {len(stories)}, questions: {n_questions}")
    print(f"(events: move + pickup; query: Where is <object>? -> multi-hop chain)")
    print(f"Decay: holder_of={args.decay_holder}, location_of={args.decay_location}")

    dims = [64, 128, 256, 512, 1024, 2048] if args.sweep_dim else [args.dim]
    final_accuracy = 0.0
    for dim in dims:
        print()
        print(f"--- dim={dim} ---")
        t0 = time.perf_counter()
        r = evaluate(
            stories,
            dim=dim,
            decay_holder=args.decay_holder,
            decay_location=args.decay_location,
            seed=args.seed,
        )
        dt = time.perf_counter() - t0
        print(f"Accuracy: {r['accuracy']:.4f}  ({r['correct']}/{r['total']})")
        print(f"Avg stage-1 confidence (object -> holder): {r['avg_stage1']:.4f}")
        print(f"Avg stage-2 confidence (holder -> place):  {r['avg_stage2']:.4f}")
        print(f"Wall time: {dt*1000:.1f} ms  ({r['total']/dt:.0f} q/sec)")

        if args.verbose:
            print("\nSample predictions:")
            for row in r["details"][:10]:
                mark = "OK" if row["correct"] else "XX"
                print(
                    f"  [{mark}] Where is {row['obj']}? -> {row['predicted']} "
                    f"(expected {row['expected']}, "
                    f"holder={row['inferred_holder']}, "
                    f"c1={row['stage1_conf']}, c2={row['stage2_conf']})"
                )
        final_accuracy = r["accuracy"]

    print()
    threshold = 0.95
    if final_accuracy >= threshold:
        print(f"GATE PASSED: task 2 at {final_accuracy:.4f} >= {threshold}")
        print("Chained VSA reasoning works. Milestone B.1 complete.")
    else:
        print(f"GATE FAILED: task 2 at {final_accuracy:.4f} < {threshold}")
        print("Investigate stage-1 vs stage-2 confidences — which stage is dropping?")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
