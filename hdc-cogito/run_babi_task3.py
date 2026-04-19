"""
Milestone B.2 — bAbI task 3 (Three Supporting Facts) with branching VSA reasoning.

Task 3 adds DROPS. A drop breaks the simple holder -> location chain
from task 2, because the object stays where it was dropped — not where
the former holder has since moved.

Example:
    Mary picked up the apple.
    Mary moved to the office.
    Mary dropped the apple.
    Mary moved to the kitchen.
    Where is the apple?   -> office  (NOT kitchen)

This requires **branching reasoning**: the model must decide whether to
follow the chain (obj -> holder -> location) or fall back to a direct
lookup (obj -> dropped_at place). It does so by comparing the
confidence of each path.

Two structural additions on top of Milestone B.1 (task 2):

  1. Third memory: `dropped_at` — written on drop events.

  2. A special "DROPPED" marker vector, added to the concept codebook.
     On drop, the reasoner writes `bind(obj, DROPPED_MARKER)` into
     holder_of. Cleanup over {people, DROPPED_MARKER} then returns
     DROPPED when the most recent state is a drop, and returns a real
     person when the most recent state is a pickup. This cleanly
     ages-out stale holders without any destructive operation.

A subtle bonus: when processing a drop event, the reasoner does not
receive the drop location as input. It **queries its own `location_of`
memory** to figure out where the dropper currently is. This is the
first time in the project that memory is consulted during ENCODING,
not just during query. It is a micro-version of the iterative reasoning
that Milestone C will generalize.

No training. No learned weights. Just four primitives plus one
cleanup-driven branch.
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
    # Event tuples:
    #   ("move",   person, place)
    #   ("pickup", person, object)
    #   ("drop",   person, object)
    # Drop does NOT carry the location — the reasoner must infer it.
    events: list[tuple] = field(default_factory=list)
    # (object, expected_place)
    queries: list[tuple[str, str]] = field(default_factory=list)


def generate_task3(num_stories: int, rng: random.Random) -> list[Story]:
    """
    Generate bAbI task 3 style stories: move, pickup, and drop events.

    Ground-truth state is kept in the generator (not in the reasoner):
        holder[obj]      = person currently holding obj (if any)
        person_loc[p]    = last known place of person p
        object_loc[obj]  = explicit location of obj (only set after a drop)

    An object that is currently held answers "Where is <obj>?" by its
    holder's current location. A dropped object answers by its
    object_loc. We only emit queries for objects with a defined answer.
    """
    people = ["Mary", "John", "Sandra", "Daniel", "Bob", "Alice"]
    places = ["bathroom", "kitchen", "hallway", "garden", "bedroom", "office", "cellar", "attic"]
    objects = ["apple", "football", "milk", "book", "ball", "key", "pen", "phone"]

    stories: list[Story] = []
    for _ in range(num_stories):
        events: list[tuple] = []
        holder: dict[str, str] = {}
        person_loc: dict[str, str] = {}
        object_loc: dict[str, str] = {}

        num_events = rng.randint(6, 14)
        for _ in range(num_events):
            kind = rng.choices(["move", "pickup", "drop"], weights=[4, 3, 2])[0]
            if kind == "move":
                p = rng.choice(people)
                l = rng.choice(places)
                events.append(("move", p, l))
                person_loc[p] = l
            elif kind == "pickup":
                p = rng.choice(people)
                o = rng.choice(objects)
                events.append(("pickup", p, o))
                # If someone else was holding, they lose it
                holder[o] = p
                # No longer at a fixed location while held
                object_loc.pop(o, None)
            elif kind == "drop":
                # Can only drop what is currently held, and only if the
                # holder has a known location.
                droppable = [
                    (o, h) for o, h in holder.items() if h in person_loc
                ]
                if not droppable:
                    continue
                o, h = rng.choice(droppable)
                events.append(("drop", h, o))
                object_loc[o] = person_loc[h]
                del holder[o]

        # Build queries
        queries: list[tuple[str, str]] = []
        for obj in set(objects):
            if obj in holder:
                h = holder[obj]
                if h in person_loc:
                    queries.append((obj, person_loc[h]))
            elif obj in object_loc:
                queries.append((obj, object_loc[obj]))

        if queries:
            stories.append(Story(events=events, queries=queries))
    return stories


# ----- Reasoner -----------------------------------------------------------

class Task3Reasoner:
    """
    VSA reasoner with branching inference.

    Three decaying HDC memories:
      holder_of   - bind(obj, person) on pickup, bind(obj, DROPPED) on drop
      location_of - bind(person, place) on move
      dropped_at  - bind(obj, place) on drop (place inferred from location_of)

    Plus a special DROPPED marker vector in the concept codebook.

    Query strategy:
      1. cleanup Unbind(holder_of, obj) against {people, DROPPED}
      2. if result is DROPPED -> Unbind(dropped_at, obj) -> place
         else                 -> Unbind(location_of, holder) -> place

    On a drop event, the reasoner does not receive the location. It
    queries its own location_of memory for the dropper's latest place
    and uses that as the drop location. This is the first case of
    memory-consult during encoding.
    """

    DROPPED_NAME = "<<DROPPED>>"

    def __init__(
        self,
        dim: int,
        decay_holder: float,
        decay_location: float,
        decay_drop: float,
        seed: int,
    ):
        self.dim = dim
        self.concepts = Codebook(dim, seed=seed)
        self.dropped_marker = self.concepts.add(self.DROPPED_NAME)
        self.holder_of = HDCMemory(dim, decay=decay_holder)
        self.location_of = HDCMemory(dim, decay=decay_location)
        self.dropped_at = HDCMemory(dim, decay=decay_drop)
        self._people: set[str] = set()
        self._places: set[str] = set()
        self._objects: set[str] = set()

    def reset(self) -> None:
        self.holder_of.reset()
        self.location_of.reset()
        self.dropped_at.reset()

    def _holder_candidates(self) -> set[str]:
        return self._people | {self.DROPPED_NAME}

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
        elif kind == "drop":
            _, person, obj = event
            self._people.add(person)
            self._objects.add(obj)
            # First: figure out WHERE the dropper currently is,
            # by querying our own location_of memory. This is
            # memory-consult during encoding — a micro-reasoning step.
            person_vec = self.concepts.add(person)
            drop_place: str | None = None
            if self._places:
                noisy = self.location_of.query(person_vec)
                drop_place, _ = self.concepts.cleanup(
                    noisy, restrict_to=self._places
                )
            # Record the drop location if we could resolve it
            if drop_place is not None and drop_place in self.concepts:
                self.dropped_at.write(
                    bind(self.concepts.add(obj), self.concepts.get(drop_place))
                )
            # Mark holder_of as DROPPED — this ages out the previous holder
            # binding and causes subsequent Unbind to prefer DROPPED in cleanup.
            self.holder_of.write(
                bind(self.concepts.add(obj), self.dropped_marker)
            )
        else:
            raise ValueError(f"unknown event kind: {kind}")

    def where_is_object(self, obj: str) -> tuple[str, float, float, str]:
        """
        Branching query.

        Returns (place, holder_stage_confidence, place_stage_confidence, inferred_holder_or_flag).
        """
        if obj not in self.concepts or not self._people or not self._places:
            return "<unknown>", 0.0, 0.0, "<none>"

        obj_vec = self.concepts.get(obj)

        # Step 1: who / what is bound to the object?
        holder_noisy = self.holder_of.query(obj_vec)
        holder_name, holder_conf = self.concepts.cleanup(
            holder_noisy, restrict_to=self._holder_candidates()
        )

        if holder_name == self.DROPPED_NAME:
            # Branch A: direct drop lookup
            place_noisy = self.dropped_at.query(obj_vec)
            place_name, place_conf = self.concepts.cleanup(
                place_noisy, restrict_to=self._places
            )
            return place_name, holder_conf, place_conf, "<dropped>"
        else:
            # Branch B: chain through location_of
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
    decay_drop: float,
    seed: int,
) -> dict:
    reasoner = Task3Reasoner(
        dim=dim,
        decay_holder=decay_holder,
        decay_location=decay_location,
        decay_drop=decay_drop,
        seed=seed,
    )

    correct = 0
    total = 0
    by_branch = {"chain": [0, 0], "dropped": [0, 0]}
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
            branch = "dropped" if holder == "<dropped>" else "chain"
            by_branch[branch][1] += 1
            if ok:
                by_branch[branch][0] += 1
            details.append({
                "story": story_idx,
                "obj": obj,
                "expected": expected,
                "predicted": predicted,
                "holder_or_flag": holder,
                "branch": branch,
                "stage1_conf": round(c1, 4),
                "stage2_conf": round(c2, 4),
                "correct": ok,
            })

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "chain_accuracy": by_branch["chain"][0] / max(by_branch["chain"][1], 1),
        "chain_count": by_branch["chain"][1],
        "dropped_accuracy": by_branch["dropped"][0] / max(by_branch["dropped"][1], 1),
        "dropped_count": by_branch["dropped"][1],
        "details": details,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--decay-holder", type=float, default=0.7)
    p.add_argument("--decay-location", type=float, default=0.8)
    p.add_argument("--decay-drop", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stories", type=int, default=500)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--sweep-dim", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)
    stories = generate_task3(args.stories, rng)
    n_q = sum(len(s.queries) for s in stories)

    print("Milestone B.2: bAbI task 3 via branching VSA reasoning")
    print(f"Stories: {len(stories)}, questions: {n_q}")
    print(f"(events: move + pickup + drop; query: Where is <object>?)")
    print(f"Decay: holder={args.decay_holder}, location={args.decay_location}, drop={args.decay_drop}")

    dims = [128, 256, 512, 1024, 2048, 4096] if args.sweep_dim else [args.dim]
    final_acc = 0.0
    for dim in dims:
        print()
        print(f"--- dim={dim} ---")
        t0 = time.perf_counter()
        r = evaluate(
            stories,
            dim=dim,
            decay_holder=args.decay_holder,
            decay_location=args.decay_location,
            decay_drop=args.decay_drop,
            seed=args.seed,
        )
        dt = time.perf_counter() - t0
        print(f"Overall accuracy: {r['accuracy']:.4f}  ({r['correct']}/{r['total']})")
        print(f"  Chain branch (held objects):   {r['chain_accuracy']:.4f}  ({r['chain_count']} questions)")
        print(f"  Dropped branch (dropped obj):  {r['dropped_accuracy']:.4f}  ({r['dropped_count']} questions)")
        print(f"Wall time: {dt*1000:.1f} ms  ({r['total']/dt:.0f} q/sec)")

        if args.verbose:
            print("\nSample predictions:")
            for row in r["details"][:12]:
                mark = "OK" if row["correct"] else "XX"
                print(
                    f"  [{mark}] [{row['branch']:>7}] Where is {row['obj']}? -> {row['predicted']} "
                    f"(expected {row['expected']}, via {row['holder_or_flag']}, "
                    f"c1={row['stage1_conf']}, c2={row['stage2_conf']})"
                )
        final_acc = r["accuracy"]

    print()
    threshold = 0.95
    if final_acc >= threshold:
        print(f"GATE PASSED: task 3 at {final_acc:.4f} >= {threshold}")
        print("Branching VSA reasoning works. Milestone B.2 complete.")
    else:
        print(f"GATE NOT YET: task 3 at {final_acc:.4f} < {threshold}")
        print("Tune decay_drop or dim; check which branch is failing.")


if __name__ == "__main__":
    main()
