"""
Milestone H.1 — First Rung-4 self-memory experiment.

The test:  can HDC self-memory store corrections for past mistakes and
retrieve them on re-encounter? This is a minimum-viable proof that the
architecture can 'learn from its own errors online' — a direct AGI
property that transformers famously lack.

Setup:
  Phase 1.  Run the task-agnostic reasoning block on a batch of stories.
            Record every (story_id, query_obj, predicted, true) tuple.

  Phase 2.  Take the subset of queries the reasoner got WRONG. For each,
            write a correction into self_memory as
                Bind(Bind(story_id_vec, obj_vec), true_answer_vec)

  Phase 3.  Re-encounter the failed queries. Retrieve from self_memory:
                Unbind(self_memory, Bind(story_id_vec, obj_vec))
            Clean up against the known places. If the retrieval matches
            the true answer, the correction was carried successfully.

  Phase 4.  Sanity check — for queries that were CORRECT in phase 1 (and
            therefore never written to self_memory), does a lookup return
            noise (as it should)? If self_memory instead returns a
            confident wrong answer, it is polluting clean queries and
            the mechanism is broken.

NO gradients. NO retraining. NO parameter updates. Just HDC memory
recording and recall. This is the simplest possible test of Rung 4.
"""

from __future__ import annotations

import random

import numpy as np

from hdc_lang import Codebook, HDCMemory, bind
from run_milestone_C import ReasoningBlock
from run_babi_task3 import Task3Reasoner, generate_task3


def story_vector(story_idx: int, dim: int, seed: int = 12345) -> np.ndarray:
    """Deterministic random bipolar vector for a given story index."""
    rng = np.random.default_rng(seed + story_idx)
    return rng.choice([-1.0, 1.0], size=dim).astype(np.float32)


def run_experiment(num_stories: int = 500, dim: int = 1024) -> dict:
    rng = random.Random(42)
    stories = generate_task3(num_stories, rng)
    print(f"Generated {len(stories)} stories, "
          f"{sum(len(s.queries) for s in stories)} queries")

    # One shared reasoner across all stories so the concept codebook is
    # consistent — 'apple' has the same vector everywhere.
    reasoner = Task3Reasoner(
        dim=dim, decay_holder=0.7, decay_location=0.8, decay_drop=0.8, seed=42
    )

    # ---- Phase 1: normal reasoning, collect results ------------------
    print()
    print("Phase 1: normal task-agnostic reasoning")
    results = []  # list of dicts
    for story_idx, story in enumerate(stories):
        reasoner.reset()
        for event in story.events:
            reasoner.observe(event)

        block = ReasoningBlock(
            concepts=reasoner.concepts,
            memories={
                "holder_of": reasoner.holder_of,
                "location_of": reasoner.location_of,
                "dropped_at": reasoner.dropped_at,
            },
            type_groups={"person": reasoner._people, "place": reasoner._places},
            max_iter=5,
        )

        for obj, expected in story.queries:
            pred, _ = block.solve(obj, target_type="place")
            # Ensure both obj and expected are in the shared codebook
            reasoner.concepts.add(obj)
            reasoner.concepts.add(expected)
            results.append({
                "story_idx": story_idx,
                "obj": obj,
                "predicted": pred,
                "true": expected,
                "correct": pred == expected,
            })

    total = len(results)
    correct = sum(r["correct"] for r in results)
    errors = total - correct
    accuracy = correct / total
    print(f"  {correct}/{total} correct ({accuracy * 100:.2f}%)")
    print(f"  {errors} errors to correct via self-memory")

    if errors == 0:
        print("No errors to correct. Experiment cannot run.")
        return {"phase1_accuracy": accuracy, "errors": 0}

    # ---- Phase 2: write corrections to self_memory -------------------
    print()
    print("Phase 2: writing corrections to self_memory")
    self_memory = HDCMemory(dim, decay=1.0)  # pure sum, no recency bias

    # Collect the set of concepts we know about, for the cleanup step later
    all_places_seen = set(r["true"] for r in results) | reasoner._places

    errors_log = [r for r in results if not r["correct"]]
    for r in errors_log:
        story_vec = story_vector(r["story_idx"], dim)
        obj_vec = reasoner.concepts.get(r["obj"])
        true_vec = reasoner.concepts.get(r["true"])
        key = bind(story_vec, obj_vec)
        self_memory.write(bind(key, true_vec))

    print(f"  wrote {len(errors_log)} correction traces into self_memory")

    # ---- Phase 3: retrieve corrections on the failed queries ---------
    print()
    print("Phase 3: retrieving from self_memory on previously-failed queries")
    fixed = 0
    weak_retrieval = 0
    wrong_retrieval = 0
    confidences = []

    for r in errors_log:
        story_vec = story_vector(r["story_idx"], dim)
        obj_vec = reasoner.concepts.get(r["obj"])
        key = bind(story_vec, obj_vec)
        retrieved = bind(self_memory.read(), key)  # Unbind == Bind in bipolar
        name, conf = reasoner.concepts.cleanup(retrieved, restrict_to=all_places_seen)
        confidences.append(conf)
        if name == r["true"]:
            fixed += 1
        elif conf < 0.2:
            weak_retrieval += 1
        else:
            wrong_retrieval += 1

    recovery = fixed / max(len(errors_log), 1)
    print(f"  fixed   : {fixed}/{len(errors_log)}  ({recovery * 100:.2f}%)")
    print(f"  wrong   : {wrong_retrieval}/{len(errors_log)}")
    print(f"  weak    : {weak_retrieval}/{len(errors_log)}")
    print(f"  avg conf on corrections: {sum(confidences) / max(len(confidences), 1):.3f}")

    # ---- Phase 4: sanity check on clean (unstored) queries -----------
    print()
    print("Phase 4: sanity check — unstored queries should NOT retrieve strongly")
    clean_results = [r for r in results if r["correct"]][:100]
    false_positives = 0
    clean_confidences = []

    for r in clean_results:
        story_vec = story_vector(r["story_idx"], dim)
        obj_vec = reasoner.concepts.get(r["obj"])
        key = bind(story_vec, obj_vec)
        retrieved = bind(self_memory.read(), key)
        name, conf = reasoner.concepts.cleanup(retrieved, restrict_to=all_places_seen)
        clean_confidences.append(conf)
        if conf > 0.3 and name != r["true"]:
            # Memory is returning a confident WRONG answer for a clean query.
            # That means self_memory is polluting the reasoner's normal path.
            false_positives += 1

    avg_clean_conf = sum(clean_confidences) / max(len(clean_confidences), 1)
    print(f"  checked {len(clean_results)} correct queries")
    print(f"  avg retrieval conf on unstored keys: {avg_clean_conf:.3f}")
    print(f"  false positives (conf > 0.3 & wrong): {false_positives}")

    # ---- Phase 5: integrated test — self_memory as fallback ----------
    print()
    print("Phase 5: integrated — normal reasoning + self_memory consultation")
    print("(For each error, first check self_memory, fall back to normal answer.)")

    total_after = 0
    correct_after = 0
    # Threshold is midway between "signal" (~0.25 on stored keys) and
    # "noise" (~0.05 on unstored keys), giving ~5x signal-to-noise margin.
    SELF_MEM_THRESHOLD = 0.15

    for r in results:
        total_after += 1
        story_vec = story_vector(r["story_idx"], dim)
        obj_vec = reasoner.concepts.get(r["obj"])
        key = bind(story_vec, obj_vec)
        retrieved = bind(self_memory.read(), key)
        name, conf = reasoner.concepts.cleanup(retrieved, restrict_to=all_places_seen)

        if conf > SELF_MEM_THRESHOLD:
            # Trust self_memory
            final = name
        else:
            # Fall through to original reasoner prediction
            final = r["predicted"]

        if final == r["true"]:
            correct_after += 1

    acc_after = correct_after / total_after
    delta = acc_after - accuracy
    print(f"  original accuracy : {accuracy * 100:.2f}%")
    print(f"  with self_memory  : {acc_after * 100:.2f}%")
    print(f"  delta             : {delta * 100:+.2f}%")

    return {
        "phase1_accuracy": accuracy,
        "errors": errors,
        "corrections_fixed": fixed,
        "corrections_recovery_rate": recovery,
        "phase5_accuracy": acc_after,
        "phase5_delta": delta,
        "false_positives_on_clean": false_positives,
    }


def main():
    print("=" * 70)
    print("Milestone H.1 — Rung 4 self-memory experiment")
    print("=" * 70)
    result = run_experiment(num_stories=500, dim=1024)
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
