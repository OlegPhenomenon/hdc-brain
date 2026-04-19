# Milestone B.2 Log — bAbI task 3 via branching VSA reasoning

**Date:** 2026-04-11
**Status:** GATE PASSED on first attempt — no hyperparameter tuning needed
**Code:** `hdc-cogito/run_babi_task3.py`
**Parent:** Milestone B of `docs/planning/hdc-cogito-proposal.md` §12

## Goal

Extend pure-VSA reasoning from chains (task 2) to **branching**. Task 3
adds DROP events, which break the simple holder → location chain: an
object stays where it was dropped, not where the previous holder has
since moved.

Example:
```
Mary picked up the apple.
Mary moved to the office.
Mary dropped the apple.
Mary moved to the kitchen.
Where is the apple?   -> office  (NOT kitchen)
```

This forces the reasoner to choose between two strategies at query
time and do so correctly based on what is actually in memory.

## Design

Three decaying HDC memories, built on top of the same `hdc_lang`
primitives as Milestones A and B.1:

| Memory | Writer | Binding | Default decay |
|-|-|-|-|
| `holder_of`   | pickup, drop | `bind(obj, person)` or `bind(obj, DROPPED_MARKER)` | 0.7 |
| `location_of` | move | `bind(person, place)` | 0.8 |
| `dropped_at`  | drop | `bind(obj, place)` | 0.8 |

**Key mechanism 1 — the DROPPED marker.** A special concept vector
`<<DROPPED>>` is added to the codebook at init. On a drop event, the
reasoner writes `bind(obj, DROPPED_MARKER)` into `holder_of`. Subsequent
`cleanup(Unbind(holder_of, obj), people ∪ {DROPPED})` returns DROPPED
when the most recent state is a drop, and returns a real person when
the most recent state is a pickup. No destructive operation needed;
the decay and the new write together age-out the previous holder.

**Key mechanism 2 — memory consult during encoding.** A drop event has
the form `("drop", person, obj)` — it does NOT carry the location. The
reasoner infers the drop location by querying its own `location_of`
memory: `cleanup(Unbind(location_of, person), places)`. This is the
first time in the project that memory is consulted during encoding,
not just during query. It is a minimal version of the iterative
reasoning that Milestone C will generalize.

**Key mechanism 3 — branching query logic.** At query time:

```
holder = cleanup(Unbind(holder_of, obj), people ∪ {DROPPED})
if holder == DROPPED:
    place = cleanup(Unbind(dropped_at,  obj),    places)      # direct
else:
    place = cleanup(Unbind(location_of, holder), places)      # chain
```

No learned component decides between the branches. The decision is
driven by which marker came out of the HDC cleanup step — which is
itself a function of what is currently strongest in `holder_of`.

## Results — first attempt, no tuning

Decay values carried over directly from Milestone B.1
(`holder=0.7`, `location=0.8`). `drop=0.8` was set by analogy to
`location` (both are "slower forgetting" memories that should outlive
unrelated writes).

500 synthetic stories, 750 questions, 6–14 events per story, mix of
move/pickup/drop events.

| dim | overall | chain branch | dropped branch | speed (q/s) |
|-|-|-|-|-|
| 128  | 0.9120 | 0.8861 (509q) | 0.9668 (241q) |  8,138 |
| 256  | 0.9653 | 0.9540 (500q) | 0.9880 (250q) | 13,558 |
| 512  | 0.9840 | 0.9780 (500q) | 0.9960 (250q) | 12,861 |
| 1024 | 0.9933 | 0.9919 (496q) | 0.9961 (254q) | 12,280 |
| 2048 | 0.9960 | 0.9960 (496q) | 0.9961 (254q) |  8,807 |
| 4096 | 0.9973 | 0.9980 (495q) | 0.9961 (255q) |  8,075 |

**Gate passed at D ≥ 256.** At D=1024 we get 99.33% accuracy across
both branches. At D=4096, 99.73%. No per-task tuning was required —
the decay values from task 2 worked cleanly on task 3.

Note that the branches are balanced: out of 750 questions, ~500 go
through the chain and ~250 go through the dropped path. Both branches
reach near-ceiling accuracy independently. This is the honest proof
that the two strategies are genuinely independent — we are not cheating
by letting one carry the other.

## Findings

1. **Branching works in pure VSA.** The decision between "chain" and
   "direct lookup" emerges from a single cleanup step against an
   augmented candidate set (people ∪ DROPPED). No learned gating
   function, no softmax over strategies, no if-else on content. The
   reasoner picks the correct branch purely because `holder_of` now
   has either a real person or the DROPPED marker as its dominant
   entry for the queried object.

2. **Decay settings from task 2 generalized cleanly to task 3.** This
   is evidence that per-memory decay is a **principled** axis, not
   per-task tuning. Memories with similar temporal semantics want
   similar decay values — `location_of` and `dropped_at` are both
   "slow forgetting" because they store facts that should survive
   many unrelated intervening writes.

3. **Memory-consult during encoding worked on first try.** When
   processing a drop, the reasoner queries its own `location_of` to
   infer the drop location, and the inferred location is correct in
   ~100% of cases at D ≥ 512. This is a crucial proof point for
   Milestone C: iterative reasoning blocks will do many more such
   self-queries, and we now know the primitives can support them.

4. **No destructive operations were needed.** The obvious temptation
   with drops was to "cancel" the previous holder binding (e.g. write
   a negative of it). We avoided this entirely by using a marker and
   letting decay handle the aging. Destructive operations in HDC are
   brittle and we should continue to avoid them.

5. **Branching does not add meaningful cost.** Speed at D=1024 is
   ~12k q/sec, vs ~15k for task 2. The extra ~25% is the cost of one
   additional Unbind + cleanup step (for the drop branch) or the
   augmented cleanup candidate set. Still in the five-digit q/sec range.

## Implications for the proposal document

(Collecting observations for when I eventually fold them back in.)

- **§4.5 self-reflection** should note that memory-consult during
  encoding is a form of proto-rung-1 behavior: the system inspects
  its own state to decide what to write. The drop handler in Task3Reasoner
  is a minimal concrete instance.
- **§7 the ladder** — Rung 2 (self-model) and Rung 4 (error archaeology)
  now have concrete precursors: the `DROPPED` marker is a proto-self-state
  flag, and the `dropped_at` memory is a proto-episodic-trace.
- **§14 open questions** — add: "Does every 'special state' in HDC-Cogito
  need its own marker vector in the concept codebook, or can we share
  one generic 'transition' marker?"

Not editing the doc now — these are notes for Milestone C+.

## Next step — Milestone B.3 (optional) or Milestone C

Two choices:

- **Milestone B.3** — tasks 4 (spatial relations) and 5 (three-argument
  facts like "Mary gave the cake to Fred"). These introduce proper
  multi-role structures and explicit role vocabulary (agent, patient,
  recipient, theme). Still no learned components.
- **Milestone C** — first iterative reasoning block. One HDC block
  that reads the current state, selects an Unbind via cleanup against
  the role codebook, applies it, and repeats until a confidence
  threshold is met. Fixed dynamics, no learned weights. Validate that
  iteration without training improves accuracy on the hardest bAbI
  tasks we already solve.

My recommendation: **go to Milestone C next**. We already have
enough evidence that static VSA handles bAbI-class problems. What we
don't yet have is evidence that **iteration itself** adds value. That
is the core hypothesis of HDC-Cogito (see proposal §3 H2), and we
should test it before doing more task-specific handlers.
