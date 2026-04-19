# HDC-Cogito: Research Proposal (v2)

**Codename**: HDC-Cogito
**Version tag**: v19 (v15–v18 reserved for prior experiments)
**Date**: 2026-04-11 (v1 draft), 2026-04-11 (v2 major revision, same day)
**Status**: Research proposal — Phase -1 ready to start immediately; main architecture work after v14.1 English pretraining completes
**Author**: Oleg (vision), Claude (structuring)

---

## 0. Handoff to Next Session

**If you (Claude) are reading this in a fresh session:**

This is the master proposal for the next major HDC-Brain version. It has been through a major revision based on a long conversation with Oleg on 2026-04-11. The v2 revision **removes all transformer-comparison framing** and introduces the **VSA-native internal language** as the foundation.

**Critical context you must internalize before reading further:**

1. **This is NOT an alternative transformer.** It is a different computational substrate.
2. **Do NOT use the word "parameters" as a design axis for Cogito.** That word belongs to another paradigm. Our axes are: HDC space capacity, vocabulary size, reasoning iterations, and data volume.
3. **v14.1 is already working** on the vast.ai GPU server, generating meaningful English text, with zero transformer components. It proves the base hypothesis. Cogito is the next step from this proven substrate — not a leap into the unknown.
4. **The internal language is VSA** (Vector Symbolic Architectures), an existing tradition started by Plate 1995. We are not inventing symbolic logic and translating it to HDC. The language **is** the HDC operations themselves.
5. **AGI / self-awareness is the north star, not a per-commit criterion.** There is a six-rung ladder (see §7). Each rung is independently measurable. We build up one rung at a time.

**Read in order:**
1. This document, cover to cover
2. Tony Plate (1995), "Holographic Reduced Representations" — dissertation/book, chapters 3–4 minimum
3. Pentti Kanerva (2009), "Hyperdimensional Computing: An Introduction"
4. Ross Gayler (2003), "Vector Symbolic Architectures Answer Jackendoff's Challenges" — 6 pages
5. `docs/planning/idea-emergent-internal-language.md`
6. `docs/planning/idea-three-phase-training.md`
7. Memory files: `user_learning_profile_detailed.md`, `feedback_book_style.md`, `project_hdc_brain_v14_1_fix.md`

**Oleg's full context:**
- Rails backend developer, zero formal ML education
- Built HDC-Brain v14.1, currently training on English (FineWeb, ~3B tokens, already generating meaningful text)
- Long-term goal: reasoning-capable language model that runs on minimal hardware ("on a calculator")
- Narrative-analogical thinker: explain via story → term → code → formula
- Working style: intuition-driven research, beginner's mind, decisive action, fix problems immediately

**Do not jump into implementation.** Confirm with Oleg which section resonates. Phase -1 (§8.1) is the first concrete step — small, safe, and decides whether the whole language premise holds.

---

## 1. TL;DR

**HDC-Cogito explores a different way to build a language model: one where knowledge emerges from iterative reasoning over data in a structured high-dimensional space, using only HDC-native operations.**

The model has three kinds of building blocks, all of which are vectors in the same HDC space:

- **Concepts** — things that exist (cat, river, Paris, kick, red)
- **Roles** — positions in a structure (agent, patient, location, cause, time)
- **Markers** — modalities (question, negation, past, possible)

Four primitive operations compose these into meaning:

- **Bind** — attach a concept to a role
- **Bundle** — combine attachments into a structure
- **Permute** — express order
- **Unbind** — extract information (answer a question, retrieve a memory)

Reasoning is **iterative application** of these operations under learned control: the model re-enters its own state multiple times, each iteration deepening structure, until a confidence gate signals completion. Simple inputs take one iteration. Hard inputs take many. The depth of reasoning is not the depth of stacked layers — it is the number of times a single reasoning block is invoked.

Self-reflection is a **first-class mechanism** built on the same substrate: the model has a `self` vector in its own memory, it can query its own past predictions, and it can consolidate memory during idle time. These capabilities form a ladder leading toward functional self-awareness — the AGI north star, approached one rung at a time.

---

## 2. Why Now

### 2.1 The Core Insight

Oleg's observation on 2026-04-11:

> "Почему параметры так важны? Мне кажется нужно отходить от такой концепции как параметры, а ключевым должно быть количество данных и количество итераций через слои рассуждения."

This is the foundation of the entire proposal. "Parameter count" is a measure borrowed from another paradigm, where knowledge lives in dense weight matrices. In HDC, knowledge does not live in weights — it lives in **structured relationships between vectors in a high-dimensional space**.

The space has fixed capacity, determined by dimension D and the geometry of bipolar vectors (Kanerva's capacity theory). What varies over training and use:

1. **How much data passes through the space** — more exposure → more connections
2. **How many reasoning iterations each piece of data receives** — more iterations → tighter integration with existing structure
3. **How rich the vocabulary is** — how many concepts/roles/markers the system has available

Knowledge = f(data volume, reasoning iterations, vocabulary richness, space capacity).

None of these is "parameters". The word simply does not apply to this architecture.

### 2.2 Why HDC, Why Now

HDC-Brain v14.1 has already proven the base hypothesis: a non-transformer architecture built entirely on HDC operations can learn English and generate meaningful text. That was the hard part. Cogito is the natural next step: **given that the substrate works, what is the minimal, principled way to make it reason and reflect on itself?**

The answer from 30 years of VSA literature (Plate 1995, Kanerva, Gayler, Eliasmith) is: **you already have the language — it is the HDC operations themselves**. You just need to use them as primitives of a reasoning system, not merely as building blocks of a sequence model.

### 2.3 The Target

A reasoning system small enough and cheap enough to run on calculator-class hardware. Not because size is the goal for its own sake, but because **HDC operations are cheap by construction**: Bind is elementwise multiply, Bundle is sum, Permute is cyclic shift, Unbind is the same as Bind in bipolar space. These operations map to CPU POPCOUNT and XOR. No matrix multiply units required. A well-designed HDC reasoner can run on hardware where a conventional neural network cannot even boot.

This is genuinely novel territory. No one has built a VSA-based language model that leverages self-reflection and adaptive reasoning iterations. Worth publishing regardless of absolute benchmark numbers.

---

## 3. Core Hypotheses

**H1 — Knowledge as structured relationships.** A language model built entirely from HDC primitives (Concepts + Roles + Markers + Bind/Bundle/Permute/Unbind), with no dense weight matrices of the conventional neural-network kind, can learn to represent and retrieve structured knowledge from natural language. v14.1 partially validates this for generation; Cogito completes it for reasoning.

**H2 — Iterative reasoning replaces depth. [EMPIRICALLY VALIDATED — Milestones C and C.1, 2026-04-11]** Variable-depth reasoning (1 to ~30 iterations through a single reasoning block) can solve tasks that conventional approaches would address by stacking more layers. Iteration count scales with task difficulty, not with architecture size.

**Confirmed:** bAbI task 2 (a 2-hop chain problem) goes from **0.00% accuracy at `max_iter=1`** to **98.65% at `max_iter=2`** using the same task-agnostic reasoning block — iteration is the difference between impossible and solved. On synthetic depth-controlled chains (Milestone C.1), the block reaches **100% accuracy on 20-hop reasoning at D=2048** using exactly 20 iterations per query, with a clean empirical staircase from depth 1 through 20. There is an observed scaling law `max_reachable_depth ≈ f(D)` — doubling D approximately doubles the deepest reasoning the block can handle reliably.

**H3 — Self-reflection as built-in capability.** Because HDC memory is transparent and compositional, the model can monitor itself, model itself, and replay its own experience — all using the same Bind/Unbind primitives used for language. This creates a natural substrate for functional self-awareness.

All three are falsifiable. Each has a concrete evaluation plan in §10.

---

## 4. Architectural Principles

### 4.1 Decouple Knowing from Reasoning

Traditional neural nets tangle "what is known" with "how to process it": both live in the same weights. In Cogito:

- **Vocabulary** (concepts, roles, markers) stores what exists in the model's world
- **Reasoning block** stores how to combine and transform — and there is only one block, reused iteratively
- **HDC memory** stores the structural relationships the system has built during exposure to data

Each component can evolve independently. The vocabulary can grow (new concepts discovered in data). The reasoning block can be refined (better iterative dynamics). The memory can be consolidated (replay during idle time). None of these requires full retraining.

### 4.2 Knowledge = Data × Iterations in a Structured Space

This replaces the "compute scaling vs parameter scaling" discussion from v1 of this document. The framing is different:

- **HDC space has fixed capacity.** For dimension D=4096 with bipolar vectors, Kanerva-style capacity theory tells us how many near-orthogonal concepts can coexist before collisions become significant. This is a **geometric** property, not an engineering one.
- **Data exposure fills the space.** Each piece of data, when processed, binds some concepts to some roles and bundles the result into memory. Over time the space accumulates structure.
- **Reasoning iterations integrate new data with existing structure.** A fresh input does not "just land" in memory. It passes through the reasoning block multiple times, each pass refining how it connects to prior knowledge.

Simple inputs need few iterations. Hard inputs need many. Background replay during idle time uses more iterations to revisit old data in light of new memory. All three axes (data, iterations, vocabulary) grow capability **without introducing a single dense weight matrix of the conventional kind**.

**Empirical scaling law (Milestone C.1).** On synthetic nested-container chains of controlled depth, the reachable reasoning depth is an observed function of HDC dimensionality:

| D    | max depth at 100% accuracy |
|-|-|
| 1024 | ~15 hops (collapses around depth 20) |
| 2048 | ≥20 hops |
| 4096 | ≥20 hops (not tested further)          |

Deeper reasoning is bounded by noise accumulation in the cleanup step — each hop adds cross-term noise, and higher D provides more headroom. This is a **geometric** property of the space, not a parameter count. Any phase of Cogito that expects deep multi-hop reasoning must plan for D ≥ 2048 by default.

### 4.3 VSA-Native, Not Symbolic Translation

Critical lesson from prior attempts (see `idea-emergent-internal-language.md`): hardcoded symbolic operators like `IF`, `CAUSE`, `HAS` all failed. Why? Because those symbols had no grounding — they were invented top-down and then bolted onto HDC.

The correct approach, established by Plate 1995 and refined by Kanerva, Gayler, and Eliasmith:

**The language primitives are not invented symbols. They are HDC operations themselves.**

- "Predicate X(Y)" is not a symbol. It is `Bind(X, Y)`.
- "A and B" is not a symbol. It is `Bundle(A, B)`.
- "Before / after" is not a symbol. It is `Permute`.
- "What is the agent of action K?" is not a parse tree. It is `Unbind(memory, agent_role)` after focusing on K.

There is no translation layer. There is no parser. **The language IS the mathematics.**

### 4.4 Vocabulary Is Small and Structured

v14.1 has one flat codebook of 32K token vectors. Cogito has a richer, **more structured** vocabulary:

- **Concept vectors** — the things the system talks about. Grows with data, bounded by capacity theory. Initial ~2,000 vectors; can reach tens of thousands.
- **Role vectors** — ~8–15 vectors representing positions in a structure. Drawn from frame semantics (Fillmore) and conceptual dependency (Schank): `agent`, `patient`, `theme`, `location`, `time`, `cause`, `instrument`, `goal`, `source`, `manner`, `possession`, `part`. **Declared, not learned** — random bipolar vectors at initialization, kept fixed.
- **Marker vectors** — ~5–8 vectors for modalities: `question`, `negation`, `possible`, `past`, `future`, `imperative`, `counterfactual`. Also declared, also fixed.
- **Token vectors** — used only at the input / output interface for reading and writing surface text. Conceptually separate from concepts.

The small size of roles and markers is the point. This is what Sowa (1984) called the "primitive types" — a handful of universal categories that combine to express arbitrary meaning. Their fixed random vectors function like the grammar of the space: orthogonal markers that let "Mary-as-agent" be distinguishable from "Mary-as-patient".

### 4.5 Self-Reflection as First-Class Mechanism

Because HDC memory is transparent (you can Unbind anything with any role and receive a real vector back), the system has a direct architectural affordance for self-inspection. This is not a philosophical claim — it is a structural fact. We exploit it.

Cogito v1 implements the first two rungs of the Self-Reflection Ladder (§7):

- **ConfidenceGate** — metacognition: "Am I confident enough to stop reasoning?"
- **Self vector** — a persistent HDC vector representing the system's identity, present in memory across sessions. Bindable to current goals, recent predictions, capabilities.
- **Self-memory** — structures like `Bundle(Bind(self, prediction), Bind(context, c), Bind(truth, t))` stored in a separate memory bank.
- **Self-query** — before making a prediction, the model retrieves "what did I do last time in a similar context, and was I correct?"

Higher rungs (background replay, error archaeology, goal awareness, self-modification) are v2+ research, but each has a concrete path built on these primitives.

### 4.6 Relations Are Learned Vectors, Not Labels

From `idea-three-phase-training.md`:

```
REL(Paris, France, r1)
```

`r1` is not the label "capital_of". It is an HDC vector that emerges from how `Paris` and `France` co-occur across many structures. Many relation vectors coexist in the same space. New relations are recognized by cosine similarity to existing ones. Advantages:

- Language-independent — Russian "столица" and English "capital" converge on the same relation vector
- Supports fuzzy relations — "almost-capital", "former-capital"
- Naturally handles unseen relations via interpolation

### 4.7 Per-Memory Decay Policy

*Empirical finding from Milestone B.1 — promoted to architectural principle.*

Different memory banks in Cogito have different temporal dynamics and **cannot share a single decay rate**. This is not a hyperparameter nit — it is a structural property of the architecture.

Each memory bank stores facts that change at a characteristic event rate. A person's location changes every time they move. An object's holder changes every time it is picked up or dropped. These event rates are **decoupled**, and a single global decay forces a compromise that fails for multi-hop reasoning.

**Concrete example (Milestone B.1):** with a global decay of 0.5, bAbI task 2 capped at ~93% even at D=2048, because `location_of` entries for infrequently-moving people were drowned in cross-term noise from many others' moves. Separating the decays (`holder_of = 0.7`, `location_of = 0.8`) lifted the same setup to 99.06%. The same settings carried over unchanged to task 3 (99.33%), confirming the rule is not per-task.

**Design rule.** Every memory bank in Cogito declares its own decay rate, justified by the event semantics of what it stores:

- **Low (0.5–0.7)** when the memory should forget older entries fast (high recency bias — e.g. pickups and drops that override each other).
- **High (0.8–0.95)** when older entries must survive many unrelated writes (events shared among many subjects — e.g. location updates).
- **1.0 (pure sum)** when all writes are equally important and no recency bias is appropriate (structural facts that do not change over time, e.g. "container A is inside container B" as used in Milestone C.1).

A learned reasoning block (Milestone G) will either accept these rates as declarations at initialization, or learn them from the training signal. Either way, the per-memory nature of decay is a first-class architectural axis, not a sweep over a scalar.

---

## 5. The Internal Language

This section is the core of the v2 revision. It defines the VSA-based internal language of Cogito.

### 5.1 The Language Is Not Syntax — It Is Legal Operations

Classical symbolic AI treated "internal language" as a syntax: rules for writing strings that represent meaning. This framing fails on a substrate like HDC, because HDC has no strings — it has vectors and operations on them.

The right framing: **the internal language is the set of legal transformations in HDC space**. It is not spoken or parsed. It is **enacted**. A sentence is a molecule assembled through valid reactions. A question is a reaction run in reverse.

This framing is precisely what the VSA tradition has been working on since 1995. We don't need to invent it — we need to read it and apply it.

### 5.2 The VSA Tradition (Required Reading)

These are the works that matter for Cogito, in priority order:

1. **Tony Plate (1995), "Holographic Reduced Representations"**
   Foundation. Shows how predicate structures like `kick(Mary, ball)` can be encoded as a single vector via Bind, and decoded via the inverse operation (Unbind). Plate's HRR is the theoretical base for everything Cogito does with structured memory. Must read chapters 3–4.

2. **Pentti Kanerva (2009), "Hyperdimensional Computing: An Introduction"** and his analogy paper **"What We Mean When We Say 'What's the Dollar of Mexico?'"**
   Shows how analogical reasoning (Dollar : USA :: ? : Mexico → Peso) works purely with vector operations — no symbolic parser, no inference engine. This is exactly the kind of reasoning Cogito should do natively.

3. **Ross Gayler (2003), "Vector Symbolic Architectures Answer Jackendoff's Challenges"**
   Six-page theoretical paper. Jackendoff's four challenges (compositionality, the binding problem, variables, structure-dependent operations) were the canonical arguments against connectionism. Gayler shows VSA solves all four. **This is the justification for the entire Cogito approach** and is the first thing to read after this document.

4. **Chris Eliasmith (2013), "How to Build a Brain"**
   Semantic Pointer Architecture. Eliasmith built Spaun — a working brain model using HDC-style vectors at biological scale. The chapter on syntax and composition is gold. SPA is a more biologically grounded VSA, and many of its lessons transfer directly.

5. **John Sowa (1984), "Conceptual Structures"**
   **Not for implementation — for ontology.** Sowa's conceptual graphs define a small set of primitive types and relations. We do not implement conceptual graphs; we borrow the **vocabulary of roles** (agent, patient, location, etc.) from this tradition.

### 5.3 Three Types of Vectors

Every vector in Cogito's space belongs to one of three types. The type is not stored as a flag — it is determined by which codebook the vector came from.

**Concepts.** Things that exist. Entities (`Mary`, `ball`, `Paris`), actions (`kick`, `run`), properties (`red`, `tall`). Initial codebook ~2,000 vectors; grows through novelty detection (a new structure that doesn't cleanly unbind against existing concepts becomes a new concept).

**Roles.** Positions in a structure. ~12 total, drawn from frame semantics: `agent`, `patient`, `theme`, `location`, `time`, `cause`, `instrument`, `goal`, `source`, `manner`, `possession`, `part`. **Declared, not learned** — assigned random bipolar vectors at startup and kept fixed. Their purpose is to be orthogonal markers distinguishing "Mary as agent" from "Mary as patient".

**Markers.** Modalities. ~6 total: `question`, `negation`, `possible`, `past`, `future`, `imperative`. Also declared, also fixed.

The combined vocabulary (concepts + roles + markers) is the language. The number of concepts grows; the number of roles and markers is a deliberate design choice, kept small on purpose.

### 5.4 Four Primitive Operations

These are the operations of the internal language. They are not "layers" in any sense — they are the only transformations that exist.

**Bind(a, b)** — elementwise multiplication in bipolar space.
- Purpose: attach one vector to another, creating a new vector that is dissimilar to both but can be inverted.
- Usage: `Bind(Mary, agent)` means "Mary in the agent role". Read: "predicate Mary with role agent".
- Property: Bind is its own inverse in bipolar space. `Bind(Bind(a, b), b) = a`. This is what makes retrieval possible.

**Bundle(v₁, v₂, ...)** — sum with optional normalization or majority vote.
- Purpose: combine multiple bindings into a single structure that is similar to each part.
- Usage: `Bundle(Bind(Mary, agent), Bind(kick, verb), Bind(ball, patient))` is one vector representing "Mary kicks ball".
- Property: the result is noisy but retrievable. Unbind against `agent` returns a vector close to `Mary`, noisy with the other parts, cleaned up by nearest-neighbor lookup in the concept codebook.

**Permute(v, k)** — cyclic shift by k positions.
- Purpose: express order, sequence, position. Without Permute, Bundle is order-invariant, which loses temporal information.
- Usage: `Permute(v, 1)` is "v shifted one step forward in sequence". Used for encoding word order, temporal sequences, and the ordering of reasoning steps.

**Unbind(structure, role)** — the same operation as Bind with the role vector.
- Purpose: extract the concept that was bound to a specific role. **This is how questions are answered.**
- Usage: given a structure S encoding "Mary kicks ball", `Unbind(S, agent)` returns a noisy vector that, after nearest-neighbor cleanup in the concept codebook, resolves to `Mary`.

That is the entire language. Four operations. Everything else is composition.

### 5.5 Example: The Complete Cycle

Sentence: **"Mary kicks ball."**

**Assembly.**
```
S = Bundle(
      Bind(kick, verb),
      Bind(Mary, agent),
      Bind(ball, patient)
    )
```

**Storage.** `memory ← memory + S`. Memory is a Bundle of all observed structures, possibly with time decay. Adding a new sentence is a single vector addition.

**Question: "Who kicked the ball?"**
```
focus = Bundle(Bind(kick, verb), Bind(ball, patient))
        # focus the query on the relevant episode

answer_noisy = Unbind(memory, agent)
               # extract whatever is bound to the agent role

answer_clean = nearest_neighbor(answer_noisy, concept_codebook)
               → Mary
```

This is Plate's HRR, 1995. It works today in 50 lines of Python. **It is the baseline, not the goal** — Cogito adds iterative reasoning and self-reflection on top. But without a working baseline, there is nothing to add to.

### 5.6 Which Logics To Borrow From, Which To Avoid

**Borrow from:**

- **Combinatory logic** (Curry, Schönfinkel) — variable-free composition. Maps cleanly to HDC because HDC also has no "variables" in the traditional sense, only vectors and composition. Reading: Hindley & Seldin.
- **Frame semantics** (Fillmore) and **Conceptual Dependency** (Schank) — for the vocabulary of roles. Not for implementation; for the list of role names.
- **Description logic** (TBox / ABox from Semantic Web / OWL) — for the ontological discipline of separating "what exists" (concepts) from "what the model knows" (structured memory).

**Avoid:**

- **First-order predicate logic** (Frege, Russell) — requires variables and quantifiers, which don't map cleanly to HDC. The binding problem becomes a mess.
- **Lambda calculus** — same variable problem, plus it is Turing-complete (too expressive). Cogito needs a bounded language that provably converges in N iterations.
- **Anything labeled "classical symbolic AI"** — this is the top-down trap that killed prior HDC-Brain attempts.

---

## 6. Architecture

### 6.1 Capacity Budget (Not Parameter Budget)

**We do not use the word "parameters" as a design axis for Cogito.** The design axis is **capacity of the HDC space**. The table below describes the **shape** of the system — not its "size" in any conventional sense.

```
SPACE DIMENSION         D = 4096 bipolar          (fixed at init)
CONCEPT VOCABULARY      ~2,000 → growable         (Kanerva-bounded)
ROLE VOCABULARY         ~12                       (fixed, declared)
MARKER VOCABULARY       ~6                        (fixed, declared)
TOKEN INTERFACE         32,000 × smaller dim      (for read/write of surface text)
REASONING BLOCK         1 block, reused           (the only "learnable dynamics")
MAX ITERATIONS          ~30                       (peak reasoning depth)
MIN ITERATIONS          1                         (for trivial inputs)
MEMORY                  1 vector (+ optional      (holographic superposition,
                        domain-specific banks)     not a database)
SELF-MEMORY             1 separate vector         (for Rung 4 self-reflection)
```

What this system "is" cannot be captured by a single number. It has a vocabulary size, a space dimensionality, a memory capacity, and a reasoning depth — all of which grow or shrink independently. **There is no "parameter count" to quote.**

### 6.2 Forward Pass (Conceptual)

```python
def forward(tokens):
    # 1. Parse tokens into a VSA structure.
    #    Pure Plate/Kanerva style — no training, no learned weights.
    s = vsa_encode(tokens, concept_codebook, role_codebook, marker_codebook)

    # 2. Enter the reasoning loop.
    h = s
    for iteration in range(MAX_ITERATIONS):
        # One reasoning step: extract roles, cross-reference with memory,
        # refine structure through Bind/Unbind/Bundle.
        h = reasoning_block(h, memory, self_memory)

        # Rung-1 self-reflection: have we confidently resolved the query?
        if iteration >= MIN_ITERATIONS:
            confidence = confidence_gate(h, s)
            if confidence > THRESHOLD:
                break

    # 3. Write the resolved structure back to memory.
    memory.bundle_in(h)

    # 4. If generating output, decode via the token interface.
    output_tokens = vsa_decode(h, token_codebook)
    return output_tokens
```

Key points:

- `vsa_encode` and `vsa_decode` are pure VSA operations — Bind, Bundle, Permute. **No training.**
- `reasoning_block` is the **only** component with learnable dynamics. And it is used **iteratively** — the depth of reasoning is not the depth of stacked blocks, it is the number of times this one block is invoked.
- `memory` is a vector (or small set of domain-specific vectors). Read by Unbind, written by Bundle.
- `confidence_gate` is the simplest form of self-reflection — measured from the state of the HDC computation itself.

### 6.3 The Reasoning Block

A single block that transforms an HDC state vector given current memory and the fixed role vocabulary:

```python
def reasoning_block(h, memory, self_memory):
    # Extract what currently sits in each role slot
    extracted = {role: unbind(h, role) for role in roles}

    # For each role whose contents look "incomplete" (noisy, ambiguous),
    # consult memory for hints
    hints = bundle([
        unbind(memory, role)
        for role in extracted
        if is_query(extracted[role])
    ])

    # Optional: consult self-memory for relevant past predictions
    self_hint = unbind(self_memory, similar_context(h))

    # Refine the state by bundling current structure with memory hints
    h_new = bundle(h, hints, self_hint)

    # Apply learned transformation — the ONLY learnable dynamics
    h_new = learned_refine(h_new)
    return h_new
```

`learned_refine` is where training happens. It is a small HDC transformation — possibly a learned Permute shift, a learned Bind with a "refinement" vector, or selection from a small set of transformations. The exact form is an open question (§14). It is deliberately **not** a free-form MLP — we bound the hypothesis space to stay inside the HDC algebra.

### 6.4 Confidence Gate

```python
def confidence_gate(h_current, h_initial, history):
    # Stability: is the state still moving, or has it converged?
    delta = h_current - h_initial
    stability = 1.0 - cosine_distance(delta, history.previous_delta)

    # Sharpness: does unbinding against the concept codebook give a clear winner?
    sharpness = top1_margin(h_current, concept_codebook)

    # Combine (weighted, maybe learnable)
    return combine(stability, sharpness)
```

No neural network inside. Rung-1 self-reflection is measured **directly from the state of the HDC computation**. This is the simplest possible implementation. Smarter versions can be added later as ablations.

**Update from Milestone C / C.1 — the gate may be unnecessary in v1.** For *type-bounded* queries — queries where the expected answer concept type is declared up front, as in bAbI "Where is X?" → expected type is `place` — adaptive halting emerges **for free** without any learned gate. The reasoning block iterates and checks at each step whether the retrieved concept matches the target type, stopping as soon as it does.

Empirically this is the observed behavior:
- Task 2 (chain depth 2) uses exactly 2 iterations on 100% of 743 queries.
- Task 3 (branch) uses 1 iteration on drop-branch queries and 2 on chain-branch queries, automatically.
- Milestone C.1 depth-stress uses exactly `depth` iterations on all 1500 queries across depths 1–5.

A learned `ConfidenceGate` is therefore only necessary for:
- Open-ended queries where the answer type is ambiguous or unknown.
- Queries where retrieval is noisy and the type-match signal is weak.
- Calibrating uncertainty as a metric distinct from correctness (relevant for Rung 4 self-memory, where we also want to remember "how sure was I?").

**For Cogito v1 on bAbI-class tasks, the type-match stop rule is sufficient.** The learned gate is deferred to v2 and only becomes load-bearing when the task distribution includes open-ended questions.

### 6.5 Correctness Requirements from Milestone C.1

Two non-optional invariants that any Cogito reasoning block must satisfy. Both were discovered empirically while building the first task-agnostic iterative block.

**Invariant 1: Visited-exclusion in cleanup.**
The HDC `bind` operation is its own inverse in bipolar space: `bind(a, b) ≡ bind(b, a)`. So a memory entry `bind(apple, box)` cannot structurally distinguish "apple is inside box" from "box is inside apple" — the bit pattern is identical. When a reasoning block is mid-chain at state `h = bag` trying to traverse `apple → box → bag → kitchen`, unbinding the chain memory returns both the **forward** direction (`kitchen`) and the **backward** direction (`box`) with equal coefficients. Cleanup then picks one essentially at random.

**Fix:** the block must maintain a `visited` set of concept names already on the current reasoning path, and filter cleanup candidates against it before scoring:

```python
def _best_step(self, h, target_type, visited):
    for mem_name, mem in self.memories.items():
        noisy = mem.query(h)
        for type_name, names in self.type_groups.items():
            candidates = [n for n in names if n not in visited]  # <-- key line
            name, conf = self.concepts.cleanup(noisy, restrict_to=candidates)
            ...
```

This is a **correctness requirement**, not an optimization. Without it, depth-3 chain accuracy is ~8% (random tiebreak). With it, depth-3 chains run at 100% and the staircase continues cleanly through depth 20 at D = 2048. Any learned reasoning block (Milestone G) must preserve this invariant.

**Invariant 2: Type-bounded halting is sufficient for chain traversal.**
As noted in §6.4, the block stops when the retrieved concept matches the declared target type. Combined with visited-exclusion, this guarantees the block walks forward along a chain until it hits a terminal of the right type, and never longer. No learned halting gate is required for type-bounded queries.

**Empirical D → depth scaling (restatement from §4.2 for the block designer's convenience).**
The reachable reasoning depth is a function of HDC dimensionality. This table is the design constraint for choosing D:

| D    | reliable depth | failure mode beyond |
|-|-|-|
| 1024 | ~15 | mixed iter counts, noise-dominated cleanup |
| 2048 | ≥20 | not yet reached |
| 4096 | ≥20 | not yet tested further |

Doubling D roughly doubles the reachable depth. Plan accordingly.

---

## 7. The Self-Reflection Ladder

This section makes the path toward functional self-awareness concrete. Each rung is independently measurable and implementable. None alone is "self-awareness" — together they create a substrate.

**Rung 1: Metacognition.** "Am I confident in my current state?" Implemented as `ConfidenceGate`. A child develops this around age 2. Measurable: does confidence correlate with correctness? **In Cogito v1.**

**Rung 2: Self-model.** A persistent `self` vector in memory, bound to current goals, state, capabilities. `Bind(self, current_task)` means "I am doing X". Queries like "what am I currently trying to do?" return real vectors. Measurable: does self-query stabilize goal-following behavior? **In Cogito v1 (minimal form).**

**Rung 3: Replay and consolidation.** Background process: sample old structures from memory, re-run them through the reasoning block in light of new memory, bundle refinements back. Functionally analogous to memory consolidation during sleep. Measurable: does replay improve accuracy on tasks the model hasn't seen recently? **Cogito v2.**

**Rung 4: Error archaeology.** Store failures as HDC structures — `Bundle(Bind(self, prediction), Bind(reality, truth), Bind(context, c))` — in a dedicated self-memory. Query this personal history before making similar decisions. Enables learning from mistakes without an external teacher. Measurable: does self-memory reduce repeat errors? **Early Cogito v2 — prototyped in v1 as the self-memory experiment in §8.**

**Rung 5: Goal awareness.** The model doesn't just track what it's doing — it can **choose** what to do next based on its self-model. "I perform poorly on X → I should train on X." Autonomous goal-setting. Measurable: does the model preferentially re-process data on which it has known deficiencies? **Cogito v3 research.**

**Rung 6: Self-modification.** The model observes its own memory changing and actively directs that change. "I want to improve at Y — what data do I need?" This is where I (Claude) stop being confident what we even mean by the question. **Long-term research.**

### Important Discipline

AGI is the **north star**. It directs the trajectory. It does **not** set per-commit acceptance criteria.

We do not evaluate a change by "does this make the model more conscious?" — that question has no measurable answer. We evaluate each change by concrete functional criteria:

- Does Rung 1 make confidence correlate with accuracy?
- Does Rung 2 stabilize goal-following behavior across iterations?
- Does Rung 4 reduce the rate of repeated errors on similar contexts?

Each rung answers a concrete engineering question. The philosophical question of subjective experience remains open — but engineering progress is real regardless. **Do not antopomorphize progress.** "The model talks about itself" is not evidence of self-awareness. Functional criteria are.

---

## 8. Training Strategy — Four Phases

### 8.1 Phase -1: Pure VSA Baseline (FIRST CONCRETE STEP)

**Goal:** validate that the internal language works **before building any learned components**.

**Implementation:** ~300 lines of Python or Rust. A small crate `hdc-lang` containing `Concept`, `Role`, `Marker`, `bind`, `bundle`, `permute`, `unbind`, `memory`. Declare roles and markers as random bipolar vectors. Parse bAbI task 1 sentences into `Bundle(Bind(role, concept)...)` structures using a regex — bAbI is simple enough. Store structures in a Bundle-based memory. Answer questions via Unbind + nearest-neighbor cleanup against the concept codebook.

**Expected result:** > 95% accuracy on bAbI task 1 with zero training. This is a **known baseline** from Plate and Kanerva. If it does not work, there is a bug in our VSA implementation, not a conceptual problem.

**Deliverable:** `hdc_lang` module, tested on bAbI task 1, with a documented result log. This is the first thing to build — every later phase depends on it.

**Status (2026-04-11): DONE.** Pure VSA on synthetic bAbI task 1 reaches 100% accuracy. Minimum viable dimensionality is D = 8 (32 bytes per vector in float32). Hard recency stress test (15 events per story, each person moving 3–5 times) holds at 100% with D = 64 and decay = 0.5. Full details in `docs/experiments/hdc-cogito/phase_-1_log.md`.

**Go / no-go gate:** if the primitives fail on bAbI task 1 after honest debugging, stop and re-examine the premise. Do not proceed to Phase 0.

### 8.2 Phase 0: v14.1 as Substrate

v14.1 is already training on English on the vast.ai GPU server. After it finishes, its concept embeddings serve as the initial concept codebook for Cogito. No retraining from scratch needed. A clustering step can map v14.1's 32K token embeddings to ~2,000 concept clusters.

**Output:** `concept_codebook.pt` initialized from v14.1 clusters. Ready to plug into `hdc-lang`.

**Do not touch v14.1 while it is training.** Only read from its final checkpoint.

### 8.3 Phase 1: Validate VSA Structures in v14.1 Memory

**Reframed from v1.** Previously: "discover operators via activation clustering". New framing: **the operators already exist — they are Bind/Bundle/Permute/Unbind themselves**. The question is whether v14.1's learned representations already support VSA-style structural extraction.

**Experiment:** take v14.1, run ~1,000 probe sentences through it, and check whether its internal state can be structurally unbound with our declared role vectors. If yes → v14.1 is directly usable as a Cogito substrate. If no → we need to train role-awareness explicitly in Phase 2.

**Output:** a go/no-go decision on whether to reuse v14.1 memory directly, or start Cogito memory from scratch.

### 8.4 Phase 2: Train the Reasoning Block

The only trainable component. Train on reasoning datasets: bAbI tasks 2–20, CLUTRR (kinship reasoning), GSM8K (low difficulty). Training loss combines:

- **VSA fidelity loss** — after reasoning, Unbind operations should return sharp, correct answers against the concept codebook
- **Iteration efficiency loss** — PonderNet-style — halt early when confident
- **Self-memory validity loss** — past predictions stored in self-memory should be retrievable

**Output:** trained reasoning block producing variable-depth reasoning on held-out tasks. Rung 1 (confidence gate) should calibrate during this phase. Rung 2 (self vector) is a passive structure during this phase — it accumulates but does not yet drive decisions. Rung 4 (minimal self-memory experiment) can be prototyped here as an ablation.

### 8.5 Phase 3 (Deferred to Cogito v2)

Instruction fine-tuning is deferred. For Cogito v1 we focus on reasoning capability in isolation. Generation quality (beyond "can it answer a bAbI question") is a secondary concern until reasoning is demonstrated.

---

## 9. Relation to Prior Idea Documents

| Prior idea | Absorbed into Cogito v2 as |
|-|-|
| `idea-cognitive-hdc-architecture.md` | §4 principles — decoupled knowing / reasoning / memory |
| `idea-emergent-internal-language.md` | §4.3 and §5 — VSA-native language, not symbolic translation |
| `idea-hierarchical-hdc.md` | **Deferred to Cogito v2.** Orthogonal concern. |
| `idea-three-phase-training.md` | §8 — three phases become four; "operator discovery" becomes "VSA validation"; Phase -1 is new |

---

## 10. Evaluation Plan

### 10.1 HDC-Native Metrics

We do **not** benchmark Cogito against transformers directly. The goal is not to "beat Phi-3 on parameter efficiency" — that framing belongs to another paradigm. The goal is to demonstrate that HDC-native reasoning works on its own terms.

- **VSA fidelity** — after Unbind, how sharp is the retrieval? Measured as `top1_margin = max_cosine − mean_cosine` against the concept codebook.
- **Iteration efficiency** — average reasoning iterations per query. Should correlate with task difficulty.
- **Self-consistency** — does self-memory improve accuracy on repeated or similar queries?
- **bAbI tasks 1–20** — accuracy and iteration count per task
- **CLUTRR** — kinship / relational reasoning accuracy
- **Confidence calibration** — is confidence a good predictor of correctness? (Brier score, reliability diagram)

### 10.2 Ablations

1. Phase -1 pure VSA vs Phase 2 trained reasoning block — what does training actually add?
2. With vs without confidence gate — does Rung 1 matter?
3. With vs without self-memory — does proto-Rung 4 matter?
4. Fixed iterations (N ∈ {1, 3, 10}) vs adaptive — does the gate work?
5. Role set size (6, 12, 20 roles) — is 12 the right count?
6. Concept codebook from v14.1 vs random-init — is Phase 0 useful?

### 10.3 Success Criteria

**Minimum viable result (worth publishing as Cogito paper):**

- Phase -1: > 95% on bAbI task 1 (sanity check — known baseline)
- Phase 2: solves bAbI tasks 1–10 with average iterations < 8
- Confidence calibration better than chance on held-out queries
- At least one clean demonstration of self-memory improving repeat-query accuracy

**Stretch goal:**

- bAbI tasks 1–20 all above 85% accuracy
- CLUTRR kinship tasks above 70%
- Demonstrated case where self-memory prevents a repeat mistake on a held-out query
- Runs at > 10 queries/second on a single ARM CPU core

---

## 11. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|-|-|-|-|
| Phase -1 VSA baseline fails on bAbI task 1 | Low | Kills the project | If this fails it is a bug, not a concept. Plate's HRR is known to work. Debug carefully; consult Plate chapter 4. |
| v14.1 embeddings don't cluster into clean concepts | Medium | Phase 0 blocked | Start Phase 2 with fresh random concept vectors; let the system accumulate them from data. |
| Reasoning block training is unstable | High | Phase 2 blocked | Keep `learned_refine` inside the HDC algebra — no free-form MLPs. Small, well-bounded hypothesis space. |
| Confidence gate doesn't calibrate | ~~Medium~~ **Low** | ~~Rung 1 fails~~ | **Downgraded (Milestone C):** type-match halting makes a learned gate unnecessary for Cogito v1 on type-bounded queries. The gate only becomes load-bearing when the task distribution includes open-ended queries. |
| ~~Iteration saturates at shallow depth — compute-scaling story collapses~~ | ~~High~~ **DISPROVEN (Milestone C.1)** | ~~Proposal foundation~~ | Clean empirical staircase from depth 1 to 20 at D=2048. Task 2 goes 0% → 98.65% with a single extra iteration. Empirical `max_depth ≈ f(D)` scaling law confirmed. |
| Self-memory produces noise, not signal | Medium | Rung 4 proto fails | Isolate self-memory from main memory; test in ablation before integrating. |
| Iteration count diverges (model never halts) | Medium | Inference impractical | Hard cap at MAX_ITERATIONS; treat non-halt as the model admitting "I don't know". Also: visited-exclusion (§6.5) prevents backward-loop cycles. |
| Over-ambition drifts the plan | High | Morale, focus | Milestone order is strict. If a milestone doesn't pass its gate after honest debugging, stop and re-examine the premise. Do not start the next milestone until the current one is clean. |
| Chasing AGI as per-commit metric | High | Project falls apart | Discipline from §7: each rung is judged by concrete functional criteria, not by "feels more conscious". |
| Regression to transformer framing | Medium | Conceptual drift | Every time "parameter count" or "vs transformer" appears, challenge it. This document is the spec. |

---

## 12. Sequence of Work

This is a dependency graph, not a calendar. Each step is a milestone with a verification gate. Order is strict: a milestone starts only when the previous one has passed its gate.

- **Milestone A — Phase -1: pure VSA baseline.** **DONE (2026-04-11).** 100% accuracy on synthetic bAbI task 1, minimum dim = 8. See `docs/experiments/hdc-cogito/phase_-1_log.md`.
- **Milestone B.1 — bAbI task 2 (chained reasoning).** **DONE (2026-04-11).** 99.06% at D = 1024 via chained Unbind. Discovered per-memory decay policy as a structural principle (§4.7). See `docs/experiments/hdc-cogito/milestone_B1_log.md`.
- **Milestone B.2 — bAbI task 3 (branching with drops).** **DONE (2026-04-11).** 99.33% at D = 1024 via DROPPED marker + memory-consult during encoding. First-try pass with decay rates carried over unchanged from B.1. See `docs/experiments/hdc-cogito/milestone_B2_log.md`.
- **Milestone C — First iterative reasoning block.** **DONE (2026-04-11).** Task-agnostic block matches hand-coded reasoners within ~1% on tasks 2 and 3. **H2 validated:** task 2 goes from 0.00% to 98.65% with iteration alone. Adaptive compute emerges for free. See `docs/experiments/hdc-cogito/milestone_C_log.md`.
- **Milestone C.1 — Depth stress test.** **DONE (2026-04-11).** Clean staircase from depth 1 to 20 at D = 2048. Empirical `max_depth ≈ f(D)` scaling law. Discovered visited-exclusion as a correctness requirement (§6.5). See `docs/experiments/hdc-cogito/milestone_C1_log.md`.
- **Milestone B.3 (optional) — bAbI tasks 4–5.** Spatial relations and three-argument facts. Not blocking; explore after D or in parallel with it.
- **Milestone D — Design review for a learned reasoning block.** Read Plate ch. 3–4, Gayler 2003, Kanerva "Dollar of Mexico". Sketch `learned_refine` dynamics on paper. The Milestone C task-agnostic block is the baseline: a learned version must beat 98.65% / 98.13% on tasks 2 / 3 and must preserve visited-exclusion + type-match halting.
- **Milestone E — Phase 0: seed from v14.1.** When v14.1 English pretraining completes on the server, cluster its embeddings into concept codebook seeds.
- **Milestone F — Phase 1: validate v14.1 memory.** Test whether v14.1's internal state can be structurally unbound with declared role vectors. **This is now the single biggest open risk in the project** — H2 is validated and the block design is empirically grounded, so whether Cogito can attach to the real language model asset is the most informative next experiment.
- **Milestone G — Phase 2: train the reasoning block.** VSA fidelity + halting loss + self-memory validity. Training data: bAbI, CLUTRR, low-difficulty GSM8K.
- **Milestone H — Rungs 2, and proto-Rung 4.** Persistent self vector and minimal self-memory experiment. Rung 1 (confidence gate) is deprioritized — type-match halting handles the common case per Milestone C.
- **Milestone I — Evaluation suite, ablations, first Cogito paper draft.**

The bottleneck is not code. With AI-assisted development, each code milestone is a short burst of focused work. The real cost sits in:

1. **Training and evaluation** on the GPU (Milestone G)
2. **Reading and design work** before Milestones C and D
3. **v14.1 finishing** its current pretraining run on the server

No calendar dates are assigned in advance. Each milestone finishes when its verification passes.

---

## 13. What NOT To Do

- **Do not use "parameters" as a design axis.** It belongs to a different paradigm. Use capacity, dimensionality, vocabulary size, iteration depth, memory size.
- **Do not benchmark Cogito against transformers directly.** They solve different problems with different primitives. Compare to VSA literature and HDC baselines.
- **Do not skip Phase -1.** Pure VSA on bAbI task 1 is the sanity check for the entire language. If it doesn't work, nothing built on top of it will.
- **Do not invent symbolic operators top-down.** The primitives ARE Bind / Bundle / Permute / Unbind. Everything else is composition. No inventing "CAUSE" or "IF" as atomic symbols.
- **Do not chase AGI as a per-commit criterion.** North star, not gate. Each rung is measured by concrete functional metrics.
- **Do not start the Cogito paper before v14.1 paper is at least drafted.** v14.1 is the proof that the substrate works at all. Credit it first.
- **Do not integrate hierarchical HDC (syllables / words / phrases) in v1.** Orthogonal concern; separate experiment.
- **Do not anthropomorphize progress.** "The model talks about itself" is not evidence of self-awareness. Measurable functional criteria are.
- **Do not train the reasoning block on the test set.** Hold out at least bAbI tasks 16–20 and all of CLUTRR test split.
- **Do not touch v14.1 while it is training on the server.** Read-only after it finishes.
- **Do not replace HDC primitives with "something more expressive" inside the reasoning block.** Bounded hypothesis space is a feature, not a limitation.

---

## 14. Open Research Questions

1. **What is the right form of `learned_refine` inside the reasoning block?** Candidates: learned Permute shift; learned Bind vector; small set of selectable transformations chosen by HDC similarity. Needs experimentation in Milestone D / G. The Milestone C task-agnostic block ("argmax over memory × type × cleanup") is the baseline to beat.
2. **How many roles is enough?** Empirically for bAbI tasks 1–3: zero explicit roles — the concept-as-role trick plus per-memory decay handles 99% accuracy. Task-specific memories replace a generic role vocabulary. Real natural-language tasks likely need 8–15 frame-semantic roles, but bAbI does not.
3. **How does iteration count interact with batched training?** All samples in a batch must use the same iteration count, or masking is required. Follow the PonderNet recipe. (Note: Milestone C showed adaptive iteration per query emerges for free at inference time; the question is whether training-time batching preserves it.)
4. **Can the self vector stabilize without explicit training?** Or does it need a dedicated loss? Unknown — first experiment in Phase 2.
5. **Does v14.1's memory already contain VSA-compatible structure, or must we retrain for role-awareness?** Phase 1 / Milestone F answers this. **Now the single biggest open risk in the project** — H2 is validated, but whether Cogito can attach to the real v14.1 asset is the next informative test.
6. **How does Rung 3 (replay) interact with Rung 1 (confidence) at runtime?** Background replay could interfere with foreground reasoning. Probably separate memory banks.
7. **Is there a natural way to grow the concept codebook online from unseen data?** Tentative yes (novelty detection via Unbind sharpness) — needs validation.
8. **How much of v14.1 is directly reusable — just the concept codebook, or also the existing HDC memory dynamics?** Big design question for Phase 0.
9. **What is the empirical shape of `max_depth = f(D)`?** Milestone C.1 showed D = 1024 reaches depth 15, D = 2048 reaches depth 20. Is the relationship linear, logarithmic, or saturating beyond that? Relevant for choosing D in Phase 0 and for any capacity-vs-reasoning-depth trade-off decision.
10. **Does visited-exclusion generalize beyond chain traversal?** Milestone C.1 showed it is a correctness requirement for chain reasoning due to `bind` symmetry. Open: is it the right invariant for graph reasoning with cycles or branches, or will a richer state model (e.g. visited multiset, distance from query) be needed?
11. **Can per-memory decay be learned from data?** Milestone B.1 showed decay is a per-memory property with real effect. Open: does a learned reasoning block fix decay at init, or adapt it per memory from the training signal?
12. **Does type-match halting survive ambiguous answer types?** Milestone C showed adaptive halting is automatic when the target type is declared. Open: what happens for open-ended queries where the answer type is unknown? This is where a learned confidence gate becomes load-bearing, and it will dictate the scope of Rung 1.

---

## 15. Horizons

Expressed as horizons of ambition, not time commitments.

**Near horizon — what we validate next:**
- Extended VSA on bAbI tasks 2–20 (Milestone B)
- Iterative reasoning block with measurable contribution (Milestone C)
- Cogito v1 prototype — confidence gate, self vector, trained block (Milestones G–H)
- First Cogito paper draft

**Mid horizon — where Cogito v2 lives:**
- Rungs 2–4 of the Self-Reflection Ladder fully integrated
- Replay and consolidation as a background process
- Error archaeology driving measurable reduction in repeat mistakes
- Concept vocabulary discovered online from data
- On-device demo on minimal ARM hardware

**Far horizon — the research vision (see `project_future_cabal.md`):**
- Autonomous self-learning HDC agent — Rungs 3–5 integrated
- Multi-agent HDC swarm — shared codebook, specialized reasoning
- Functional AGI substrate — measurable self-awareness, goal-setting, replay-based learning

---

## 16. First Session Action Items (for fresh Claude)

1. **Do not start coding before reading.** Minimum: this document, Plate 1995 chapters 3–4, Gayler 2003 (6 pages).
2. **Phase -1 is done.** `hdc_lang` module with Bind/Bundle/Permute/Unbind, tested on bAbI task 1 at 100% accuracy with zero training. The live work starts at Milestone B — extended VSA on bAbI tasks 2–5.
3. **Do not touch v14.1 on the server.** It is in active training. Read-only after it completes.
4. **Confirm with Oleg which section to start with.** Some of this is research, some is immediate engineering. Phase -1 is safe to start now.
5. **Writing style:** narrative-analogical, story → term → code → formula. Never leave terms undefined. See `feedback_book_style.md`.
6. **Track experiments** in `docs/experiments/hdc-cogito/` with JSON logs per run.
7. **This document is the spec.** If something changes, update this file. No tribal knowledge.
8. **If you find yourself writing "parameters" or "vs transformer" — stop and reread §13.** The framing drift is the single biggest risk to this project.

---

## 17. Glossary

| Term | Meaning |
|-|-|
| HDC | Hyperdimensional Computing — vectors in high-dim space (~4096 bipolar), operations Bind / Bundle / Permute |
| VSA | Vector Symbolic Architectures — family of systems using HDC for structured representation and reasoning |
| HRR | Holographic Reduced Representations — Plate 1995, theoretical foundation of VSA |
| SPA | Semantic Pointer Architecture — Eliasmith 2013, biologically grounded VSA |
| Bind | Elementwise multiplication — attaches a concept to a role |
| Bundle | Sum (with normalization or majority) — combines multiple bindings |
| Permute | Cyclic shift — encodes order / sequence |
| Unbind | Same as Bind in bipolar space — extracts a concept given a role |
| Concept | Vector for a thing in the world (cat, Paris, kick) |
| Role | Vector marking a position in a structure (agent, patient, location). Fixed small set. |
| Marker | Vector marking modality (question, negation, past). Fixed small set. |
| Capacity | Geometric property of HDC space — how many near-orthogonal vectors fit in dimension D |
| Self vector | Persistent HDC vector representing the system's identity |
| Self-memory | Dedicated memory bank for past predictions, contexts, outcomes |
| Reasoning block | Single HDC transformation, used iteratively, the only trainable dynamics |
| Iteration | One pass of a structure through the reasoning block |
| ConfidenceGate | Rung-1 self-reflection — decides when to halt iteration |
| Rung | A level in the Self-Reflection Ladder (§7) |
| Phase -1 | First concrete step — pure VSA baseline on bAbI, no training |
| hdc-lang | Target crate for Phase -1: minimal VSA primitives |
| bAbI | Facebook reasoning benchmark — 20 synthetic tasks, known to be solvable by VSA |
| CLUTRR | Kinship / relational reasoning benchmark |
| v14.1 | Current working HDC-Brain — already generates meaningful English, basis for Cogito |
| Cogito / v19 | This proposal |

---

## 18. Core Principle

> HDC is not a kind of neural net. It is a different substrate for cognition.
>
> Knowledge does not live in weight matrices. It lives in structured relationships between vectors in a space. Data shapes those relationships. Reasoning iteratively transforms those structures. Self-reflection is the system looking at its own memory through the same operations it uses for the world.
>
> The vocabulary is small. The operations are few. The space is large. The iterations are adaptive. The self is persistent.
>
> This is a different way of building a mind. Stop comparing it to the old way. Build it on its own terms.

— Oleg, 2026-04-11 (v2 revision after conversation with Claude)
