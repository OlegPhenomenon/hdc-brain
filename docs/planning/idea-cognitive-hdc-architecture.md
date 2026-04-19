# Idea: Cognitive HDC Architecture — Multiple Spaces + Reasoning

**Date**: 2026-04-06
**Status**: Raw brainstorm, promising direction
**Author**: Oleg (ideation), Claude (structuring)

## Core Insight

Language model should NOT be a single vector space with statistics.
It should be a SYSTEM of interconnected spaces with reasoning.

## Architecture: Multiple Intertwined Vector Spaces

Not one database — several, each for different entity types:

```
SPACE: CITIES         SPACE: COUNTRIES      SPACE: CONCEPTS
├── Paris             ├── France            ├── capital (→ type: city)
├── London            ├── England           ├── river (→ type: geography)  
├── Berlin            ├── Germany           ├── president (→ type: person)
├── Tokyo             ├── Japan             ├── population (→ type: number)
└── ...               └── ...              └── ...
```

Spaces are INTERTWINED — cross-links between them:
- Paris ↔ France (capital_of relation)
- Paris ↔ Seine (located_on relation)
- France ↔ Europe (part_of relation)

## Two-Phase Learning

### Phase 1: Structural Learning
Learn language PATTERNS first:
- "The X of Y is Z" — structural template
- "X is located in Y" — structural template
- Subject-Verb-Object patterns
- Question patterns → answer patterns

### Phase 2: Semantic Learning  
Learn MEANING of words through abstractions:
- "capital" → abstract concept → implies TYPE: city
- "president" → abstract concept → implies TYPE: person
- "population" → abstract concept → implies TYPE: number

This is NOT statistics ("capital" often appears near city names).
This is UNDERSTANDING ("capital" MEANS "main city of a country").

## Reasoning Chain Example

Input: "What is the capital of France?"

```
Step 1 (Parse):     "capital of France" — structure: "X of Y"
Step 2 (Concepts):  "capital" → abstract type → CITY
Step 3 (Locate):    "France" → found in COUNTRIES space
Step 4 (Search):    In CITIES space, find nearest to (France + capital_of)
Step 5 (Result):    Paris — closest city linked to France with capital_of relation
Step 6 (Template):  Use structure "The X of Y is Z" → "The capital of France is Paris"
```

## Logic Example: Russian Riddle

"А и Б сидели на трубе. А упала, Б пропала. Кто остался на трубе?"
Answer: "И" (the conjunction "and" is also a letter/entity sitting on the pipe)

This requires:
1. Parse sentence structure
2. Identify entities: А, И, Б (not just А and Б!)
3. Track state changes: А → fell, Б → disappeared
4. Deduce: И was not mentioned as leaving
5. Understand wordplay: "и" is both conjunction AND entity

Statistical model: impossible at small scale.
Reasoning system: solvable with proper logic.

## Key Differences from Transformers

| Aspect | Transformer (GPT) | Cognitive HDC |
|--------|-------------------|---------------|
| Knowledge | Compressed in weights | Explicit in typed spaces |
| Reasoning | Emergent (hope it learns) | Explicit chain of steps |
| "Understanding" | Statistical correlation | Categorical + relational |
| Creativity | Interpolation in latent space | Combine structures + fill with content |
| Scalability | More data = better | Better organization = better |

## Open Questions

1. How to learn the category system? (supervised? self-discovered?)
2. How to represent cross-space relations in HDC?
3. How does reasoning chain work in binary vectors?
4. How to handle ambiguity (same word, different spaces)?
5. How to go from this architecture to actual text generation?

## Oleg's Intuition

"Logic should work differently — like reasoning, deduction"
Not: pattern match → output
But: understand → categorize → search → reason → output

## Relation to Previous Ideas

- Builds on hierarchical HDC idea (same session)
- Connects to v18 worm architecture (cascading reasoning)
- Connects to future_cabal.md (autonomous agent with curiosity)
- "Quantum parallelism" = multiple spaces processing simultaneously

## Additional Ideas (2026-04-06, later in session)

### Orchestrator Space
- Multiple mini-spaces (cities, people, actions, properties...)
- One LARGE orchestrator space that routes queries between them
- Mini-spaces can be spawned dynamically as needed

### Internal Language (Mentalese / Language of Thought)
- Oleg's conviction: humans have a fundamental pre-linguistic "language"
- Not words, but logical primitives (like DNA, or like transistor logic)
- Related to Jerry Fodor's "Language of Thought" hypothesis (1975)
- Evidence: deaf children without any language still reason logically
- Previous experiments (v15-v18) tried internal logic language — didn't work yet
- Oleg: "I'm not giving up on this idea"

### Three-Layer Architecture Vision
```
Layer 3 (outer):   HUMAN LANGUAGE (English, Russian)
                     ↕ encode/decode
Layer 2 (middle):  ENTITY SPACES (cities, people, concepts...)
                     ↕ 
Layer 1 (core):    INTERNAL LANGUAGE (logical primitives)
                   CAUSE(X,Y), HAS(X,Y), IS_TYPE(X,Y)
                   BEFORE/AFTER, PART_OF, LEADS_TO...
```

### Chicken-and-Egg Problem
- If primitives are hardcoded → expert system (1980s, doesn't scale)
- If learned from text → need language first (circular)
- Possible solution: grow primitives from EXPERIENCE, not text
  (like a child learns physics before words)
- This is closer to AGI than LM

## Status

Parking for deep exploration after v14.1 training completes.
This may be the foundation for v15 or v16 architecture.
Internal language idea is a long-term research direction — Oleg's core conviction.
