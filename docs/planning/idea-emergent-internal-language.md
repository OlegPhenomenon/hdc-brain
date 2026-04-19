# Idea: Emergent Internal Language — Grow Operators, Don't Hardcode

**Date**: 2026-04-06
**Status**: Key research direction, not yet tested
**Author**: Oleg (ideation + prior experiments), Claude (structuring)

## Problem Statement

Oleg previously attempted to build an internal language from **top-down**:
defined a set of logical operators (IF, AND, CAUSE, HAS, etc.)
and tried to make the model operate with them.

**Result**: didn't work. Model couldn't use operators it had no grounding for.

## Why Top-Down Fails

Like giving a baby a logic textbook. The model has no EXPERIENCE
to attach operators to. CAUSE(X,Y) is just another token — meaningless
without having "felt" causation through data patterns.

## Human Analogy: Bottom-Up Learning

```
Age 0-6mo:  ball falls → pain (causality through BODY)
Age 6-12mo: ball ALWAYS falls down (generalization from repetition)
Age 1-2yr:  if I push → it falls (prediction, proto-IF-THEN)
Age 2-3yr:  "ball fell BECAUSE I pushed" (language for existing concept)
```

The operator CAUSE was never hardcoded. It GREW from thousands
of experiences of causality. Logic is the top floor, not the foundation.

## Proposed Approach: Bottom-Up Operator Discovery

### Phase 1: Train base model (what we're doing now with v14.1)
- Model learns text patterns: "X is Y", "X causes Y", "X leads to Y"
- These patterns are implicit in the weights
- No explicit operators

### Phase 2: Activation archaeology
- Freeze trained weights
- Run thousands of diverse sentences through the model
- Collect internal activations (HDC vectors at each layer)
- These activations encode HOW the model processes different structures

### Phase 3: Cluster and discover operators
```
Cluster A: activations for "X causes Y", "X leads to Y", "because of X, Y"
  → This cluster IS the concept of CAUSATION (emerged, not defined)

Cluster B: activations for "X is a Y", "X is type of Y", "X belongs to Y"  
  → This cluster IS the concept of CATEGORIZATION

Cluster C: activations for "X has Y", "X contains Y", "Y is part of X"
  → This cluster IS the concept of POSSESSION/CONTAINMENT
```

### Phase 4: Extract operators into separate space
- Each cluster becomes a named operator in a dedicated LOGIC space
- Operator = centroid HDC vector of its cluster
- New sentences can be projected onto operators to "parse" their logic

### Phase 5: Reasoning with discovered operators
```
Input: "The capital of France is Paris"
Activation → matches CATEGORIZATION operator
Parsed: IS_TYPE(Paris, capital) AND BELONGS_TO(Paris, France)
Stored in structured spaces, not just as text pattern
```

## Key Principle

**Don't design logic — mine it from a trained model.**

Like evolution: DNA wasn't designed. It crystallized from chemical chaos.
Structures that could replicate survived. Similarly:
- Train model on raw text (chaos of patterns)
- Patterns that repeat across many contexts = logical primitives
- Extract them = discovered operators

Archaeologist, not engineer.

## Connection to Multiple Spaces Architecture

Once operators are discovered, they enable the cognitive architecture:

```
ORCHESTRATOR (large space)
  ├── LOGIC SPACE (discovered operators: CAUSE, HAS, IS_TYPE...)
  ├── ENTITY SPACES (cities, people, concepts — auto-clustered)  
  └── LANGUAGE SPACE (words, grammar, generation)

Query flow:
  "столица Франции" 
    → LANGUAGE: parse structure
    → LOGIC: IS_TYPE(?, CAPITAL) AND BELONGS_TO(?, France)
    → ENTITIES: search CITIES near France+capital
    → LOGIC: result = Paris
    → LANGUAGE: generate "Париж"
```

## Advantages Over Top-Down

| Top-Down (failed) | Bottom-Up (proposed) |
|-------------------|---------------------|
| Operators defined by human | Operators discovered from data |
| Model doesn't understand them | Model already uses them implicitly |
| Fixed set, can't grow | New operators emerge with more data |
| Language-dependent | Language-independent (mentalese!) |
| Expert system vibes (1980s) | Emergent intelligence vibes |

## Experiment Checklist (future)

- [ ] Complete v14.1 training (current)
- [ ] Build activation collection pipeline
- [ ] Run diverse English sentences (~100K) through frozen model
- [ ] Collect layer-wise HDC activations
- [ ] Apply clustering (k-means or spectral on cosine similarity)
- [ ] Analyze clusters — do they correspond to linguistic categories?
- [ ] If yes → extract as operators, build logic space
- [ ] Test: can operators generalize to unseen sentence structures?

## Prior Art (related research)

- Probing classifiers (Belinkov et al.) — finding what NNs learn internally
- Concept bottleneck models — explicit concept layer
- Sparse autoencoders on LLM activations (Anthropic, 2024) — finding interpretable features
- Jerry Fodor, "Language of Thought" (1975) — theoretical foundation

## Oleg's Position

"I'm not giving up on internal language idea."
Previous attempts: top-down with explicit operators → failed.
New direction: bottom-up, grow operators from trained model.
This is a core long-term research conviction, not a quick experiment.
