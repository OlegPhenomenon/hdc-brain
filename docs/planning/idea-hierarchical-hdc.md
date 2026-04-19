# Idea: Hierarchical HDC with Syllable/Word/Phrase Trees

**Date**: 2026-04-06
**Status**: Raw brainstorm, not validated
**Author**: Oleg (ideation), Claude (structuring)

## Core Idea

Instead of flat token sequence, encode text as hierarchy:
- **Syllable level**: atomic units (~15K syllables in English)
- **Word level**: composed from syllables via HDC binding
- **Phrase/sentence level**: composed from words

Each level lives in the same high-dimensional space, enabling cross-level similarity.

## Motivation

1. **Trigrams are too many** — combinatorial explosion
2. **Words have typo problem** — but syllables give fuzzy matching for free
3. **Parallel processing** — different levels can be processed simultaneously on GPU/CPU cores
4. **"Quantum-like" parallelism** — information flows from multiple angles at once, results interfere

## Example Encoding

```
word("Paris") = bind(pos1 * syllable("Pa"), pos2 * syllable("ris"))
word("Patis") = bind(pos1 * syllable("Pa"), pos2 * syllable("tis"))
# These word vectors are PARTIALLY similar (shared "Pa" syllable)

phrase("capital of France") = bind(pos1 * word("capital"), pos2 * word("of"), pos3 * word("France"))
```

## Parallel Processing Vision

```
Level 1 (GPU threads):  syllable processing   (Pa, ris, ca, pi, tal...)
Level 2 (GPU threads):  word processing        (Paris, capital, France...)
Level 3 (GPU threads):  phrase processing      (capital of France, is Paris)
                    ↕ cross-level interference ↕
              all levels exchange through shared HDC space
```

## Open Questions

1. **Syllable ambiguity**: one syllable → many words. "tion" appears in thousands of words.
   - Is this a feature (shared structure) or a bug (confusion)?
   - Context at word level should disambiguate

2. **Stored sentences = memorized phrases?**
   - If we store full sentences, model becomes a phrase lookup table
   - Human brain also has memorized phrases but can improvise
   - Maybe: store STRUCTURAL patterns (det + adj + noun) separately from content
   - Creativity = combining known structures with new content

3. **Cosine similarity across levels**:
   - How do syllable vectors "find each other" in high-D space?
   - Should syllables within a word be close or orthogonal?
   - Binding makes composed vectors quasi-orthogonal to components — is this desired?

4. **Optimal granularity**:
   - Maybe word-level is the sweet spot (Oleg's intuition)
   - Syllables as substructure for typo resilience only
   - Skip sentence level to avoid memorization trap

## Oleg's Concerns (valid)

- Too many underwater rocks ("podvodnye kamni")
- Sentences in space = memorized phrases, kills creativity
- Syllable distance unclear — how close should they be?
- Maybe word-level only is sufficient

## Related Concepts

- Holographic Reduced Representations (Tony Plate, 1995)
- Recursive HDC binding for tree structures
- Morphological embeddings (FastText uses subword info)
- Human brain: Broca's area (structure) + Wernicke's area (meaning) = parallel processing

## Decision

Parking this idea for now. Revisit after v14.1 training completes.
Possible experiment: word-level HDC with syllable substructure for robustness.
