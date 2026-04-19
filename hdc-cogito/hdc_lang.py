"""
hdc_lang — minimal VSA primitives for Phase -1 of HDC-Cogito.

Plate-style Holographic Reduced Representations with bipolar vectors.
No training, no learned weights. Everything the model "does" happens
through four operations: Bind, Bundle, Permute, Unbind.

The whole point of this file: if this doesn't work on bAbI task 1,
the entire HDC-Cogito premise is wrong and we stop.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


# ----- Core vector operations ---------------------------------------------

def random_bipolar(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a bipolar vector from {-1, +1}^dim."""
    return rng.choice([-1.0, 1.0], size=dim).astype(np.float32)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Bind: elementwise multiplication.

    In bipolar space, bind is its own inverse:
        bind(bind(a, b), b) == a
    because each element of b is in {-1, +1} and b*b == 1 elementwise.
    """
    return a * b


def bundle(*vectors: np.ndarray) -> np.ndarray:
    """
    Bundle: sum of vectors.

    The result is similar (in cosine) to each input. Retrieval happens
    by Unbind against a role vector, followed by cleanup against a codebook.
    We do NOT normalize — coefficients carry weight information (e.g. decay).
    """
    if len(vectors) == 0:
        raise ValueError("bundle() requires at least one vector")
    return np.sum(np.stack(vectors, axis=0), axis=0)


def permute(v: np.ndarray, k: int = 1) -> np.ndarray:
    """Permute: cyclic shift by k positions. Encodes order/sequence."""
    return np.roll(v, k)


def unbind(structure: np.ndarray, role: np.ndarray) -> np.ndarray:
    """
    Unbind: same operation as Bind in bipolar space.

    Given structure = Bundle(Bind(a, role_a), Bind(b, role_b)),
    unbind(structure, role_a) returns a + noise, where noise is
    approximately orthogonal to a in high-dim space.
    """
    return bind(structure, role)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0 for zero vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ----- Codebook ------------------------------------------------------------

class Codebook:
    """
    A named collection of HDC vectors. Three jobs:
      1. Assign a fresh random bipolar vector to a new name on first use
      2. Look up by name
      3. Cleanup — given a noisy vector, find the closest stored vector
         (this is how Unbind results get "resolved" to actual concepts)
    """

    def __init__(self, dim: int, seed: int = 0):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._vectors: dict[str, np.ndarray] = {}

    def add(self, name: str) -> np.ndarray:
        """Return the vector for `name`, creating it if new."""
        if name not in self._vectors:
            self._vectors[name] = random_bipolar(self.dim, self._rng)
        return self._vectors[name]

    def get(self, name: str) -> np.ndarray:
        return self._vectors[name]

    def __contains__(self, name: str) -> bool:
        return name in self._vectors

    def __len__(self) -> int:
        return len(self._vectors)

    def names(self) -> list[str]:
        return list(self._vectors.keys())

    def cleanup(
        self,
        noisy: np.ndarray,
        restrict_to: Optional[Iterable[str]] = None,
    ) -> tuple[str, float]:
        """
        Find the stored vector most similar to `noisy`.
        Returns (best_name, cosine_similarity).
        """
        candidates = list(restrict_to) if restrict_to is not None else list(self._vectors.keys())
        if not candidates:
            return "<empty>", 0.0
        best_name = candidates[0]
        best_sim = -float("inf")
        for name in candidates:
            sim = cosine(noisy, self._vectors[name])
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return best_name, best_sim


# ----- Memory --------------------------------------------------------------

class HDCMemory:
    """
    Decaying holographic memory.

    Each write multiplies existing state by `decay` and adds the new
    structure. decay = 1.0 means pure sum (no recency bias).
    decay < 1.0 means recent writes dominate retrieval.

    The entire memory is a single vector. It is read by Unbind
    against a role vector.
    """

    def __init__(self, dim: int, decay: float = 1.0):
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.dim = dim
        self.decay = decay
        self._state = np.zeros(dim, dtype=np.float32)

    def write(self, structure: np.ndarray) -> None:
        self._state = self.decay * self._state + structure

    def read(self) -> np.ndarray:
        return self._state

    def query(self, role: np.ndarray) -> np.ndarray:
        """Unbind the memory state against a role vector."""
        return unbind(self._state, role)

    def reset(self) -> None:
        self._state = np.zeros(self.dim, dtype=np.float32)
