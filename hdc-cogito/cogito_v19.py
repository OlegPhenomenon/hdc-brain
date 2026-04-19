"""
Cogito v19 — VSA-native language model.

This is NOT v14. This is a different architecture built from scratch
on pure HDC operations. v14's codebook is used only as the initial
concept pool (Phase 0 substrate). Everything else is VSA.

Architecture (proposal §6.2):
    tokens → vsa_encode → reasoning_loop → memory.bundle_in → vsa_decode → tokens

Knowledge does not live in weights. It lives in structured relationships
between vectors in HDC memory. The ONLY way the model learns is by
accumulating experience in memory through Bind/Bundle/Unbind.

Three types of vectors (§5.3):
    Concepts  — things that exist (from v14 codebook, orthogonal subset)
    Roles     — positions in structure (declared, fixed, ~12)
    Markers   — modalities (declared, fixed, ~6)

Four operations (§5.4):
    Bind    — attach concept to role (elementwise multiply)
    Bundle  — combine bindings (sum)
    Permute — express order (cyclic shift)
    Unbind  — extract info (same as Bind in bipolar)

The reasoning block is the ONLY component with iterative dynamics.
It is reused N times per forward pass — depth of reasoning is iteration
count, not layer count.
"""
from __future__ import annotations

import numpy as np
from hdc_lang import random_bipolar, bind, bundle, permute, unbind, cosine, HDCMemory


ROLE_NAMES = [
    "agent", "patient", "theme", "location", "time",
    "cause", "instrument", "goal", "source", "manner",
    "possession", "part",
]

MARKER_NAMES = [
    "question", "negation", "possible", "past", "future", "imperative",
]


class AssociativeMemory:
    """Item-based associative memory.

    Each (key, value) pair is stored individually. Retrieval uses
    cosine similarity to rank keys and returns a weighted blend of
    the top-k matching values. This avoids the ~30-50 pair capacity
    ceiling of bundled holographic memory while staying in VSA.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._keys: list[np.ndarray] = []
        self._vals: list[np.ndarray] = []

    def store(self, key: np.ndarray, value: np.ndarray):
        self._keys.append(key)
        self._vals.append(value)

    def query(self, q: np.ndarray, top_k: int = 5) -> np.ndarray:
        if not self._keys:
            return np.zeros(self.dim, dtype=np.float32)
        keys = np.stack(self._keys)  # (N, D)
        sims = (keys @ q) / np.sqrt(self.dim)
        top_ids = np.argsort(sims)[-top_k:]
        result = np.zeros(self.dim, dtype=np.float32)
        for idx in top_ids:
            s = sims[idx]
            if s > 0:
                result += s * self._vals[idx]
        return result

    def __len__(self):
        return len(self._keys)


class CogitoBrain:
    """Cogito v19 core.

    This is a language model that works by:
    1. Reading text into HDC memory (training = filling memory)
    2. Predicting next token via iterative reasoning over memory
    3. Recording its own predictions in self-memory (Rung 4)

    No gradient descent. No backprop. No weight matrices.
    Knowledge = accumulated VSA structures in memory.
    """

    def __init__(
        self,
        token_pool: np.ndarray,
        dim: int | None = None,
        context_len: int = 5,
        max_iter: int = 10,
        conf_threshold: float = 0.05,
        retrieval_k: int = 5,
        seed: int = 42,
    ):
        self.V, self.D = token_pool.shape
        self.dim = dim or self.D
        assert self.dim == self.D
        self.token_pool = token_pool
        self.context_len = context_len
        self.max_iter = max_iter
        self.conf_threshold = conf_threshold
        self.retrieval_k = retrieval_k

        rng = np.random.default_rng(seed)

        # Declared role vectors (§5.3) — fixed random bipolar
        self.roles = {
            name: random_bipolar(self.dim, rng) for name in ROLE_NAMES
        }

        # Declared marker vectors (§5.3) — fixed random bipolar
        self.markers = {
            name: random_bipolar(self.dim, rng) for name in MARKER_NAMES
        }

        # Self vector (Rung 2, §7)
        self.self_vector = random_bipolar(self.dim, rng)

        # Memory banks (§4.7 — per-memory decay)
        # Associative: context → next token
        self.assoc = AssociativeMemory(self.dim)
        # Self-memory (Rung 4): context → (my prediction, correctness)
        self.self_mem = AssociativeMemory(self.dim)

        self.n_observed = 0

    def _context_vector(self, token_ids: list[int]) -> np.ndarray:
        """Encode a short token sequence into a single VSA context vector.

        Uses positional permute + bundle (classic HDC sequence encoding).
        Only the last `context_len` tokens are used.
        """
        ids = token_ids[-self.context_len:]
        ctx = np.zeros(self.dim, dtype=np.float32)
        for pos, tid in enumerate(ids):
            tv = self.token_pool[tid]
            ctx += permute(tv, pos)
        return ctx

    def observe(self, token_ids: list[int], next_id: int):
        """Learn one (context → next_token) association.

        This is how Cogito "trains" — no gradient, no loss function.
        Store context vector as key, next-token vector as value.
        """
        ctx = self._context_vector(token_ids)
        next_vec = self.token_pool[next_id]
        self.assoc.store(ctx, next_vec)
        self.n_observed += 1

    def observe_text(self, all_ids: list[int]):
        """Observe an entire text — sliding window of associations."""
        for t in range(self.context_len, len(all_ids)):
            ctx_ids = all_ids[t - self.context_len : t]
            self.observe(ctx_ids, all_ids[t])

    def _reasoning_step(
        self, h: np.ndarray, ctx: np.ndarray, iteration: int
    ) -> np.ndarray:
        """One iteration of the reasoning block (§6.3).

        Each iteration refines the next-token hypothesis by consulting
        memory and self-memory with the CURRENT hypothesis as an
        additional query signal.
        """
        # 1. Query associative memory with context
        mem_hint = self.assoc.query(ctx, top_k=self.retrieval_k)

        # 2. Query self-memory (Rung 4) — what did I predict before
        #    in similar situations?
        self_hint = self.self_mem.query(ctx, top_k=3)

        # 3. Cross-reference: use current hypothesis to re-query memory
        #    (the hypothesis may be closer to the true answer than raw context)
        if iteration > 0:
            cross_hint = self.assoc.query(h, top_k=3)
        else:
            cross_hint = np.zeros(self.dim, dtype=np.float32)

        # 4. Bundle everything — this is where iteration helps:
        #    each pass adds more signal from different angles
        weights = np.array([1.0, 1.0, 0.3, 0.5])
        refined = (
            weights[0] * h
            + weights[1] * mem_hint
            + weights[2] * self_hint
            + weights[3] * cross_hint
        )
        return refined

    def _confidence(self, h: np.ndarray) -> tuple[int, float, float]:
        """Compute confidence: top-1 margin against token pool."""
        sims = (self.token_pool @ h) / np.sqrt(self.dim)
        order = np.argsort(sims)[::-1]
        best_id = int(order[0])
        best_sim = float(sims[best_id])
        second_sim = float(sims[order[1]])
        margin = best_sim - second_sim
        return best_id, best_sim, margin

    def predict(self, token_ids: list[int]) -> tuple[int, int, float]:
        """Predict next token given a prefix.

        Returns (predicted_token_id, iterations_used, confidence).
        """
        ctx = self._context_vector(token_ids)

        # Initial hypothesis: direct memory query
        h = self.assoc.query(ctx, top_k=self.retrieval_k)

        # Reasoning loop (§6.2) — iterative refinement
        best_id, best_sim, margin = self._confidence(h)
        for iteration in range(self.max_iter):
            h = self._reasoning_step(h, ctx, iteration)
            new_id, new_sim, new_margin = self._confidence(h)
            if new_margin > self.conf_threshold:
                best_id, best_sim, margin = new_id, new_sim, new_margin
                break
            best_id, best_sim, margin = new_id, new_sim, new_margin

        # Rung 4: record prediction in self-memory
        pred_vec = self.token_pool[best_id]
        self.self_mem.store(ctx, pred_vec)

        return best_id, iteration + 1, margin

    def generate(
        self,
        prefix_ids: list[int],
        max_tokens: int = 30,
        eos_id: int | None = None,
    ) -> tuple[list[int], list[int]]:
        """Autoregressive generation."""
        seq = list(prefix_ids)
        iters_used = []
        for _ in range(max_tokens):
            pred_id, iters, conf = self.predict(seq)
            seq.append(pred_id)
            iters_used.append(iters)
            if eos_id is not None and pred_id == eos_id:
                break
        return seq, iters_used

    def stats(self) -> dict:
        return {
            "n_observed": self.n_observed,
            "dim": self.dim,
            "vocab": self.V,
            "context_len": self.context_len,
            "max_iter": self.max_iter,
            "assoc_size": len(self.assoc),
            "self_mem_size": len(self.self_mem),
        }
