"""
Cogito v19 first demo — pure VSA language model.

Zero trained weights. Knowledge = VSA structures in memory.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

V14_DIR = Path("/Users/oleghasjanov/Documents/learning/hoffman_swarm/hdc-brain-v14")
sys.path.insert(0, str(V14_DIR))

from cogito_v19 import CogitoBrain
from run_milestone_F1 import load_v14_pool


TRAIN_TEXT = (
    "Все счастливые семьи похожи друг на друга каждая несчастливая семья "
    "несчастлива по своему Все смешалось в доме Облонских Жена узнала что муж "
    "был в связи с бывшею в их доме француженкою гувернанткой и объявила мужу "
    "что не может жить с ним в одном доме Положение это продолжалось уже третий "
    "день и мучительно чувствовалось и самими супругами и всеми членами семьи и "
    "домочадцами Все члены семьи и домочадцы чувствовали что нет смысла в их "
    "сожительстве и что на каждом постоялом дворе случайно сошедшиеся люди более "
    "связаны между собой чем они члены семьи и домочадцы Облонских"
)

TEST_TEXT = (
    "Жена не выходила из своих комнат мужа третий день не было дома Дети бегали "
    "по всему дому как потерянные англичанка поссорилась с экономкой"
)


def evaluate(brain: CogitoBrain, test_ids: list[int], sp):
    """Measure next-token accuracy on held-out text."""
    ctx_len = brain.context_len
    correct = 0
    total = 0
    iters_sum = 0
    ranks = []

    for t in range(ctx_len, len(test_ids) - 1):
        ctx = test_ids[t - ctx_len : t]
        true_id = test_ids[t]
        pred_id, iters, conf = brain.predict(ctx)
        iters_sum += iters
        total += 1
        if pred_id == true_id:
            correct += 1
        # Compute rank of true token
        ctx_vec = brain._context_vector(ctx)
        h = brain.assoc.query(ctx_vec, top_k=brain.retrieval_k)
        for it in range(brain.max_iter):
            h = brain._reasoning_step(h, ctx_vec, it)
        sims = (brain.token_pool @ h) / math.sqrt(brain.dim)
        rank = int((sims > sims[true_id]).sum())
        ranks.append(rank)

    median_rank = int(np.median(ranks)) if ranks else brain.V
    mean_rank = float(np.mean(ranks)) if ranks else brain.V
    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "avg_iters": iters_sum / max(total, 1),
        "median_rank": median_rank,
        "mean_rank": mean_rank,
    }


def main():
    print("=" * 60)
    print("Cogito v19 — pure VSA language model")
    print("=" * 60)

    pool = load_v14_pool()
    V, D = pool.shape
    sp = spm.SentencePieceProcessor(model_file=str(V14_DIR / "bpe_ru_16k.model"))
    print(f"Substrate: V={V}, D={D}")

    train_ids = sp.encode(TRAIN_TEXT)
    test_ids = sp.encode(TEST_TEXT)
    print(f"Train: {len(train_ids)} tokens  Test: {len(test_ids)} tokens")
    print()

    for ctx_len in (3, 5):
        for k in (3, 10):
            brain = CogitoBrain(
                token_pool=pool,
                context_len=ctx_len,
                max_iter=5,
                conf_threshold=0.1,
                retrieval_k=k,
                seed=42,
            )

            t0 = time.time()
            brain.observe_text(train_ids)
            dt = time.time() - t0
            print(f"ctx={ctx_len} k={k}: trained {brain.n_observed} assoc in {dt:.2f}s")

            t0 = time.time()
            r = evaluate(brain, test_ids, sp)
            dt = time.time() - t0
            print(f"  accuracy: {r['accuracy']*100:.1f}%  "
                  f"median_rank: {r['median_rank']}  mean_rank: {r['mean_rank']:.0f}  "
                  f"avg_iters: {r['avg_iters']:.1f}  "
                  f"time: {dt:.1f}s")
            print(f"  (random baseline rank: {V//2})")
            print()

    # Generation
    print("=" * 60)
    print("Generation (ctx=5, k=10)")
    print("=" * 60)
    brain = CogitoBrain(
        token_pool=pool, context_len=5, max_iter=5,
        conf_threshold=0.1, retrieval_k=10, seed=42,
    )
    brain.observe_text(train_ids)

    for prompt in ["Все счастливые семьи", "Жена узнала что", "члены семьи и"]:
        ids = sp.encode(prompt)
        seq, iters = brain.generate(ids, max_tokens=10)
        text = sp.decode(seq)
        print(f"  \"{prompt}\" → \"{text}\"")
        print(f"    iters: {iters}")
    print()


if __name__ == "__main__":
    main()
