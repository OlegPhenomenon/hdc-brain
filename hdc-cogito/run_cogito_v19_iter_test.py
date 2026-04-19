"""
Test whether reasoning iterations help in Cogito v19.

Force all iterations (conf_threshold=inf) and compare accuracy
at iter=1 vs iter=5 vs iter=10. If more iterations improve accuracy,
H2 holds on real text with the full Cogito pipeline.
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
    "связаны между собой чем они члены семьи и домочадцы Облонских "
    "Жена не выходила из своих комнат мужа третий день не было дома Дети бегали "
    "по всему дому как потерянные англичанка поссорилась с экономкой и написала "
    "записку приятельнице прося приискать ей новое место повар ушел вчера со двора "
    "во время самого обеда черная кухарка и кучер просили расчета "
    "На третий день после ссоры князь Степан Аркадьич Облонский Стива как его "
    "звали в свете в обычный час то есть в восемь часов утра проснулся не в "
    "спальне жены а в своем кабинете на сафьянном диване"
)

TEST_TEXT = (
    "Он повернул свое полное изнеженное тело на пружинах дивана как бы желая "
    "опять заснуть надолго с другой стороны крепко обнял подушку"
)


def score_at_iters(brain: CogitoBrain, test_ids: list[int], max_iter: int):
    """Measure accuracy when forced to run exactly max_iter iterations."""
    ctx_len = brain.context_len
    correct = 0
    total = 0
    ranks = []

    for t in range(ctx_len, len(test_ids) - 1):
        ctx = test_ids[t - ctx_len : t]
        true_id = test_ids[t]
        ctx_vec = brain._context_vector(ctx)

        h = brain.assoc.query(ctx_vec, top_k=brain.retrieval_k)
        for it in range(max_iter):
            h = brain._reasoning_step(h, ctx_vec, it)

        sims = (brain.token_pool @ h) / math.sqrt(brain.dim)
        pred_id = int(np.argmax(sims))
        rank = int((sims > sims[true_id]).sum())

        if pred_id == true_id:
            correct += 1
        ranks.append(rank)
        total += 1

    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "median_rank": int(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "p10_rank": int(np.percentile(ranks, 10)),
        "p90_rank": int(np.percentile(ranks, 90)),
    }


def main():
    print("=" * 60)
    print("Cogito v19 — iteration effect test")
    print("=" * 60)

    pool = load_v14_pool()
    sp = spm.SentencePieceProcessor(model_file=str(V14_DIR / "bpe_ru_16k.model"))

    train_ids = sp.encode(TRAIN_TEXT)
    test_ids = sp.encode(TEST_TEXT)
    print(f"Train: {len(train_ids)} tokens  Test: {len(test_ids)} tokens")
    print()

    brain = CogitoBrain(
        token_pool=pool,
        context_len=5,
        max_iter=20,
        conf_threshold=999.0,  # never stop early
        retrieval_k=10,
        seed=42,
    )
    brain.observe_text(train_ids)
    print(f"Memory: {brain.n_observed} associations")
    print()

    print(f"{'iters':>6} {'acc':>8} {'med_rank':>10} {'mean_rank':>10} {'p10':>6} {'p90':>6}")
    print("-" * 52)
    for n_iter in (0, 1, 2, 3, 5, 10, 15, 20):
        r = score_at_iters(brain, test_ids, max_iter=n_iter)
        print(
            f"{n_iter:>6} {r['accuracy']*100:>7.1f}% {r['median_rank']:>10} "
            f"{r['mean_rank']:>10.0f} {r['p10_rank']:>6} {r['p90_rank']:>6}"
        )

    # Also test: does more TRAINING data help?
    print()
    print("=" * 60)
    print("Data scaling test (does more text help?)")
    print("=" * 60)
    all_ids = sp.encode(TRAIN_TEXT)
    fracs = [0.25, 0.5, 0.75, 1.0]
    for frac in fracs:
        n = int(len(all_ids) * frac)
        b = CogitoBrain(
            token_pool=pool, context_len=5, max_iter=5,
            conf_threshold=999.0, retrieval_k=10, seed=42,
        )
        b.observe_text(all_ids[:n])
        r = score_at_iters(b, test_ids, max_iter=5)
        print(
            f"  {frac*100:>5.0f}% data ({b.n_observed:>4} assoc): "
            f"acc={r['accuracy']*100:.1f}%  med_rank={r['median_rank']}  "
            f"mean_rank={r['mean_rank']:.0f}"
        )


if __name__ == "__main__":
    main()
