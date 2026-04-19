"""
v14 adaptive thought halting — first AGI-direction experiment on real v14.

Diagnostic (`run_v14_thought_diag.py`) showed:
- n=1 top1_acc = 5.3%, true_prob = 0.038
- n=2 top1_acc = 22.8%, true_prob = 0.188
- n=3 top1_acc = 33.3%, true_prob = 0.209  (best)
- n=4 broken (unlearned gate)

Default chat.py uses n=2. This experiment asks two questions:

Q1. How much perplexity improves on a held-out Russian paragraph when
    we move from the default n=2 to a fixed n=3?
Q2. Can an adaptive policy (start at n=1, add thoughts only when
    uncertain) match fixed n=3 perplexity at less compute?

The adaptive policy is the concrete AGI-direction claim: the model
decides itself how many iterations it needs, without a learned halting
gate. If adaptive ≈ fixed-3 in perplexity while using fewer average
thoughts, we have free adaptive compute on a real language model.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm

V14_DIR = Path("/Users/oleghasjanov/Documents/learning/hoffman_swarm/hdc-brain-v14")
sys.path.insert(0, str(V14_DIR))
from hdc_brain_v14 import HDCBrainV14  # noqa: E402


EVAL_TEXTS = [
    # Tolstoy (classic), unlikely to be in v14 web training verbatim
    "Все счастливые семьи похожи друг на друга каждая несчастливая семья "
    "несчастлива по своему Все смешалось в доме Облонских Жена узнала что муж "
    "был в связи с бывшею в их доме француженкою гувернанткой и объявила мужу "
    "что не может жить с ним в одном доме",
    # Dostoevsky
    "В начале июля в чрезвычайно жаркое время под вечер один молодой человек "
    "вышел из своей каморки которую нанимал от жильцов в Столярном переулке на "
    "улицу и медленно как бы в нерешимости отправился к Ку",
    # Contemporary Russian (popular science tone)
    "Наш мозг обрабатывает огромное количество информации каждую секунду но мы "
    "не замечаем большую её часть потому что внимание это ограниченный ресурс "
    "и сознание показывает нам только то что считает важным для текущей задачи",
    # Dialogue-ish sentence
    "Она посмотрела в окно и увидела как падает первый снег и тогда она поняла "
    "что детство уже давно закончилось но сердце всё ещё помнит ту зимнюю радость",
]


def load_model(ckpt_path: Path, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HDCBrainV14(vocab_size=ckpt["vocab_size"], **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model


@torch.no_grad()
def score_fixed(model, ids: list[int], n_thoughts: int, device: str):
    """Compute mean NLL per token for positions [1, T) with fixed n_thoughts.
    Uses one full-sequence forward — efficient but position_t sees future-masked self.
    """
    x = torch.tensor([ids], device=device)
    logits, _ = model(x, n_thoughts=n_thoughts)  # (1, T, V)
    log_probs = F.log_softmax(logits[0], dim=-1)  # (T, V)
    total_nll = 0.0
    n = 0
    for t in range(len(ids) - 1):
        total_nll += -log_probs[t, ids[t + 1]].item()
        n += 1
    return total_nll / max(n, 1), n


@torch.no_grad()
def score_adaptive(
    model,
    ids: list[int],
    *,
    threshold: float,
    max_thoughts: int,
    device: str,
):
    """Adaptive: for each position, forward with n=1, if top1<threshold, escalate.

    We re-forward the whole prefix at higher n_thoughts. This is the
    dumb version that does k separate forwards per uncertain position;
    it is slow but purely correct. Fixing this with a sliced ThoughtLoop
    is a later optimisation.
    """
    total_nll = 0.0
    n = 0
    thoughts_used = []
    for t in range(len(ids) - 1):
        prefix = torch.tensor([ids[: t + 1]], device=device)
        prefix = prefix[:, -model.max_seq_len :]
        # Start with n=1
        n_used = 1
        logits, _ = model(prefix, n_thoughts=1)
        step = logits[0, -1, :]
        probs = F.softmax(step, dim=-1)
        top1 = probs.max().item()
        while top1 < threshold and n_used < max_thoughts:
            n_used += 1
            logits, _ = model(prefix, n_thoughts=n_used)
            step = logits[0, -1, :]
            probs = F.softmax(step, dim=-1)
            top1 = probs.max().item()
        nll = -math.log(max(probs[ids[t + 1]].item(), 1e-12))
        total_nll += nll
        n += 1
        thoughts_used.append(n_used)
    return total_nll / max(n, 1), n, thoughts_used


@torch.no_grad()
def score_perstep_fixed(model, ids: list[int], n_thoughts: int, device: str):
    """Equivalent of adaptive's per-step forwards but at fixed n_thoughts.
    This makes adaptive vs fixed comparable — same loop structure, only
    the halting condition differs."""
    total_nll = 0.0
    n = 0
    for t in range(len(ids) - 1):
        prefix = torch.tensor([ids[: t + 1]], device=device)
        prefix = prefix[:, -model.max_seq_len :]
        logits, _ = model(prefix, n_thoughts=n_thoughts)
        step = logits[0, -1, :]
        probs = F.softmax(step, dim=-1)
        nll = -math.log(max(probs[ids[t + 1]].item(), 1e-12))
        total_nll += nll
        n += 1
    return total_nll / max(n, 1), n


def main():
    device = "cpu"
    ckpt = V14_DIR / "best_hdc_brain_v14_sft.pt"
    tok = V14_DIR / "bpe_ru_16k.model"
    sp = spm.SentencePieceProcessor(model_file=str(tok))
    model = load_model(ckpt, device)
    print(f"model loaded. thought_gates={[round(g,3) for g in torch.sigmoid(model.thought_loop.thought_gates).tolist()]}")
    print()

    # Tokenise everything once
    encoded = [sp.encode(t) for t in EVAL_TEXTS]
    total_tok = sum(len(e) - 1 for e in encoded)
    print(f"eval corpus: {len(EVAL_TEXTS)} passages, {total_tok} prediction targets")
    print()

    # --- Part 1: FAST full-sequence fixed n=2 vs n=3 ---
    # (this is what chat.py would see if it had --n_thoughts=N)
    print("=" * 60)
    print("Part 1: full-sequence fixed n_thoughts")
    print("=" * 60)
    for n in (1, 2, 3):
        t0 = time.time()
        nlls = []
        for ids in encoded:
            nll, _ = score_fixed(model, ids, n_thoughts=n, device=device)
            nlls.append(nll)
        dt = time.time() - t0
        avg_nll = sum(nlls) / len(nlls)
        ppl = math.exp(avg_nll)
        print(f"  n={n}: nll={avg_nll:.4f}  ppl={ppl:.2f}  time={dt:.1f}s")

    # --- Part 2: per-step fixed (slow but comparable to adaptive) ---
    # To keep time bounded, we run only on the two shortest passages.
    short = sorted(encoded, key=len)[:2]
    short_tok = sum(len(e) - 1 for e in short)
    print()
    print("=" * 60)
    print(f"Part 2: per-step incremental (comparable to adaptive), {short_tok} targets")
    print("=" * 60)

    fixed_results = {}
    for n in (1, 2, 3):
        t0 = time.time()
        nlls = []
        for ids in short:
            nll, _ = score_perstep_fixed(model, ids, n_thoughts=n, device=device)
            nlls.append(nll)
        dt = time.time() - t0
        avg_nll = sum(nlls) / len(nlls)
        ppl = math.exp(avg_nll)
        fixed_results[n] = (avg_nll, ppl, dt)
        print(f"  fixed n={n}: nll={avg_nll:.4f}  ppl={ppl:.2f}  time={dt:.1f}s  ({short_tok/dt:.2f} tok/s)")

    # --- Part 3: adaptive ---
    print()
    print("=" * 60)
    print("Part 3: adaptive halting (start n=1, escalate if top1<threshold)")
    print("=" * 60)
    adaptive_results = {}
    for thr in (0.3, 0.5, 0.7):
        t0 = time.time()
        nlls = []
        all_thoughts = []
        for ids in short:
            nll, _, thoughts = score_adaptive(
                model, ids, threshold=thr, max_thoughts=3, device=device
            )
            nlls.append(nll)
            all_thoughts.extend(thoughts)
        dt = time.time() - t0
        avg_nll = sum(nlls) / len(nlls)
        ppl = math.exp(avg_nll)
        avg_thoughts = sum(all_thoughts) / len(all_thoughts)
        dist = {1: 0, 2: 0, 3: 0}
        for x in all_thoughts:
            dist[x] += 1
        adaptive_results[thr] = (avg_nll, ppl, dt, avg_thoughts, dist)
        print(
            f"  thr={thr}: nll={avg_nll:.4f}  ppl={ppl:.2f}  time={dt:.1f}s  "
            f"({short_tok/dt:.2f} tok/s)  avg_thoughts={avg_thoughts:.2f}  dist={dist}"
        )

    # --- Report ---
    out_path = Path(
        "/Users/oleghasjanov/Documents/learning/hoffman_swarm/docs/experiments/hdc-cogito/v14_adaptive_log.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# v14 adaptive thought halting")
    lines.append("")
    lines.append("Date: 2026-04-11")
    lines.append("")
    lines.append(f"Eval corpus: {len(EVAL_TEXTS)} passages, {total_tok} targets total.")
    lines.append("")
    lines.append("## Full-sequence fixed n_thoughts (chat.py equivalent)")
    lines.append("")
    lines.append("| n | ppl |")
    lines.append("|---:|---:|")
    # We reprint from accumulator — recomputing would be cheap
    for n in (1, 2, 3):
        t0 = time.time()
        nlls = []
        for ids in encoded:
            nll, _ = score_fixed(model, ids, n_thoughts=n, device=device)
            nlls.append(nll)
        avg = sum(nlls) / len(nlls)
        lines.append(f"| {n} | {math.exp(avg):.2f} |")
    lines.append("")
    lines.append(f"Per-step incremental results below use only the two shortest passages ({short_tok} targets) for tractability.")
    lines.append("")
    lines.append("## Per-step fixed")
    lines.append("")
    lines.append("| n | ppl | time (s) | tok/s |")
    lines.append("|---:|---:|---:|---:|")
    for n in (1, 2, 3):
        nll, ppl, dt = fixed_results[n]
        lines.append(f"| {n} | {ppl:.2f} | {dt:.1f} | {short_tok/dt:.2f} |")
    lines.append("")
    lines.append("## Adaptive halting")
    lines.append("")
    lines.append("| threshold | ppl | avg thoughts | time (s) | dist (n=1/2/3) |")
    lines.append("|---:|---:|---:|---:|---|")
    for thr, (nll, ppl, dt, avg_t, dist) in adaptive_results.items():
        lines.append(
            f"| {thr} | {ppl:.2f} | {avg_t:.2f} | {dt:.1f} | "
            f"{dist[1]}/{dist[2]}/{dist[3]} |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- Compare adaptive ppl against fixed n=3 ppl. If adaptive is")
    lines.append("  within a few % of fixed-3 while using avg_thoughts < 3, we")
    lines.append("  have adaptive compute for free on v14.")
    lines.append("- The distribution dist tells us what fraction of positions")
    lines.append("  actually needed the extra compute.")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(f"report written: {out_path}")


if __name__ == "__main__":
    main()
