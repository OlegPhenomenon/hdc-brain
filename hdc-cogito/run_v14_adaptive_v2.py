"""
v14 adaptive thought halting v2 — smarter signal.

Finding from v1: starting at n=1 and escalating on top1<threshold did
worse than fixed n=3, because when n=1 is wrong-and-confident the true
token gets ~0 probability and NLL explodes.

v2 tries the intelligent version:
- Always do n=2 (already 2.5x better than n=1)
- Escalate to n=3 only when a real uncertainty signal fires:
    * entropy > H_THRESHOLD     (model is diffuse)
    * top1 - top2 < GAP_THRESH  (top two are close)
    * both above together

If even a mild escalation strategy matches fixed-n=3 perplexity at
avg_thoughts < 3, adaptive compute is real on v14.
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
    "Все счастливые семьи похожи друг на друга каждая несчастливая семья "
    "несчастлива по своему Все смешалось в доме Облонских Жена узнала что муж "
    "был в связи с бывшею в их доме француженкою гувернанткой",
    "Наш мозг обрабатывает огромное количество информации каждую секунду но мы "
    "не замечаем большую её часть потому что внимание это ограниченный ресурс",
]


def load_model(ckpt_path: Path, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HDCBrainV14(vocab_size=ckpt["vocab_size"], **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model


def entropy_of(probs: torch.Tensor) -> float:
    return -(probs * torch.log(probs.clamp_min(1e-9))).sum().item()


def top_gap(probs: torch.Tensor) -> float:
    v, _ = torch.topk(probs, 2)
    return (v[0] - v[1]).item()


@torch.no_grad()
def score_fixed_perstep(model, ids: list[int], n_thoughts: int, device: str):
    total_nll = 0.0
    n = 0
    for t in range(len(ids) - 1):
        prefix = torch.tensor([ids[: t + 1]], device=device)[:, -model.max_seq_len:]
        logits, _ = model(prefix, n_thoughts=n_thoughts)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        total_nll += -math.log(max(probs[ids[t + 1]].item(), 1e-12))
        n += 1
    return total_nll / max(n, 1), n


@torch.no_grad()
def score_adaptive(
    model,
    ids: list[int],
    *,
    ent_threshold: float,
    gap_threshold: float,
    device: str,
):
    """Baseline n=2; escalate to n=3 if entropy>H OR gap<G."""
    total_nll = 0.0
    n = 0
    thoughts_used = []
    for t in range(len(ids) - 1):
        prefix = torch.tensor([ids[: t + 1]], device=device)[:, -model.max_seq_len:]
        logits, _ = model(prefix, n_thoughts=2)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        ent = entropy_of(probs)
        gap = top_gap(probs)
        n_used = 2
        if ent > ent_threshold or gap < gap_threshold:
            logits, _ = model(prefix, n_thoughts=3)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            n_used = 3
        total_nll += -math.log(max(probs[ids[t + 1]].item(), 1e-12))
        n += 1
        thoughts_used.append(n_used)
    return total_nll / max(n, 1), n, thoughts_used


def main():
    device = "cpu"
    ckpt = V14_DIR / "best_hdc_brain_v14_sft.pt"
    tok = V14_DIR / "bpe_ru_16k.model"
    sp = spm.SentencePieceProcessor(model_file=str(tok))
    model = load_model(ckpt, device)
    print()
    encoded = [sp.encode(t) for t in EVAL_TEXTS]
    n_tgts = sum(len(e) - 1 for e in encoded)
    print(f"eval corpus: {len(encoded)} passages, {n_tgts} targets")
    print()

    # Baselines
    print("=== fixed baselines ===")
    baselines = {}
    for n in (2, 3):
        t0 = time.time()
        nlls = []
        for ids in encoded:
            nll, _ = score_fixed_perstep(model, ids, n_thoughts=n, device=device)
            nlls.append(nll)
        dt = time.time() - t0
        avg = sum(nlls) / len(nlls)
        ppl = math.exp(avg)
        baselines[n] = (avg, ppl, dt)
        print(f"  fixed n={n}: ppl={ppl:.2f}  time={dt:.1f}s  ({n_tgts/dt:.2f} tok/s)")

    # Adaptive sweep
    print()
    print("=== adaptive (n=2 baseline, escalate to n=3) ===")
    configs = [
        (10.0, -1.0),   # never escalate → should match fixed n=2
        (4.0, 0.0),     # escalate if ent>4.0
        (3.0, 0.0),     # escalate if ent>3.0 (stricter)
        (2.0, 0.0),     # escalate if ent>2.0 (very aggressive)
        (-1.0, 0.1),    # escalate if top1-top2<0.1
        (3.0, 0.15),    # combined
        (0.0, 1e9),     # always escalate → should match fixed n=3
    ]
    adaptive_rows = []
    for ent_t, gap_t in configs:
        t0 = time.time()
        nlls = []
        thoughts = []
        for ids in encoded:
            nll, _, ts = score_adaptive(
                model, ids, ent_threshold=ent_t, gap_threshold=gap_t, device=device
            )
            nlls.append(nll)
            thoughts.extend(ts)
        dt = time.time() - t0
        avg = sum(nlls) / len(nlls)
        ppl = math.exp(avg)
        avg_t = sum(thoughts) / len(thoughts)
        frac_n3 = sum(1 for x in thoughts if x == 3) / len(thoughts)
        adaptive_rows.append((ent_t, gap_t, ppl, avg_t, frac_n3, dt))
        print(
            f"  ent>{ent_t:<5} or gap<{gap_t:<5}:  ppl={ppl:.2f}  "
            f"avg_t={avg_t:.2f}  frac_n3={frac_n3:.2f}  time={dt:.1f}s"
        )

    # Report
    out_path = Path(
        "/Users/oleghasjanov/Documents/learning/hoffman_swarm/docs/experiments/hdc-cogito/v14_adaptive_v2_log.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# v14 adaptive thought halting v2")
    lines.append("")
    lines.append("Date: 2026-04-11  device: CPU  eval: 2 Russian passages, per-step")
    lines.append("")
    lines.append("## Fixed baselines")
    lines.append("")
    lines.append("| n | ppl | time (s) | tok/s |")
    lines.append("|---:|---:|---:|---:|")
    for n in (2, 3):
        nll, ppl, dt = baselines[n]
        lines.append(f"| {n} | {ppl:.2f} | {dt:.1f} | {n_tgts/dt:.2f} |")
    lines.append("")
    lines.append("## Adaptive (start n=2, escalate to n=3 when entropy>H or top1-top2<G)")
    lines.append("")
    lines.append("| H | G | ppl | avg thoughts | % n=3 | time (s) |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for ent_t, gap_t, ppl, avg_t, frac_n3, dt in adaptive_rows:
        lines.append(
            f"| {ent_t} | {gap_t} | {ppl:.2f} | {avg_t:.2f} | "
            f"{frac_n3*100:.0f}% | {dt:.1f} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport written: {out_path}")


if __name__ == "__main__":
    main()
