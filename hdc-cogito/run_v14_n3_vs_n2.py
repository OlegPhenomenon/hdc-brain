"""
Side-by-side: v14 generation with n=2 (current chat.py default) vs n=3.

Diagnostic and per-step ppl both indicate n=3 is strongly better on v14
(ppl 102 vs 329 on the held-out passages, and 33% top-1 vs 23%).
This script produces the human-readable comparison: on each prompt,
generate with both settings from the same seed and store both outputs
side by side so Oleg can read them and judge coherence.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm

V14_DIR = Path("/Users/oleghasjanov/Documents/learning/hoffman_swarm/hdc-brain-v14")
sys.path.insert(0, str(V14_DIR))
from hdc_brain_v14 import HDCBrainV14  # noqa: E402


PROMPTS = [
    "Человек это",
    "Главное в жизни —",
    "Почему небо голубое?",
    "Я думаю, что",
    "Россия —",
    "Компьютер работает благодаря",
    "Сегодня я расскажу вам",
    "Самая большая проблема в том, что",
]


def load_model(ckpt_path: Path, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HDCBrainV14(vocab_size=ckpt["vocab_size"], **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model


@torch.no_grad()
def generate(model, sp, prompt, *, n_thoughts, seed,
             max_tokens=60, temperature=0.8, top_k=40, rep_penalty=1.3, device="cpu"):
    torch.manual_seed(seed)
    ids = sp.encode(prompt)
    x = torch.tensor([ids], device=device)
    gen = []
    top1s = []
    ents = []
    for _ in range(max_tokens):
        logits, _ = model(x[:, -model.max_seq_len:], n_thoughts=n_thoughts)
        raw = logits[0, -1, :].clone()
        full = F.softmax(raw, dim=-1)
        top1s.append(full.max().item())
        ents.append(-(full * torch.log(full.clamp_min(1e-9))).sum().item())
        raw = raw / temperature
        if rep_penalty > 1.0 and gen:
            for t in set(gen):
                if raw[t] > 0:
                    raw[t] /= rep_penalty
                else:
                    raw[t] *= rep_penalty
        if top_k > 0:
            v, _ = torch.topk(raw, top_k)
            raw[raw < v[-1]] = float("-inf")
        probs = F.softmax(raw, dim=-1)
        nxt = torch.multinomial(probs, 1)
        tid = nxt.item()
        gen.append(tid)
        x = torch.cat([x, nxt.unsqueeze(0)], dim=1)
        if tid == sp.eos_id():
            break
    return {
        "text": sp.decode(ids + gen),
        "n_tokens": len(gen),
        "avg_top1": sum(top1s) / len(top1s),
        "avg_ent": sum(ents) / len(ents),
    }


def main():
    device = "cpu"
    ckpt = V14_DIR / "best_hdc_brain_v14_sft.pt"
    tok = V14_DIR / "bpe_ru_16k.model"
    sp = spm.SentencePieceProcessor(model_file=str(tok))
    model = load_model(ckpt, device)

    lines = ["# v14 generation: n=2 vs n=3 side-by-side", "",
             "Date: 2026-04-12  seed=42+i, temp=0.8, top_k=40, rep_penalty=1.3, max_tokens=60",
             "", "The two generations use the same seed for the torch RNG, so differences",
             "come purely from the thought_loop count.", ""]
    agg = {"n2": [], "n3": []}
    times = {"n2": 0.0, "n3": 0.0}
    totals = {"n2": 0, "n3": 0}
    for i, p in enumerate(PROMPTS, 1):
        print(f"[{i}] {p!r}")
        t0 = time.time(); r2 = generate(model, sp, p, n_thoughts=2, seed=42+i, device=device); d2 = time.time()-t0
        t0 = time.time(); r3 = generate(model, sp, p, n_thoughts=3, seed=42+i, device=device); d3 = time.time()-t0
        print(f"  n=2 ({r2['n_tokens']} tok, {d2:.1f}s, top1={r2['avg_top1']:.3f}):")
        print(f"    {r2['text']}")
        print(f"  n=3 ({r3['n_tokens']} tok, {d3:.1f}s, top1={r3['avg_top1']:.3f}):")
        print(f"    {r3['text']}")
        print()
        lines.append(f"### {i}. `{p}`")
        lines.append("")
        lines.append(f"**n=2** (`top1={r2['avg_top1']:.3f}` `ent={r2['avg_ent']:.2f}`  `{d2:.1f}s`)")
        lines.append("")
        lines.append(f"> {r2['text']}")
        lines.append("")
        lines.append(f"**n=3** (`top1={r3['avg_top1']:.3f}` `ent={r3['avg_ent']:.2f}`  `{d3:.1f}s`)")
        lines.append("")
        lines.append(f"> {r3['text']}")
        lines.append("")
        agg["n2"].append(r2["avg_top1"]); agg["n3"].append(r3["avg_top1"])
        times["n2"] += d2; times["n3"] += d3
        totals["n2"] += r2["n_tokens"]; totals["n3"] += r3["n_tokens"]

    avg2 = sum(agg["n2"]) / len(agg["n2"])
    avg3 = sum(agg["n3"]) / len(agg["n3"])
    tps2 = totals["n2"] / times["n2"]
    tps3 = totals["n3"] / times["n3"]
    lines.append("## Aggregate"); lines.append("")
    lines.append("| metric | n=2 | n=3 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| avg top-1 prob | {avg2:.3f} | {avg3:.3f} |")
    lines.append(f"| throughput (tok/s) | {tps2:.2f} | {tps3:.2f} |")
    lines.append(f"| total time (s) | {times['n2']:.1f} | {times['n3']:.1f} |")
    lines.append("")
    out = Path("/Users/oleghasjanov/Documents/learning/hoffman_swarm/docs/experiments/hdc-cogito/v14_n2_vs_n3_log.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {out}")
    print(f"AGGREGATE avg_top1: n=2={avg2:.3f}  n=3={avg3:.3f}  (relative diff {(avg3-avg2)/avg2*100:+.1f}%)")
    print(f"SPEED: n=2={tps2:.2f} tok/s  n=3={tps3:.2f} tok/s  (n=3 is {tps2/tps3:.2f}x slower)")


if __name__ == "__main__":
    main()
