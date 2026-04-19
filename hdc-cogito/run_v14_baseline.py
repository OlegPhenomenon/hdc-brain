"""
v14 baseline — non-interactive generation on CPU.

Goal: establish what v14 actually produces on a fixed set of prompts,
at a fixed seed, so every subsequent experiment has a concrete thing to
beat or match. This is the zero point for all AGI-direction work on v14.

Writes results to docs/experiments/hdc-cogito/v14_baseline_log.md.
"""
from __future__ import annotations

import os
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
    print(f"Loading {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HDCBrainV14(vocab_size=ckpt["vocab_size"], **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, ckpt


@torch.no_grad()
def generate(
    model: HDCBrainV14,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    *,
    max_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 40,
    rep_penalty: float = 1.3,
    n_thoughts: int = 2,
    device: str = "cpu",
    seed: int = 42,
):
    """One greedy+topk+rep-penalty generation with per-step diagnostics."""
    torch.manual_seed(seed)
    ids = sp.encode(prompt)
    x = torch.tensor([ids], device=device)

    generated_ids: list[int] = []
    entropies: list[float] = []
    top1_probs: list[float] = []

    for _ in range(max_tokens):
        logits, _ = model(x[:, -model.max_seq_len :], n_thoughts=n_thoughts)
        step_logits = logits[0, -1, :].clone()
        full_probs = F.softmax(step_logits, dim=-1)
        ent = -(full_probs * torch.log(full_probs.clamp_min(1e-9))).sum().item()
        entropies.append(ent)
        top1_probs.append(full_probs.max().item())

        step_logits = step_logits / temperature
        if rep_penalty > 1.0 and generated_ids:
            for t in set(generated_ids):
                if step_logits[t] > 0:
                    step_logits[t] /= rep_penalty
                else:
                    step_logits[t] *= rep_penalty
        if top_k > 0:
            v, _ = torch.topk(step_logits, top_k)
            step_logits[step_logits < v[-1]] = float("-inf")
        probs = F.softmax(step_logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        tid = nxt.item()
        generated_ids.append(tid)
        x = torch.cat([x, nxt.unsqueeze(0)], dim=1)
        if tid == sp.eos_id():
            break

    text = sp.decode(ids + generated_ids)
    return {
        "text": text,
        "n_tokens": len(generated_ids),
        "avg_entropy": sum(entropies) / max(len(entropies), 1),
        "avg_top1": sum(top1_probs) / max(len(top1_probs), 1),
    }


def main():
    device = "cpu"
    ckpt = V14_DIR / "best_hdc_brain_v14_sft.pt"
    tok = V14_DIR / "bpe_ru_16k.model"
    if not ckpt.exists():
        print(f"missing: {ckpt}")
        return
    sp = spm.SentencePieceProcessor(model_file=str(tok))
    model, info = load_model(ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}  val_loss={info.get('val_loss', '?')}  sft={info.get('sft', False)}")
    print(f"  vocab={sp.get_piece_size()}  max_seq_len={model.max_seq_len}  hdc_dim={model.hdc_dim}")
    print()

    out_path = Path("/Users/oleghasjanov/Documents/learning/hoffman_swarm/docs/experiments/hdc-cogito/v14_baseline_log.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# v14 baseline generation log")
    lines.append("")
    lines.append(f"Date: 2026-04-11  device: {device}  seed: 42")
    lines.append("")
    lines.append(f"- checkpoint: `{ckpt.name}` val_loss={info.get('val_loss', '?'):.4f}")
    lines.append(f"- params: {n_params:,}")
    lines.append(f"- hdc_dim: {model.hdc_dim}, n_blocks: {len(model.blocks)}, max_thoughts: {model.thought_loop.max_thoughts}")
    lines.append(f"- gen config: max_tokens=60, temperature=0.8, top_k=40, rep_penalty=1.3, n_thoughts=2")
    lines.append("")
    gates = torch.sigmoid(model.thought_loop.thought_gates).tolist()
    lines.append(f"Learned thought gates: {[round(g, 3) for g in gates]}")
    lines.append("")
    lines.append("## Generations")
    lines.append("")

    total_tok = 0
    total_sec = 0.0
    ent_all = []
    top1_all = []
    for i, p in enumerate(PROMPTS, 1):
        print(f"[{i}] prompt: {p!r}")
        t0 = time.time()
        r = generate(model, sp, p, n_thoughts=2, device=device, seed=42 + i)
        dt = time.time() - t0
        total_tok += r["n_tokens"]
        total_sec += dt
        ent_all.append(r["avg_entropy"])
        top1_all.append(r["avg_top1"])
        print(f"    -> {r['text']}")
        print(f"    tokens={r['n_tokens']} time={dt:.1f}s  ent={r['avg_entropy']:.2f}  top1={r['avg_top1']:.3f}")
        lines.append(f"### {i}. `{p}`")
        lines.append("")
        lines.append(f"> {r['text']}")
        lines.append("")
        lines.append(
            f"`{r['n_tokens']} tok`  `{dt:.1f}s`  `{r['n_tokens']/max(dt,1e-6):.2f} tok/s`  "
            f"`avg_entropy={r['avg_entropy']:.2f}`  `avg_top1={r['avg_top1']:.3f}`"
        )
        lines.append("")

    tok_per_sec = total_tok / max(total_sec, 1e-6)
    avg_ent = sum(ent_all) / len(ent_all)
    avg_top1 = sum(top1_all) / len(top1_all)
    print()
    print(f"TOTAL: {total_tok} tokens in {total_sec:.1f}s => {tok_per_sec:.2f} tok/s")
    print(f"AVG entropy: {avg_ent:.2f}  AVG top1: {avg_top1:.3f}")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- total tokens: {total_tok}, total time: {total_sec:.1f}s, throughput: **{tok_per_sec:.2f} tok/s**")
    lines.append(f"- avg entropy per step: {avg_ent:.2f}")
    lines.append(f"- avg top-1 prob per step: {avg_top1:.3f}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLog written: {out_path}")


if __name__ == "__main__":
    main()
