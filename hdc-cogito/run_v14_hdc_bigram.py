"""
v14 + HDC bigram memory bias.

Hypothesis: the same v14 bipolar codebook that the transformer-ish core
uses can also power an explicit HDC bigram memory. At generation time
we maintain a running bundle S = Σ_t code[x_t] ⊙ perm(code[x_{t+1}])
over the prefix. To predict next token, we unbind with
perm⁻¹(S ⊙ code[x_last]) and cosine-match against the whole codebook,
producing a distribution that we mix into v14's logits.

Why this is interesting:
- It's an explicit structural memory that lives outside the parameters
  of v14 but USES v14's learned embeddings (so no retraining at all).
- The bundle is accumulated at inference — on every new token it grows.
- It gives v14 a real scratchpad: "what typically followed this token
  earlier in this same context?" — a form of in-context bigram lookup
  that should especially help repeated or rare local patterns.

If n=2 + HDC bias ≥ n=3 baseline, we get n=3 quality at n=2 cost.
If it just matches n=2, we learn HDC bigram is too weak in isolation.
If it helps on top of n=3, we get best-of-both.

This is the first experiment that puts HDC mechanism INSIDE v14's
inference loop rather than building a separate reasoning system.
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


def bipolar_codebook(model: HDCBrainV14) -> torch.Tensor:
    """Extract the sign'd bipolar codebook that v14 effectively uses internally."""
    with torch.no_grad():
        cb = torch.sign(model.codebook.data)  # (V, D), {-1, +1} (and 0 → 1)
        cb = torch.where(cb == 0, torch.ones_like(cb), cb)
    return cb


def make_perm_indices(dim: int, shift: int = 1):
    return (torch.arange(dim) + shift) % dim


@torch.no_grad()
def hdc_bigram_logits(
    codebook: torch.Tensor,
    prefix_ids: list[int],
    last_id: int,
    perm_idx: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute an HDC-based estimate for next-token logits.

    Builds S = Σ_{t<len-1} code[prefix[t]] ⊙ perm(code[prefix[t+1]]) over
    the whole visible prefix. Then next-token estimate is:
        estimate = inv_perm( S ⊙ code[last_id] )
    which for permutation-as-shift means applying the inverse shift.
    Finally cosine similarity against the codebook gives a score per
    vocab entry. We treat that score as logits.
    """
    V, D = codebook.shape
    device = codebook.device
    if len(prefix_ids) < 2:
        return torch.zeros(V, device=device)

    codes = codebook[prefix_ids]  # (L, D)
    a = codes[:-1]  # (L-1, D)
    b_permuted = codes[1:].index_select(1, perm_idx)  # (L-1, D) after forward shift
    bundle = (a * b_permuted).sum(dim=0)  # (D,)

    last_code = codebook[last_id]  # (D,)
    noisy_next_permuted = bundle * last_code  # (D,)
    # invert the permutation: shift in the other direction
    inv_perm = (torch.arange(D, device=device) - 1) % D
    noisy_next = noisy_next_permuted.index_select(0, inv_perm)

    # Cosine similarity against codebook → per-vocab score
    # codebook rows are {-1,+1} so each has norm sqrt(D); divide once.
    sims = (codebook @ noisy_next) / math.sqrt(D)
    return sims / temperature  # higher temperature softens the distribution


@torch.no_grad()
def score_hdc_augmented(
    model,
    ids: list[int],
    codebook: torch.Tensor,
    perm_idx: torch.Tensor,
    *,
    n_thoughts: int,
    alpha: float,
    hdc_temp: float,
    device: str,
):
    """Per-step scoring. At each position t we:
      1. forward v14 with n_thoughts → logits_lm
      2. compute HDC bigram logits from prefix[:t+1]
      3. mix: logits = (1-alpha)*logits_lm + alpha*logits_hdc  (in log-prob space)
      4. measure NLL on true next token
    """
    total_nll = 0.0
    n = 0
    for t in range(len(ids) - 1):
        prefix = torch.tensor([ids[: t + 1]], device=device)[:, -model.max_seq_len:]
        logits, _ = model(prefix, n_thoughts=n_thoughts)
        lm_log = F.log_softmax(logits[0, -1, :], dim=-1)

        if alpha > 0.0:
            hdc_l = hdc_bigram_logits(
                codebook, ids[: t + 1], ids[t], perm_idx, temperature=hdc_temp
            )
            hdc_log = F.log_softmax(hdc_l, dim=-1)
            mixed = (1.0 - alpha) * lm_log + alpha * hdc_log
            mixed = F.log_softmax(mixed, dim=-1)  # renormalise
        else:
            mixed = lm_log

        total_nll += -mixed[ids[t + 1]].item()
        n += 1
    return total_nll / max(n, 1), n


def main():
    device = "cpu"
    ckpt = V14_DIR / "best_hdc_brain_v14_sft.pt"
    tok = V14_DIR / "bpe_ru_16k.model"
    sp = spm.SentencePieceProcessor(model_file=str(tok))
    model = load_model(ckpt, device)
    cb = bipolar_codebook(model).to(device)
    V, D = cb.shape
    perm_idx = make_perm_indices(D, shift=1).to(device)
    print(f"codebook: V={V}, D={D}")
    print()

    encoded = [sp.encode(t) for t in EVAL_TEXTS]
    n_tgts = sum(len(e) - 1 for e in encoded)
    print(f"eval corpus: {len(encoded)} passages, {n_tgts} targets")
    print()

    # pure LM baselines
    print("=== pure LM baselines (per-step) ===")
    lm_results = {}
    for n in (2, 3):
        t0 = time.time()
        nlls = []
        for ids in encoded:
            nll, _ = score_hdc_augmented(
                model, ids, cb, perm_idx,
                n_thoughts=n, alpha=0.0, hdc_temp=1.0, device=device,
            )
            nlls.append(nll)
        dt = time.time() - t0
        avg = sum(nlls) / len(nlls)
        ppl = math.exp(avg)
        lm_results[n] = (avg, ppl, dt)
        print(f"  n={n}, alpha=0.0: ppl={ppl:.2f}  time={dt:.1f}s")

    # HDC-augmented sweep
    print()
    print("=== HDC bigram augmented (n=2) ===")
    results = []
    for alpha in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
        for hdc_temp in (1.0,):
            t0 = time.time()
            nlls = []
            for ids in encoded:
                nll, _ = score_hdc_augmented(
                    model, ids, cb, perm_idx,
                    n_thoughts=2, alpha=alpha, hdc_temp=hdc_temp, device=device,
                )
                nlls.append(nll)
            dt = time.time() - t0
            avg = sum(nlls) / len(nlls)
            ppl = math.exp(avg)
            results.append(("n=2", alpha, hdc_temp, ppl, dt))
            print(f"  n=2  alpha={alpha:<4}  temp={hdc_temp:<4}  ppl={ppl:.2f}  time={dt:.1f}s")

    print()
    print("=== HDC bigram augmented (n=3) ===")
    for alpha in (0.0, 0.05, 0.1, 0.2):
        t0 = time.time()
        nlls = []
        for ids in encoded:
            nll, _ = score_hdc_augmented(
                model, ids, cb, perm_idx,
                n_thoughts=3, alpha=alpha, hdc_temp=1.0, device=device,
            )
            nlls.append(nll)
        dt = time.time() - t0
        avg = sum(nlls) / len(nlls)
        ppl = math.exp(avg)
        results.append(("n=3", alpha, 1.0, ppl, dt))
        print(f"  n=3  alpha={alpha:<4}  temp=1.0   ppl={ppl:.2f}  time={dt:.1f}s")

    # Report
    out = Path(
        "/Users/oleghasjanov/Documents/learning/hoffman_swarm/docs/experiments/hdc-cogito/v14_hdc_bigram_log.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# v14 + HDC bigram memory bias")
    lines.append("")
    lines.append(f"Date: 2026-04-11  device: CPU  eval: {n_tgts} targets (2 Russian passages)")
    lines.append("")
    lines.append("## Pure LM")
    lines.append("")
    lines.append("| n | ppl |")
    lines.append("|---:|---:|")
    for n in (2, 3):
        _, ppl, _ = lm_results[n]
        lines.append(f"| {n} | {ppl:.2f} |")
    lines.append("")
    lines.append("## Augmented (mixed with HDC bigram log-probs)")
    lines.append("")
    lines.append("| n | alpha | hdc_temp | ppl |")
    lines.append("|---:|---:|---:|---:|")
    for tag, a, t, ppl, _ in results:
        lines.append(f"| {tag} | {a} | {t} | {ppl:.2f} |")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
