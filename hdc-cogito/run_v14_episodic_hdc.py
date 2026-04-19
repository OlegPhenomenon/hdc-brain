"""
v14 + episodic HDC buffer beyond max_seq_len.

v14 has max_seq_len=512. Past that, tokens are dropped entirely. We
build a complementary HDC episodic buffer that accumulates every token
the model has already seen (even those dropped from the window), using
bipolar codes and cyclic permutation for positional coding. At each
prediction step, we query the buffer with the current in-window prefix
and mix the resulting log-probs into v14's logits.

This is the one experiment from autonomous_session_2026-04-12.md's
"what points toward" list that is safe to run without Oleg's sign-off,
because it is pure inference.

Setup:
- Long Russian passage (1500+ tokens).
- Evaluate positions [1024, end): ensures most of the earlier content
  is OUT of v14's 512 window.
- Compare: pure v14 at positions >512, vs v14 + episodic HDC bias.

If the buffer helps on out-of-window positions, HDC is providing real
long-context memory that v14 can't access natively.
If it doesn't, HDC cleanup at this scale is too noisy to be useful.
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


# Long passage — 5 paragraphs of Russian, ~1500+ BPE tokens hopefully.
LONG_TEXT = """
Все счастливые семьи похожи друг на друга каждая несчастливая семья
несчастлива по своему Все смешалось в доме Облонских Жена узнала что муж
был в связи с бывшею в их доме француженкою гувернанткой и объявила мужу
что не может жить с ним в одном доме Положение это продолжалось уже третий
день и мучительно чувствовалось и самими супругами и всеми членами семьи
и домочадцами Все члены семьи и домочадцы чувствовали что нет смысла в
их сожительстве и что на каждом постоялом дворе случайно сошедшиеся люди
более связаны между собой чем они члены семьи и домочадцы Облонских.

Жена не выходила из своих комнат мужа третий день не было дома Дети бегали
по всему дому как потерянные англичанка поссорилась с экономкой и написала
записку приятельнице прося приискать ей новое место повар ушел вчера со
двора во время самого обеда черная кухарка и кучер просили расчета.

На третий день после ссоры князь Степан Аркадьич Облонский Стива как его
звали в свете в обычный час то есть в восемь часов утра проснулся не в
спальне жены а в своем кабинете на сафьянном диване Он повернул свое полное
изнеженное тело на пружинах дивана как бы желая опять заснуть надолго с
другой стороны крепко обнял подушку и прижался к ней щекою но вдруг
вскочил сел на диван и открыл глаза.

Да да как это было подумал он вспоминая сон Да как это было Да Алабин
давал обед в Дармштадте нет не в Дармштадте а что то американское Да но
там Дармштадт был в Америке Да Алабин давал обед на стеклянных столах да
и столы пели Il mio tesoro нет не Il mio tesoro а что то получше и какие
то маленькие графинчики и они же женщины говорил он.

Глаза Степана Аркадьича весело заблестели и он задумался улыбаясь Да хорошо
было очень хорошо Много еще было там отличного да не скажешь словами и
мыслями даже наяву не выразишь И он заметив полосу света пробившуюся сбоку
одной из суконных стор весело скинул ноги с дивана отыскал ими шитые женой
прошлого года ко дню рождения золотообрезные сафьянные туфли и по старой
девятилетней привычке не вставая потянулся рукой к тому месту где в спальне
у него висел халат.
"""


def load_model(ckpt_path: Path, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HDCBrainV14(vocab_size=ckpt["vocab_size"], **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model


def bipolar_codebook(model):
    with torch.no_grad():
        cb = torch.sign(model.codebook.data)
        cb = torch.where(cb == 0, torch.ones_like(cb), cb)
    return cb


def permute(v: torch.Tensor, shift: int) -> torch.Tensor:
    idx = (torch.arange(v.shape[-1], device=v.device) - shift) % v.shape[-1]
    return v.index_select(-1, idx)


@torch.no_grad()
def episodic_bundle(codebook: torch.Tensor, ids: list[int], decay: float) -> torch.Tensor:
    """Build S = Σ_t decay^(t_last - t) * permute(code[x_t], t) over all tokens.

    At generation time, query S with the "next step slot" by computing
    unbind against the expected role for the next position. We'll only
    store an absolute-position-keyed episodic vector.

    Returns a bundle vector of shape (D,).
    """
    D = codebook.shape[1]
    device = codebook.device
    bundle = torch.zeros(D, device=device)
    for t, tid in enumerate(ids):
        c = codebook[tid]
        bundle = decay * bundle + permute(c, t)
    return bundle


@torch.no_grad()
def episodic_query(bundle: torch.Tensor, codebook: torch.Tensor, position: int) -> torch.Tensor:
    """Query bundle for the token at `position`: unbind with inverse permute.

    Since we stored `permute(c, t)`, to recover `c` at position t we
    apply permute by -t. Cosine match against codebook → per-vocab score.
    """
    D = codebook.shape[1]
    noisy = permute(bundle, -position)
    sims = (codebook @ noisy) / math.sqrt(D)
    return sims


@torch.no_grad()
def score(
    model,
    ids: list[int],
    codebook: torch.Tensor,
    *,
    use_episodic: bool,
    alpha: float,
    decay: float,
    start_from: int,
    device: str,
):
    total_nll = 0.0
    n = 0
    for t in range(start_from, len(ids) - 1):
        prefix = torch.tensor([ids[: t + 1]], device=device)[:, -model.max_seq_len:]
        logits, _ = model(prefix, n_thoughts=3)
        lm_log = F.log_softmax(logits[0, -1, :], dim=-1)

        if use_episodic and alpha > 0.0:
            # Episode is everything before the current window start
            window_start = max(0, (t + 1) - model.max_seq_len)
            if window_start > 0:
                out_of_window = ids[:window_start]
                bundle = episodic_bundle(codebook, out_of_window, decay=decay)
                q = episodic_query(bundle, codebook, position=t + 1)
                hdc_log = F.log_softmax(q, dim=-1)
                mixed = (1.0 - alpha) * lm_log + alpha * hdc_log
                mixed = F.log_softmax(mixed, dim=-1)
            else:
                mixed = lm_log
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
    ids = sp.encode(LONG_TEXT.strip())
    print(f"long text: {len(ids)} tokens, window={model.max_seq_len}")
    if len(ids) < model.max_seq_len + 30:
        print("text too short to have meaningful out-of-window positions")
        return

    start = model.max_seq_len  # only score positions where there IS out-of-window content
    n_targets = len(ids) - 1 - start
    print(f"scoring positions {start}..{len(ids)-1}  ({n_targets} targets)")
    print()

    # Pure v14 baseline at n=3 on out-of-window tail
    t0 = time.time()
    nll, _ = score(model, ids, cb, use_episodic=False, alpha=0.0, decay=1.0, start_from=start, device=device)
    dt = time.time() - t0
    print(f"  v14 pure (n=3):        ppl={math.exp(nll):.2f}  time={dt:.1f}s")
    baseline_ppl = math.exp(nll)

    # Sweep alphas and decays
    print()
    print("=== v14 + episodic HDC (n=3) ===")
    rows = []
    for alpha in (0.05, 0.1, 0.2):
        for decay in (0.999, 0.99, 0.95):
            t0 = time.time()
            nll, _ = score(model, ids, cb, use_episodic=True, alpha=alpha, decay=decay,
                           start_from=start, device=device)
            dt = time.time() - t0
            ppl = math.exp(nll)
            rows.append((alpha, decay, ppl, dt))
            flag = " <-- improved" if ppl < baseline_ppl else ""
            print(f"  alpha={alpha:<4}  decay={decay:<6}  ppl={ppl:.2f}  time={dt:.1f}s{flag}")

    out = Path(
        "/Users/oleghasjanov/Documents/learning/hoffman_swarm/docs/experiments/hdc-cogito/v14_episodic_hdc_log.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# v14 + episodic HDC buffer beyond 512-token window",
        "",
        "Date: 2026-04-12  device: CPU  n_thoughts=3",
        "",
        f"Long text: {len(ids)} BPE tokens (v14 window={model.max_seq_len}).",
        f"Scored positions {start}..{len(ids)-1} — every target has ≥1 token out of window.",
        "",
        "## Pure v14 baseline",
        "",
        f"- ppl = **{baseline_ppl:.2f}**",
        "",
        "## With episodic HDC bias",
        "",
        "| alpha | decay | ppl | vs baseline |",
        "|---:|---:|---:|---|",
    ]
    for alpha, decay, ppl, _ in rows:
        delta = (ppl - baseline_ppl) / baseline_ppl * 100
        lines.append(f"| {alpha} | {decay} | {ppl:.2f} | {delta:+.1f}% |")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
