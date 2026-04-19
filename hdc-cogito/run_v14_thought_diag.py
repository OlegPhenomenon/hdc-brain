"""
v14 thought-loop diagnostic.

Question: does running more thoughts at INFERENCE time actually change
the model's predictions in a useful direction?

v14 was trained with max_thoughts=4, but ThoughtLoop default at eval is
only 2. If the iterative refinement story holds, then for a given
prefix:
- top-1 probability of the "correct" next token should RISE as
  n_thoughts grows (more thinking ⇒ more confidence)
- or at least the ranking of the true continuation should improve
- and the change should be larger for initially uncertain steps

If n_thoughts has no effect, or makes things worse, then adaptive halting
is pointless and the whole thought_loop direction needs rethinking.

Method:
- Take a long fixed Russian paragraph, encode to BPE.
- For every position t in [32, T), feed x[:t] through model for
  n_thoughts in {1,2,3,4}.
- Record: true next token's rank and prob, top-1 prob, entropy.
- Segment results by initial (n_thoughts=1) uncertainty.

Writes a compact stats table — pass/fail is decided by whether top-1
climbs on UNCERTAIN steps when thoughts are added.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm

V14_DIR = Path("/Users/oleghasjanov/Documents/learning/hoffman_swarm/hdc-brain-v14")
sys.path.insert(0, str(V14_DIR))
from hdc_brain_v14 import HDCBrainV14  # noqa: E402


# A reasonably long coherent Russian text — taken from a classic so v14
# was never trained on it verbatim (v14 trained on web Russian).
TEXT = (
    "Все счастливые семьи похожи друг на друга каждая несчастливая семья "
    "несчастлива по своему Все смешалось в доме Облонских Жена узнала что муж "
    "был в связи с бывшею в их доме француженкою гувернанткой и объявила мужу "
    "что не может жить с ним в одном доме Положение это продолжалось уже третий "
    "день и мучительно чувствовалось и самими супругами и всеми членами семьи и "
    "домочадцами Все члены семьи и домочадцы чувствовали что нет смысла в их "
    "сожительстве и что на каждом постоялом дворе случайно сошедшиеся люди более "
    "связаны между собой чем они члены семьи и домочадцы Облонских"
)


def load_model(ckpt_path: Path, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HDCBrainV14(vocab_size=ckpt["vocab_size"], **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model


@torch.no_grad()
def diag(model, ids: list[int], device: str):
    """For every suffix position, measure effect of n_thoughts.

    Returns a list of per-step dicts with keys:
      t, true_id, ent1, top1_1, rank1, ent2, top1_2, rank2, ... , delta_top1
    """
    T = len(ids)
    rows = []
    for t in range(32, T - 1):
        x = torch.tensor([ids[:t + 1]], device=device)
        true_id = ids[t + 1]
        per_n = {}
        for n in (1, 2, 3, 4):
            logits, _ = model(x[:, -model.max_seq_len:], n_thoughts=n)
            step = logits[0, -1, :]
            probs = F.softmax(step, dim=-1)
            true_prob = probs[true_id].item()
            top1 = probs.max().item()
            rank = (probs > true_prob).sum().item()  # 0 == truly top-1
            ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum().item()
            per_n[n] = {
                "top1": top1,
                "true_prob": true_prob,
                "rank": rank,
                "ent": ent,
            }
        rows.append({
            "t": t,
            "true_id": true_id,
            "per_n": per_n,
        })
    return rows


def summarise(rows):
    """Aggregate the diagnostic into a decidable pass/fail picture."""
    n_total = len(rows)
    if n_total == 0:
        return {}
    out = {"n_positions": n_total, "per_n": {}}
    for n in (1, 2, 3, 4):
        top1s = [r["per_n"][n]["top1"] for r in rows]
        tprobs = [r["per_n"][n]["true_prob"] for r in rows]
        ents = [r["per_n"][n]["ent"] for r in rows]
        ranks = [r["per_n"][n]["rank"] for r in rows]
        ranked1 = sum(1 for r in ranks if r == 0)
        out["per_n"][n] = {
            "avg_top1": sum(top1s) / n_total,
            "avg_true_prob": sum(tprobs) / n_total,
            "avg_ent": sum(ents) / n_total,
            "top1_acc": ranked1 / n_total,
            "avg_rank": sum(ranks) / n_total,
        }

    # Segment by uncertainty at n=1
    buckets = {"very_uncertain": [], "uncertain": [], "confident": []}
    for r in rows:
        t1 = r["per_n"][1]["top1"]
        if t1 < 0.2:
            buckets["very_uncertain"].append(r)
        elif t1 < 0.5:
            buckets["uncertain"].append(r)
        else:
            buckets["confident"].append(r)
    out["buckets"] = {}
    for name, bucket in buckets.items():
        if not bucket:
            continue
        per_n = {}
        for n in (1, 2, 3, 4):
            top1s = [r["per_n"][n]["top1"] for r in bucket]
            tprobs = [r["per_n"][n]["true_prob"] for r in bucket]
            per_n[n] = {
                "avg_top1": sum(top1s) / len(bucket),
                "avg_true_prob": sum(tprobs) / len(bucket),
            }
        out["buckets"][name] = {"n": len(bucket), "per_n": per_n}
    return out


def format_report(s: dict) -> str:
    lines = []
    lines.append("# v14 thought-loop diagnostic")
    lines.append("")
    lines.append("Does running more thoughts at inference change predictions?")
    lines.append("")
    lines.append(f"Positions analysed: {s['n_positions']}")
    lines.append("")
    lines.append("## Overall (every position)")
    lines.append("")
    lines.append("| n_thoughts | avg top1 | avg true_prob | avg ent | top1 acc | avg rank |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for n in (1, 2, 3, 4):
        r = s["per_n"][n]
        lines.append(
            f"| {n} | {r['avg_top1']:.4f} | {r['avg_true_prob']:.4f} | "
            f"{r['avg_ent']:.3f} | {r['top1_acc']:.3f} | {r['avg_rank']:.1f} |"
        )
    lines.append("")
    lines.append("## Segmented by n=1 uncertainty (true_prob change is the key column)")
    lines.append("")
    for name in ("very_uncertain", "uncertain", "confident"):
        if name not in s["buckets"]:
            continue
        b = s["buckets"][name]
        lines.append(f"### {name}  (n={b['n']})")
        lines.append("")
        lines.append("| n_thoughts | avg top1 | avg true_prob |")
        lines.append("|---:|---:|---:|")
        for n in (1, 2, 3, 4):
            r = b["per_n"][n]
            lines.append(f"| {n} | {r['avg_top1']:.4f} | {r['avg_true_prob']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    device = "cpu"
    ckpt = V14_DIR / "best_hdc_brain_v14_sft.pt"
    tok = V14_DIR / "bpe_ru_16k.model"
    sp = spm.SentencePieceProcessor(model_file=str(tok))
    model = load_model(ckpt, device)
    gates = torch.sigmoid(model.thought_loop.thought_gates).tolist()
    print(f"model loaded. thought_gates={[round(g,3) for g in gates]}")

    ids = sp.encode(TEXT)
    print(f"text: {len(ids)} tokens; analysing positions 32..{len(ids)-1}")
    rows = diag(model, ids, device)
    s = summarise(rows)
    print()
    print("OVERALL:")
    for n in (1, 2, 3, 4):
        r = s["per_n"][n]
        print(
            f"  n={n}: top1={r['avg_top1']:.4f} true_prob={r['avg_true_prob']:.4f} "
            f"ent={r['avg_ent']:.3f} top1_acc={r['top1_acc']:.3f}"
        )
    print()
    for name in ("very_uncertain", "uncertain", "confident"):
        if name not in s["buckets"]:
            continue
        b = s["buckets"][name]
        print(f"{name} (n={b['n']}):")
        for n in (1, 2, 3, 4):
            r = b["per_n"][n]
            print(f"  n={n}: top1={r['avg_top1']:.4f} true_prob={r['avg_true_prob']:.4f}")
        print()

    out_path = Path(
        "/Users/oleghasjanov/Documents/learning/hoffman_swarm/docs/experiments/hdc-cogito/v14_thought_diag_log.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_report(s), encoding="utf-8")
    print(f"report written: {out_path}")


if __name__ == "__main__":
    main()
