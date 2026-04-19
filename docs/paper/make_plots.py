"""Generate plots for HDC-Brain paper from experiment logs."""
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

BASE = Path(__file__).parent.parent / "experiments" / "v14.1-finetune-mixed"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def load_jsonl(path):
    events = []
    with open(path) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return events


def parse_finetune_log(path):
    """Parse finetune.log for iter, loss, BPB."""
    steps = []
    evals = []
    with open(path) as f:
        for line in f:
            m = re.match(r"iter\s+(\d+)\s+\|\s+loss\s+([\d.]+)", line)
            if m:
                steps.append((int(m.group(1)), float(m.group(2))))
                continue
            m = re.match(r"\[[\d.]+h, iter (\d+)\] Train:([\d.]+) Val:([\d.]+) BPB:([\d.]+)", line)
            if m:
                evals.append((int(m.group(1)), float(m.group(2)),
                              float(m.group(3)), float(m.group(4))))
    return steps, evals


# ============================================================
# Figure 1: Pretrain BPB curve
# ============================================================
# Only keep CLEAN pretrain evals (before CogitLayer experiments)
# BPB > 6 = CogitLayer experiment which disrupted training
print("Plotting pretrain curve...")
events = load_jsonl(BASE / "experiment.jsonl")
pretrain_evals = [e for e in events if e.get("event") == "eval" and e["bpb"] < 6.0]
# Deduplicate by iter (keep earliest — clean run)
seen = {}
for e in pretrain_evals:
    if e["iter"] not in seen:
        seen[e["iter"]] = e
pretrain_evals = sorted(seen.values(), key=lambda e: e["iter"])

iters = [e["iter"] for e in pretrain_evals]
bpbs = [e["bpb"] for e in pretrain_evals]
train_losses = [e.get("train", 0) for e in pretrain_evals]
val_losses = [e.get("val", 0) for e in pretrain_evals]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

best_bpb = min(bpbs)
best_iter = iters[bpbs.index(best_bpb)]

ax1.plot(iters, bpbs, '-', color='#1f77b4', linewidth=1.8, label='Val BPB')
ax1.axhline(y=best_bpb, color='red', linestyle='--', alpha=0.6,
            label=f'Best: {best_bpb:.3f} (iter {best_iter})')
ax1.scatter([best_iter], [best_bpb], color='red', s=60, zorder=5)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Bits Per Byte (BPB)')
ax1.set_title('Pretrain BPB — HDC-Brain v14.1 on FineWeb-Edu 3B')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(5.3, 5.85)

ax2.plot(iters, train_losses, '-', color='#2ca02c', linewidth=1.8, label='Train')
ax2.plot(iters, val_losses, '-', color='#d62728', linewidth=1.8, label='Val')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss (cross-entropy)')
ax2.set_title('Pretrain Loss — Train vs Val')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "fig_pretrain.png", dpi=150, bbox_inches='tight')
plt.savefig(OUT / "fig_pretrain.pdf", bbox_inches='tight')
plt.close()
print(f"  {len(pretrain_evals)} evals, BPB {best_bpb:.3f}..{max(bpbs):.3f}")

# ============================================================
# Figure 2: Finetune v3 curve (quality_v3 dataset — final result)
# ============================================================
print("Plotting finetune v3 curve...")
V3_LOG = Path(__file__).parent.parent / "experiments" / "finetune_v3.log"
steps, evals = parse_finetune_log(V3_LOG)
s_iter, s_loss = zip(*steps)
e_iter, e_train, e_val, e_bpb = zip(*evals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(s_iter, s_loss, '-', color='#1f77b4', linewidth=0.6, alpha=0.4, label='Train step')
ax1.plot(e_iter, e_train, 'o-', color='#2ca02c', markersize=4, linewidth=1.5, label='Train eval')
ax1.plot(e_iter, e_val, 's-', color='#d62728', markersize=4, linewidth=1.5, label='Val eval')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss (cross-entropy)')
ax1.set_title('Finetune Loss — quality_v3 (75M tokens)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

best_ft_bpb = min(e_bpb)
best_ft_iter = e_iter[list(e_bpb).index(best_ft_bpb)]

ax2.plot(e_iter, e_bpb, 'o-', color='#1f77b4', markersize=5, linewidth=1.8)
ax2.axhline(y=best_ft_bpb, color='red', linestyle='--', alpha=0.6,
            label=f'Best: {best_ft_bpb:.3f} (iter {best_ft_iter})')
ax2.axhline(y=5.434, color='gray', linestyle=':', alpha=0.5, label='Base: 5.434')
ax2.scatter([best_ft_iter], [best_ft_bpb], color='red', s=60, zorder=5)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('BPB')
ax2.set_title(f'Finetune BPB — from Base 5.434 to {best_ft_bpb:.3f}')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "fig_finetune.png", dpi=150, bbox_inches='tight')
plt.savefig(OUT / "fig_finetune.pdf", bbox_inches='tight')
plt.close()
print(f"  {len(evals)} evals, BPB {best_ft_bpb:.3f}..{max(e_bpb):.3f}")

# ============================================================
# Figure 3: Parameter breakdown (EXACT from model)
# ============================================================
# Measured: total 299,290,629
# codebook=131,072,000, controller=167,890,944
# attn=98,304, memory=65,536, ln=131,072, thought=24,580, out=8,193
print("Plotting parameter breakdown...")

# Group small params into "Other (attn + memory + LN + thought + output)"
other_total = 98_304 + 65_536 + 131_072 + 24_580 + 8_193  # 327,685

components = [
    ('Controller\n(8 × FFN)', 167_890_944, '#ff7f0e'),
    ('Codebook\n(32K × 4096 bipolar)', 131_072_000, '#1f77b4'),
    ('Other\n(attn + mem + LN)', other_total, '#2ca02c'),
]

sizes = [c[1] for c in components]
labels_with_size = [f'{c[0]}\n{c[1]/1e6:.1f}M' if c[1] > 1e6 else f'{c[0]}\n{c[1]/1e3:.0f}K'
                    for c in components]
colors = [c[2] for c in components]

fig, ax = plt.subplots(figsize=(9, 7))
wedges, texts, autotexts = ax.pie(
    sizes, labels=None,
    autopct=lambda p: f'{p:.1f}%' if p > 1 else f'{p:.3f}%',
    colors=colors, startangle=90,
    textprops={'fontsize': 11},
    pctdistance=0.72,
)
for t in autotexts:
    t.set_color('white')
    t.set_fontweight('bold')
    t.set_fontsize(12)

# Legend instead of labels on the pie
ax.legend(wedges, labels_with_size, loc='center left',
          bbox_to_anchor=(1.05, 0.5), fontsize=11, frameon=False)

ax.set_title(f'HDC-Brain v14.1 Parameter Breakdown (299M total)',
             fontsize=13, pad=30)
# Note about binary codebook
ax.text(0, -1.25,
        'Codebook is bipolar (±1) — 1 bit/param at inference.\n'
        'Effective inference size: codebook 16 MB + controller (int8) 168 MB ≈ 184 MB',
        ha='center', fontsize=9, style='italic', color='gray')
plt.tight_layout()
plt.savefig(OUT / "fig_params.png", dpi=150, bbox_inches='tight')
plt.savefig(OUT / "fig_params.pdf", bbox_inches='tight')
plt.close()

# ============================================================
# Figure 4: Attention parameter comparison (linear scale, clear)
# ============================================================
print("Plotting attention comparison...")
# For hdc_dim=4096, n_heads=4, head_dim=1024:
# Standard transformer attention: Q + K + V + O = 4 × (4096×4096) = 67,108,864
# Our binding attention: 4 heads × 3 (bv_q, bv_k, bv_v) × 1024 head_dim = 12,288
transformer_attn = 4 * 4096 * 4096  # 67M
our_attn = 4 * 3 * 1024  # 12K
reduction = transformer_attn / our_attn

fig, ax = plt.subplots(figsize=(8, 5))
models = ['Standard Transformer\n(QKV + O projections)', 'HDC-Brain\n(Binding vectors)']
values = [transformer_attn, our_attn]
colors_bar = ['#d62728', '#2ca02c']

bars = ax.bar(models, values, color=colors_bar, width=0.55)
ax.set_ylabel('Parameters per Attention Layer')
ax.set_yscale('log')
ax.set_title(f'Attention Layer Parameters: {reduction:.0f}× Reduction\n'
             f'(HDC-Brain: 4 heads × 3 binding vectors × head_dim=1024)',
             fontsize=12)

# Annotations
ax.text(0, transformer_attn * 1.5,
        f'{transformer_attn/1e6:.1f}M\nparams',
        ha='center', fontsize=13, fontweight='bold')
ax.text(1, our_attn * 1.5,
        f'{our_attn/1e3:.1f}K\nparams',
        ha='center', fontsize=13, fontweight='bold')

ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(1e3, 2e8)
plt.tight_layout()
plt.savefig(OUT / "fig_attn_comparison.png", dpi=150, bbox_inches='tight')
plt.savefig(OUT / "fig_attn_comparison.pdf", bbox_inches='tight')
plt.close()

print("\nAll figures regenerated!")
print(f"Output dir: {OUT}")
