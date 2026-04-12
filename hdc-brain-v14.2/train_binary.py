"""
HDC-Brain v14.2 — гибридное обучение.

Codebook: Binary Voting (без градиентов, без float)
Blocks:   Backprop (обычный gradient descent)

Это proof of concept: можно ли обучить 88% параметров без backprop?
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

from hdc_brain_v14_2 import HDCBrainV14_2, CONFIGS, _ste_sign
from binary_codebook import BinaryVotingCodebook

PROJECT = Path(__file__).parent.parent
V14_DIR = PROJECT / "hdc-brain-v14"
DATA_DIR = PROJECT / "hdc-brain-v15"
SP_MODEL = V14_DIR / "bpe_ru_16k.model"


class HybridModel(torch.nn.Module):
    """Модель с бинарным кодбуком + float блоками.

    Кодбук: BinaryVotingCodebook (int8, обучается голосованием)
    Блоки: HDCBlock (float32, обучаются backprop)
    """

    def __init__(self, vocab_size, hdc_dim=1024, max_seq_len=128,
                 n_blocks=4, controller_dim=256, n_rules=32,
                 dropout=0.1, max_thoughts=3, threshold=5.0):
        super().__init__()
        self.hdc_dim = hdc_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Бинарный кодбук — НЕ участвует в backprop
        self.binary_cb = BinaryVotingCodebook(vocab_size, hdc_dim, threshold)

        # Блоки с BinaryController — минимум float
        from hdc_brain_v14_2 import HDCBlock, ThoughtLoop
        self.blocks = torch.nn.ModuleList([
            HDCBlock(hdc_dim, controller_dim, n_rules, n_filters=8,
                     dropout=dropout, use_binary_controller=True)
            for _ in range(n_blocks)
        ])
        self.thought_loop = ThoughtLoop(hdc_dim, max_thoughts)
        self.output_ln = torch.nn.LayerNorm(hdc_dim)
        self.output_scale = torch.nn.Parameter(torch.tensor(1.0))

    def _cyclic_position(self, x):
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        indices = (torch.arange(D, device=x.device).unsqueeze(0) - positions.unsqueeze(1)) % D
        return x.gather(2, indices.unsqueeze(0).expand(B, -1, -1))

    def forward(self, idx, targets=None, n_thoughts=None):
        B, T = idx.shape

        # Encode через бинарный кодбук (без градиентов для кодбука)
        tokens = self.binary_cb.encode(idx)  # (B, T, D) float {-1, +1}
        tokens = self._cyclic_position(tokens)

        # Process через float блоки (с градиентами)
        h = self.thought_loop(tokens, self.blocks, n_thoughts)

        # Output
        h = self.output_ln(h)

        # Logits через бинарный кодбук
        cb_float = self.binary_cb.codebook.float()
        logits = F.linear(h, cb_float) * self.output_scale

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss, h  # h нужен для binary voting

    @torch.no_grad()
    def generate(self, start_ids, max_len=100, temperature=0.8, top_k=40):
        idx = start_ids.clone()
        for _ in range(max_len):
            context = idx[:, -self.max_seq_len:]
            logits, _, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def load_data(path, sp, max_tokens=0):
    print(f"Loading {path}...")
    text = Path(path).read_text(encoding="utf-8")
    ids = sp.encode(text)
    if max_tokens > 0:
        ids = ids[:max_tokens]
    print(f"  {len(ids):,} tokens")
    return np.array(ids, dtype=np.int32)


def get_batch(data, batch_size, seq_len, device):
    max_start = len(data) - seq_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[s : s + seq_len] for s in starts])
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return (torch.from_numpy(x).long().to(device),
            torch.from_numpy(y).long().to(device))


@torch.no_grad()
def evaluate(model, data, batch_size, seq_len, device, n_batches=10):
    model.eval()
    total_loss = 0.0
    total_hamming = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, seq_len, device)
        logits, loss, h = model(x, y)
        total_loss += loss.item()

        # Hamming метрика: предсказанные биты vs target биты
        pred_ids = logits.argmax(-1)
        pred_codes = model.binary_cb.codebook[pred_ids]  # (B, T, D)
        true_codes = model.binary_cb.codebook[y]          # (B, T, D)
        hamming = (pred_codes != true_codes).float().sum(-1).mean()
        total_hamming += hamming.item()

    model.train()
    return total_loss / n_batches, total_hamming / n_batches


def generate_sample(model, sp, device, prompt="Россия это", max_len=50):
    model.eval()
    ids = sp.encode(prompt)
    start_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(start_ids, max_len=max_len)
    text = sp.decode(out[0].tolist())
    model.train()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="small", choices=["tiny", "small"])
    parser.add_argument("--data", default=str(DATA_DIR / "data_ru_wiki_50m.txt"))
    parser.add_argument("--max-tokens", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-thoughts", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4, help="LR for blocks only")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Flip threshold for binary codebook (higher=careful)")
    parser.add_argument("--vote-every", type=int, default=1,
                        help="Vote on codebook every N steps")
    parser.add_argument("--flip-every", type=int, default=10,
                        help="Apply flips every N steps")
    parser.add_argument("--pressure-decay", type=float, default=0.99)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--gen-every", type=int, default=500)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    vocab_size = sp.get_piece_size()

    all_ids = load_data(args.data, sp, args.max_tokens)
    split = int(len(all_ids) * 0.9)
    train_data, val_data = all_ids[:split], all_ids[split:]
    print(f"Train: {len(train_data):,}  Val: {len(val_data):,}")

    cfg = CONFIGS[args.config]
    model = HybridModel(
        vocab_size=vocab_size,
        hdc_dim=cfg["hdc_dim"],
        max_seq_len=cfg["max_seq_len"],
        n_blocks=cfg["n_blocks"],
        controller_dim=cfg["controller_dim"],
        n_rules=cfg["n_rules"],
        dropout=cfg["dropout"],
        max_thoughts=cfg["max_thoughts"],
        threshold=args.threshold,
    ).to(device)

    seq_len = cfg["max_seq_len"]
    n_block_params = sum(p.numel() for p in model.blocks.parameters())
    n_cb = model.binary_cb.vocab_size * model.binary_cb.dim

    print(f"\n{'='*60}")
    print(f"HDC-Brain v14.2 — HYBRID training")
    print(f"  Codebook: {n_cb:,} bits (Binary Voting, threshold={args.threshold})")
    print(f"  Blocks:   {n_block_params:,} float params (Backprop, lr={args.lr})")
    print(f"  D={cfg['hdc_dim']} blocks={cfg['n_blocks']} thoughts={cfg['max_thoughts']}")
    print(f"  batch={args.batch_size} seq={seq_len} steps={args.steps}")
    print(f"{'='*60}\n")

    # Optimizer только для float параметров (blocks + thought_loop + output)
    float_params = list(model.blocks.parameters()) + \
                   list(model.thought_loop.parameters()) + \
                   list(model.output_ln.parameters()) + \
                   [model.output_scale]
    optimizer = torch.optim.AdamW(float_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    best_val = float("inf")
    t_start = time.time()

    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, seq_len, device)

        # Forward
        logits, loss, h = model(x, y, n_thoughts=args.n_thoughts)

        # Backprop — ТОЛЬКО для float блоков
        loss.backward()
        torch.nn.utils.clip_grad_norm_(float_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Binary Voting — обновление кодбука
        if step % args.vote_every == 0:
            # h уже detached от графа (мы сделали backward)
            # Берём h перед output_ln для голосования
            with torch.no_grad():
                h_for_vote = h.detach()
                model.binary_cb.vote(h_for_vote, y)

        if step % args.flip_every == 0:
            n_flips = model.binary_cb.apply_flips()
            model.binary_cb.decay_pressure(args.pressure_decay)

        # Eval
        if step % args.eval_every == 0 or step == 1:
            val_loss, val_hamming = evaluate(model, val_data, args.batch_size, seq_len, device)
            bpb = val_loss / math.log(2)
            elapsed = time.time() - t_start
            tok_per_sec = step * args.batch_size * seq_len / elapsed
            cb_stats = model.binary_cb.stats()

            print(f"step {step:>5}/{args.steps}  "
                  f"train={loss.item():.3f}  val={val_loss:.3f}  "
                  f"BPB={bpb:.2f}  hamming={val_hamming:.0f}  "
                  f"tok/s={tok_per_sec:.0f}  "
                  f"flips={cb_stats['total_flips']}  "
                  f"pressure={cb_stats['pressure_mean']:.3f}")

        if step % args.gen_every == 0:
            cb_stats = model.binary_cb.stats()
            print(f"  CB stats: flips={cb_stats['total_flips']} "
                  f"pressure_max={cb_stats['pressure_max']:.1f} "
                  f"near_flip={cb_stats['pressure_above_half']} "
                  f"ones={cb_stats['ones_ratio']:.3f}")

            for prompt in ["Россия это", "Москва является", "В начале было"]:
                text = generate_sample(model, sp, device, prompt)
                print(f"  [{prompt}] → {text[:120]}")
            print()

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "binary_codebook": model.binary_cb.codebook.clone(),
                    "blocks_state": {k: v for k, v in model.state_dict().items()
                                     if not k.startswith("binary_cb")},
                    "config": cfg,
                    "step": step,
                    "val_loss": val_loss,
                }, Path(__file__).parent / "best_hybrid.pt")

    elapsed = time.time() - t_start
    cb_stats = model.binary_cb.stats()
    print(f"\n{'='*60}")
    print(f"Training complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Best val loss: {best_val:.3f} (BPB: {best_val/math.log(2):.2f})")
    print(f"Total codebook flips: {cb_stats['total_flips']}")
    print(f"{'='*60}")

    print("\nFinal generation:")
    for prompt in ["Россия это", "Все люди", "В начале", "Кошка сидит"]:
        text = generate_sample(model, sp, device, prompt, max_len=80)
        print(f"  [{prompt}] → {text[:150]}")


if __name__ == "__main__":
    main()
