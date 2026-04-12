"""
HDC-Brain v14.2 — training script for M3 Mac.

Обучает edge-модель на русском тексте.
Цель: увидеть результат за минуты, не за недели.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

from hdc_brain_v14_2 import HDCBrainV14_2, CONFIGS

# === Paths ===
PROJECT = Path(__file__).parent.parent
V14_DIR = PROJECT / "hdc-brain-v14"
DATA_DIR = PROJECT / "hdc-brain-v15"  # русские данные тут
SP_MODEL = V14_DIR / "bpe_ru_16k.model"


def load_data(path: str, sp: spm.SentencePieceProcessor, max_tokens: int = 0):
    """Загрузить текст и токенизировать."""
    print(f"Loading {path}...")
    text = Path(path).read_text(encoding="utf-8")
    print(f"  {len(text):,} chars")
    ids = sp.encode(text)
    if max_tokens > 0:
        ids = ids[:max_tokens]
    print(f"  {len(ids):,} tokens")
    return np.array(ids, dtype=np.int32)


def get_batch(data: np.ndarray, batch_size: int, seq_len: int, device: str):
    """Случайный батч из данных."""
    max_start = len(data) - seq_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[s : s + seq_len] for s in starts])
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return (torch.from_numpy(x).long().to(device),
            torch.from_numpy(y).long().to(device))


def evaluate(model, data, batch_size, seq_len, device, n_batches=10):
    """Средний loss на валидации."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(data, batch_size, seq_len, device)
            _, loss = model(x, y)
            total_loss += loss.item()
    model.train()
    return total_loss / n_batches


def generate_sample(model, sp, device, prompt="Россия это", max_len=50):
    """Сгенерировать текст для визуальной проверки."""
    model.eval()
    ids = sp.encode(prompt)
    start_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(start_ids, max_len=max_len, temperature=0.8, top_k=40)
    text = sp.decode(out[0].tolist())
    model.train()
    return text


def main():
    parser = argparse.ArgumentParser(description="Train HDC-Brain v14.2")
    parser.add_argument("--config", default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--data", default=str(DATA_DIR / "data_ru_wiki_50m.txt"))
    parser.add_argument("--max-tokens", type=int, default=500_000,
                        help="Max training tokens (0=all)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-thoughts", type=int, default=1,
                        help="Thoughts per step (1=fast, 3=quality)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--gen-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    # Flush stdout для фоновых процессов
    sys.stdout.reconfigure(line_buffering=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    vocab_size = sp.get_piece_size()
    print(f"Vocab: {vocab_size}")

    # Load data
    all_ids = load_data(args.data, sp, args.max_tokens)
    split = int(len(all_ids) * 0.9)
    train_data = all_ids[:split]
    val_data = all_ids[split:]
    print(f"Train: {len(train_data):,}  Val: {len(val_data):,}")

    # Model
    cfg = CONFIGS[args.config].copy()
    model = HDCBrainV14_2(vocab_size=vocab_size, **cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    seq_len = cfg["max_seq_len"]

    print(f"\n{'='*60}")
    print(f"HDC-Brain v14.2 [{args.config}]")
    print(f"  Params: {n_params:,}")
    print(f"  D={cfg['hdc_dim']} blocks={cfg['n_blocks']} "
          f"ctrl={cfg['controller_dim']} thoughts={cfg['max_thoughts']}")
    print(f"  seq_len={seq_len} batch={args.batch_size}")
    print(f"  lr={args.lr} steps={args.steps}")
    print(f"{'='*60}\n")

    # Optimizer — codebook learns slower
    codebook_params = [model.codebook]
    other_params = [p for n, p in model.named_parameters() if n != "codebook"]
    optimizer = torch.optim.AdamW([
        {"params": codebook_params, "lr": args.lr * 0.1},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    # Training loop
    best_val = float("inf")
    save_dir = Path(__file__).parent
    t_start = time.time()

    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, seq_len, device)

        _, loss = model(x, y, n_thoughts=args.n_thoughts)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if step % args.eval_every == 0 or step == 1:
            val_loss = evaluate(model, val_data, args.batch_size, seq_len, device)
            bpb = val_loss / math.log(2)
            elapsed = time.time() - t_start
            tok_per_sec = step * args.batch_size * seq_len / elapsed

            print(f"step {step:>5}/{args.steps}  "
                  f"train={loss.item():.3f}  val={val_loss:.3f}  "
                  f"BPB={bpb:.2f}  "
                  f"tok/s={tok_per_sec:.0f}  "
                  f"time={elapsed:.0f}s")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg,
                    "vocab_size": vocab_size,
                    "step": step,
                    "val_loss": val_loss,
                }, save_dir / "best_v14_2.pt")

        if step % args.gen_every == 0:
            # LogicLayer gates
            gates = [f"{torch.sigmoid(b.logic.logic_gate).item():.3f}" for b in model.blocks]
            print(f"  logic gates: [{', '.join(gates)}]")
            # ThoughtLoop gates
            tgates = [f"{torch.sigmoid(g).item():.3f}" for g in model.thought_loop.thought_gates]
            print(f"  thought gates: [{', '.join(tgates)}]")

            for prompt in ["Россия это", "Москва является", "В начале было"]:
                text = generate_sample(model, sp, device, prompt)
                print(f"  [{prompt}] → {text[:120]}")
            print()

        if step % args.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "vocab_size": vocab_size,
                "step": step,
                "val_loss": val_loss,
            }, save_dir / f"checkpoint_step{step}.pt")

    # Final
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Best val loss: {best_val:.3f} (BPB: {best_val/math.log(2):.2f})")
    print(f"{'='*60}")

    # Final generation
    print("\nFinal generation:")
    for prompt in ["Россия это", "Все люди", "В начале", "Кошка сидит"]:
        text = generate_sample(model, sp, device, prompt, max_len=80)
        print(f"  [{prompt}] → {text[:150]}")


if __name__ == "__main__":
    main()
