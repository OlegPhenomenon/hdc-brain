"""
Бинарный градиентный спуск — v2. Один слой. Чистая формула.

Модель: только кодбук. Без блоков. Без float. Без backprop.

Forward:
  контекст = Σ permute(codebook[token_i], pos_i)   — bundle входных токенов
  sign(контекст) → предсказание                    — бинаризация
  dot(предсказание, codebook) → logits              — ближайший токен

Обучение:
  идеал = sign(контекст)                            — каким должен быть код цели
  ошибка = идеал != codebook[цель]                  — какие биты неправильные
  счётчик += голоса                                 — накапливаем по батчу
  flip когда |счётчик| > порог                      — переключаем
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

PROJECT = Path(__file__).parent.parent
V14_DIR = PROJECT / "hdc-brain-v14"
DATA_DIR = PROJECT / "hdc-brain-v15"
SP_MODEL = V14_DIR / "bpe_ru_16k.model"


class BinaryLM:
    """Языковая модель = один кодбук. Ноль блоков. Ноль float."""

    def __init__(self, vocab_size, dim=1024, context_len=5, threshold=3.0,
                 device="cpu"):
        self.V = vocab_size
        self.D = dim
        self.context_len = context_len
        self.threshold = threshold
        self.device = device

        # Кодбук: (V, D) int8 {-1, +1}
        self.codebook = (torch.randint(0, 2, (vocab_size, dim), device=device) * 2 - 1).to(torch.int8)

        # Счётчик голосов для каждого бита: (V, D) int16
        # Положительный = "хотят +1", отрицательный = "хотят -1"
        self.votes = torch.zeros(vocab_size, dim, device=device, dtype=torch.int16)

        self.total_flips = 0
        self.total_examples = 0

    @torch.no_grad()
    def _context_vector(self, token_ids):
        """Bundle контекста: сумма permute(code, pos).

        token_ids: (B, T) — последние context_len токенов
        return: (B, D) float — контекстный вектор
        """
        B, T = token_ids.shape
        D = self.D
        ctx = torch.zeros(B, D, device=self.device, dtype=torch.float32)

        for pos in range(T):
            code = self.codebook[token_ids[:, pos]].float()  # (B, D)
            # Cyclic permute на pos позиций
            code = torch.roll(code, shifts=int(pos), dims=-1)
            ctx += code

        return ctx

    @torch.no_grad()
    def predict(self, token_ids):
        """Предсказать следующий токен.

        token_ids: (B, T) — контекст
        return: logits (B, V), context (B, D)
        """
        ctx = self._context_vector(token_ids[:, -self.context_len:])
        # Logits = dot product контекста с каждым кодом
        logits = torch.matmul(ctx, self.codebook.float().T)  # (B, V)
        return logits, ctx

    @torch.no_grad()
    def train_step(self, context_ids, target_ids):
        """Один шаг бинарного обучения.

        context_ids: (B, T)
        target_ids: (B,) — правильный следующий токен

        Формула:
          контекст = bundle(permute(codes))
          идеал = sign(контекст)           — каким должен быть код цели
          ошибка = идеал != codebook[цель]  — где биты неправильные
          голос: +1 к "сделать как идеал" для ошибочных бит
        """
        B = target_ids.shape[0]
        ctx = self._context_vector(context_ids[:, -self.context_len:])

        # Идеальный код для каждого target в этом батче
        ideal = torch.sign(ctx).to(torch.int8)  # (B, D)
        ideal = torch.where(ideal == 0, torch.ones_like(ideal), ideal)

        # Текущий код target-токенов
        current = self.codebook[target_ids]  # (B, D) int8

        # Где расхождение
        wrong = (ideal != current)  # (B, D) bool

        # Голос: для каждого неправильного бита — "сделай как ideal"
        # ideal=+1, current=-1, wrong=True → голос +1 (хотим +1)
        # ideal=-1, current=+1, wrong=True → голос -1 (хотим -1)
        vote = ideal.short() * wrong.short()  # (B, D) int16, 0 где правильно

        # Аккумулируем голоса по target_ids
        # Нужен scatter_add: для каждого target в батче добавить его vote
        for b in range(B):
            tid = target_ids[b]
            self.votes[tid] += vote[b]

        self.total_examples += B

        # Метрики
        hamming = wrong.float().sum(-1).mean().item()  # средняя ошибка по батчу
        return hamming

    @torch.no_grad()
    def apply_flips(self):
        """Переключить биты где накопилось достаточно голосов."""
        flip_pos = self.votes > self.threshold   # большинство хочет +1
        flip_neg = self.votes < -self.threshold  # большинство хочет -1

        n_flips = int(flip_pos.sum() + flip_neg.sum())

        if n_flips > 0:
            self.codebook[flip_pos] = 1
            self.codebook[flip_neg] = -1
            self.votes[flip_pos | flip_neg] = 0
            self.total_flips += n_flips

        return n_flips

    @torch.no_grad()
    def decay_votes(self, factor=0.9):
        """Затухание старых голосов."""
        self.votes = (self.votes.float() * factor).to(torch.int16)

    @torch.no_grad()
    def generate(self, start_ids, sp, max_len=30, temperature=1.0):
        """Генерация текста."""
        ids = start_ids.clone()
        for _ in range(max_len):
            ctx = ids[:, -self.context_len:]
            logits, _ = self.predict(ctx)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return sp.decode(ids[0].tolist())


def load_data(path, sp, max_tokens=0):
    print(f"Loading {path}...")
    text = Path(path).read_text(encoding="utf-8")
    ids = sp.encode(text)
    if max_tokens > 0:
        ids = ids[:max_tokens]
    print(f"  {len(ids):,} tokens")
    return np.array(ids, dtype=np.int32)


def get_batch(data, batch_size, context_len, device):
    """Батч: context_len токенов → следующий токен."""
    max_start = len(data) - context_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[s: s + context_len] for s in starts])
    y = np.array([data[s + context_len] for s in starts])
    return (torch.from_numpy(x).long().to(device),
            torch.from_numpy(y).long().to(device))


@torch.no_grad()
def evaluate(model, data, batch_size, device, n_batches=20):
    """Оценка: accuracy + средний hamming."""
    total_correct = 0
    total_hamming = 0.0
    total = 0

    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, model.context_len, device)
        logits, ctx = model.predict(x)
        pred = logits.argmax(-1)
        total_correct += (pred == y).sum().item()
        total += y.shape[0]

        # Hamming: идеал vs текущий код
        ideal = torch.sign(ctx).to(torch.int8)
        ideal = torch.where(ideal == 0, torch.ones_like(ideal), ideal)
        current = model.codebook[y]
        hamming = (ideal != current).float().sum(-1).mean().item()
        total_hamming += hamming

    return total_correct / total, total_hamming / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR / "data_ru_wiki_50m.txt"))
    parser.add_argument("--max-tokens", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--context-len", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--flip-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--gen-every", type=int, default=1000)
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

    model = BinaryLM(
        vocab_size=vocab_size, dim=args.dim,
        context_len=args.context_len, threshold=args.threshold,
        device=device,
    )

    size_kb = (vocab_size * args.dim) / 8 / 1024
    print(f"\n{'='*60}")
    print(f"BINARY GRADIENT DESCENT — one layer, pure formula")
    print(f"  Codebook: {vocab_size} x {args.dim} = {size_kb:.0f} KB")
    print(f"  Context: {args.context_len} tokens")
    print(f"  Threshold: {args.threshold}")
    print(f"  Batch: {args.batch_size}")
    print(f"  NO blocks, NO float weights, NO backprop")
    print(f"{'='*60}\n")

    t_start = time.time()
    best_acc = 0.0

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.context_len, device)
        hamming = model.train_step(x, y)

        if step % args.flip_every == 0:
            n_flips = model.apply_flips()
            model.decay_votes(0.9)

        if step % args.eval_every == 0 or step == 1:
            acc, val_hamming = evaluate(model, val_data, args.batch_size, device)
            elapsed = time.time() - t_start
            examples_per_sec = model.total_examples / elapsed
            vote_max = model.votes.abs().max().item()

            print(f"step {step:>5}/{args.steps}  "
                  f"acc={acc:.4f}  hamming={val_hamming:.0f}  "
                  f"flips={model.total_flips:,}  vote_max={vote_max}  "
                  f"ex/s={examples_per_sec:.0f}  time={elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc

        if step % args.gen_every == 0:
            for prompt in ["Россия это", "В начале было", "Москва"]:
                ids = sp.encode(prompt)
                start = torch.tensor([ids], dtype=torch.long, device=device)
                text = model.generate(start, sp, max_len=20)
                print(f"  [{prompt}] → {text[:80]}")
            print()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Total flips: {model.total_flips:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
