"""
Бинарный градиентный спуск — v3. Взвешенные пары, не каша.

Модель: кодбук. Предсказание через отдельные сравнения с каждым
токеном контекста, ближний — громче, дальний — тише.

Forward:
  Для каждого кандидата c:
    score(c) = Σ weight[i] × dot(codebook[c], codebook[context_i])
  weight[i] убывает: ближний токен важнее дальнего

Обучение:
  Для каждого контекстного токена отдельно:
    "код target должен быть ближе к коду context[i]"
  Голос с весом: ближний — сильный, дальний — слабый
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
    """Бинарная LM с взвешенными парами."""

    def __init__(self, vocab_size, dim=2048, context_len=10, threshold=5.0,
                 decay=0.5, device="cpu"):
        self.V = vocab_size
        self.D = dim
        self.context_len = context_len
        self.threshold = threshold
        self.decay = decay  # вес = decay^расстояние. 0.5 → [1, 0.5, 0.25, 0.12, ...]
        self.device = device

        # Кодбук: (V, D) int8 {-1, +1}
        self.codebook = (torch.randint(0, 2, (vocab_size, dim), device=device) * 2 - 1).to(torch.int8)

        # Счётчик голосов: (V, D) float
        self.votes = torch.zeros(vocab_size, dim, device=device)

        # Предвычислим веса позиций
        self.pos_weights = torch.tensor(
            [decay ** i for i in range(context_len)],
            device=device
        )  # [1.0, 0.5, 0.25, ...] — ближний = индекс 0

        self.total_flips = 0

    @torch.no_grad()
    def predict(self, context_ids):
        """Предсказать следующий токен через взвешенные пары.

        context_ids: (B, T)
        return: logits (B, V)

        Для каждого кандидата: score = Σ w_i × dot(кандидат, context[i])
        """
        B, T = context_ids.shape
        ctx_len = min(T, self.context_len)

        # Коды контекстных токенов, от ближнего к дальнему
        # context_ids[:, -1] = ближайший, context_ids[:, -2] = предыдущий, ...
        cb_float = self.codebook.float()  # (V, D)

        scores = torch.zeros(B, self.V, device=self.device)

        for i in range(ctx_len):
            # Токен на расстоянии i от конца (0 = самый близкий)
            tok_ids = context_ids[:, -(i + 1)]  # (B,)
            tok_codes = self.codebook[tok_ids].float()  # (B, D)
            weight = self.pos_weights[i]

            # dot(каждый кандидат, этот токен) × вес
            # (B, D) @ (D, V) → (B, V)
            pair_scores = torch.matmul(tok_codes, cb_float.T)
            scores += weight * pair_scores

        return scores

    @torch.no_grad()
    def train_step(self, context_ids, target_ids):
        """Бинарное обучение: только притяжение (дедупликация отдельно).

        target должен быть похож на свои context токены.
        Коллапс предотвращается через dedup_codebook() каждые N шагов.
        """
        B, T = context_ids.shape
        ctx_len = min(T, self.context_len)

        target_codes = self.codebook[target_ids]  # (B, D) int8

        for i in range(ctx_len):
            tok_ids = context_ids[:, -(i + 1)]  # (B,)
            tok_codes = self.codebook[tok_ids]   # (B, D) int8
            weight = self.pos_weights[i].item()

            # Притяжение: сделай target похожим на context[i]
            disagree = (target_codes != tok_codes)  # (B, D) bool
            vote = tok_codes.float() * disagree.float() * weight
            self.votes.index_add_(0, target_ids, vote)

        # Hamming: ближайший контекст vs target
        nearest_codes = self.codebook[context_ids[:, -1]]
        hamming = (target_codes != nearest_codes).float().sum(-1).mean().item()
        return hamming

    @torch.no_grad()
    def dedup_codebook(self, sample_size=500, similarity_threshold=0.85, flip_ratio=0.1):
        """Развести коды которые стали слишком похожи.

        Берём случайную выборку пар, находим слишком похожие,
        рандомизируем часть битов у одного из пары.
        """
        # Случайная выборка токенов
        ids = torch.randint(0, self.V, (sample_size,), device=self.device)
        codes = self.codebook[ids].float()  # (S, D)

        # Попарное сходство (нормализованный dot product)
        sim = torch.matmul(codes, codes.T) / self.D  # (S, S) в [-1, 1]

        # Найти слишком похожие пары (исключая диагональ)
        sim.fill_diagonal_(-2.0)
        too_similar = (sim > similarity_threshold).nonzero()

        n_deduped = 0
        seen = set()
        for pair in too_similar:
            i, j = pair[0].item(), pair[1].item()
            if i in seen or j in seen:
                continue
            seen.add(j)  # рандомизируем второй в паре

            tok_id = ids[j]
            n_flip = int(self.D * flip_ratio)
            flip_positions = torch.randint(0, self.D, (n_flip,), device=self.device)
            self.codebook[tok_id, flip_positions] *= -1
            self.votes[tok_id] = 0  # сбросить голоса
            n_deduped += 1

        return n_deduped

    @torch.no_grad()
    def apply_flips(self):
        fp = self.votes > self.threshold
        fn = self.votes < -self.threshold
        n = int(fp.sum() + fn.sum())
        if n > 0:
            self.codebook[fp] = 1
            self.codebook[fn] = -1
            self.votes[fp | fn] = 0
            self.total_flips += n
        return n

    @torch.no_grad()
    def decay_votes(self, factor=0.9):
        self.votes *= factor

    @torch.no_grad()
    def generate(self, start_ids, sp, max_len=30, temperature=1.0, top_k=40):
        ids = start_ids.clone()
        for _ in range(max_len):
            logits = self.predict(ids)
            logits = logits[:, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = float('-inf')
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
    max_start = len(data) - context_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[s: s + context_len] for s in starts])
    y = np.array([data[s + context_len] for s in starts])
    return (torch.from_numpy(x).long().to(device),
            torch.from_numpy(y).long().to(device))


@torch.no_grad()
def evaluate(model, data, batch_size, device, n_batches=20):
    total_correct = 0
    total_top10 = 0
    total = 0
    total_hamming = 0.0

    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, model.context_len, device)
        logits = model.predict(x)

        pred = logits.argmax(-1)
        total_correct += (pred == y).sum().item()

        # Top-10 accuracy
        top10 = logits.topk(10, dim=-1).indices
        total_top10 += (top10 == y.unsqueeze(1)).any(dim=1).sum().item()

        total += y.shape[0]

        # Hamming: ближайший контекст vs target
        nearest_codes = model.codebook[x[:, -1]]
        target_codes = model.codebook[y]
        hamming = (nearest_codes != target_codes).float().sum(-1).mean().item()
        total_hamming += hamming

    return (total_correct / total, total_top10 / total,
            total_hamming / n_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR / "data_ru_wiki_50m.txt"))
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--decay", type=float, default=0.5,
                        help="Position weight decay: 0.5 = [1, 0.5, 0.25, ...]")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--flip-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--gen-every", type=int, default=2000)
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
        decay=args.decay, device=device,
    )

    weights_str = ", ".join(f"{w:.3f}" for w in model.pos_weights[:5].tolist())
    size_kb = (vocab_size * args.dim) / 8 / 1024

    print(f"\n{'='*60}")
    print(f"BINARY GD v3 — weighted pairs, no bundling")
    print(f"  Codebook: {vocab_size} x {args.dim} = {size_kb:.0f} KB")
    print(f"  Context: {args.context_len}, weights: [{weights_str}, ...]")
    print(f"  Threshold: {args.threshold}, batch: {args.batch_size}")
    print(f"{'='*60}\n")

    t_start = time.time()
    best_acc = 0.0

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.context_len, device)
        hamming = model.train_step(x, y)

        if step % args.flip_every == 0:
            model.apply_flips()
            model.decay_votes(0.9)

        # Дедупликация: развести слишком похожие коды
        if step % 200 == 0:
            n_dedup = model.dedup_codebook()
            if n_dedup > 0:
                print(f"  [dedup] step {step}: разведено {n_dedup} пар")

        if step % args.eval_every == 0 or step == 1:
            acc, top10, val_hamming = evaluate(model, val_data, args.batch_size, device)
            elapsed = time.time() - t_start
            ex_per_sec = step * args.batch_size / elapsed

            print(f"step {step:>5}/{args.steps}  "
                  f"acc={acc:.4f}  top10={top10:.4f}  "
                  f"hamming={val_hamming:.0f}  "
                  f"flips={model.total_flips:,}  "
                  f"ex/s={ex_per_sec:.0f}  time={elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc

        if step % args.gen_every == 0:
            for prompt in ["Россия это", "В начале было", "Москва является"]:
                ids = sp.encode(prompt)
                start = torch.tensor([ids], dtype=torch.long, device=device)
                text = model.generate(start, sp, max_len=20, top_k=40)
                print(f"  [{prompt}] → {text[:100]}")
            print()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Best top-10: (see above)")
    print(f"Total flips: {model.total_flips:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
