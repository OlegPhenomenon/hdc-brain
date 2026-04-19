"""
Binary v14 — полностью бинарное обучение со снимками (blame).

Архитектура: v14 упрощённая
  1. Binary Codebook → Cyclic Permutation
  2. N × BinaryBlock (attention через binding + controller)
  3. Output: h @ codebook.T

Обучение: snapshot blame
  1. Forward с сохранением снимков до/после каждого блока
  2. Ошибка на выходе: sign(h) vs target_code
  3. Blame: какой блок изменил биты что оказались неправильными?
  4. Голосование на параметры + флип по порогу

Ноль float весов. Ноль backprop. Ноль градиентов.
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


def random_binary(shape, device="cpu"):
    """Случайный бинарный вектор {-1, +1}."""
    return (torch.randint(0, 2, shape, device=device) * 2 - 1).to(torch.int8)


def sign_nonzero(x):
    """sign() без нулей — нули заменяются на +1."""
    s = torch.sign(x).to(torch.int8)
    return torch.where(s == 0, torch.ones_like(s), s)


class BinaryAttention:
    """Бинарное внимание через binding.

    Q = h * bv_q, K = h * bv_k, V = h * bv_v
    scores = Q @ K.T / sqrt(D)
    weights = softmax(scores, causal)
    out = weights @ V

    Параметры: bv_q, bv_k, bv_v — бинарные {-1, +1}
    """

    def __init__(self, D, device="cpu"):
        self.D = D
        self.device = device

        self.bv_q = random_binary((D,), device)
        self.bv_k = random_binary((D,), device)
        self.bv_v = random_binary((D,), device)

        self.votes_v = torch.zeros(D, device=device)
        self.total_flips = 0

    @torch.no_grad()
    def forward(self, h):
        """h: (B, T, D) int8 → (B, T, D) float (attention output)."""
        h_f = h.float()
        Q = h_f * self.bv_q.float()
        K = h_f * self.bv_k.float()
        V = h_f * self.bv_v.float()

        scale = self.D ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        T = h.shape[1]
        causal = torch.tril(torch.ones(T, T, device=self.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float('-inf'))
        weights = torch.softmax(scores, dim=-1)

        return torch.matmul(weights, V)  # (B, T, D)


class BinaryController:
    """Бинарный контроллер: N filters × binding.

    output = Σ (input * filter_k)  — сумма binding'ов
    Каждый фильтр — бинарный вектор {-1, +1}

    При K фильтров это как "K голосов" за значение каждого бита.
    """

    def __init__(self, D, n_filters=8, device="cpu"):
        self.D = D
        self.n_filters = n_filters
        self.device = device

        self.filters = random_binary((n_filters, D), device)
        self.votes = torch.zeros(n_filters, D, device=device)
        self.total_flips = 0

    @torch.no_grad()
    def forward(self, h):
        """h: (B, T, D) int8 → (B, T, D) float (controller output)."""
        h_f = h.float()
        # (B, T, 1, D) * (K, D) → (B, T, K, D) → sum over K → (B, T, D)
        bound = h_f.unsqueeze(2) * self.filters.float()
        return bound.sum(dim=2)  # (B, T, D)


class BinaryBlock:
    """Attention + Controller + Residual + Sign.

    h → attention(h) → controller(h + attn) → sign(h + result)

    Снимок: сохраняем sign(h) до и после блока.
    """

    def __init__(self, D, n_filters=8, device="cpu"):
        self.attention = BinaryAttention(D, device)
        self.controller = BinaryController(D, n_filters, device)

    @torch.no_grad()
    def forward(self, h):
        """h: (B, T, D) int8 → (B, T, D) int8."""
        attn_out = self.attention.forward(h)   # (B, T, D) float
        h_with_attn = h.float() + attn_out     # residual

        ctrl_out = self.controller.forward(sign_nonzero(h_with_attn))  # (B, T, D) float
        h_new = h.float() + attn_out + ctrl_out  # full residual

        return sign_nonzero(h_new)


class BinaryV14:
    """Полностью бинарная v14 со snapshot blame."""

    def __init__(self, vocab_size, dim=1024, n_blocks=4, n_filters=8,
                 context_len=10, threshold=5.0, decay=0.5, device="cpu"):
        self.V = vocab_size
        self.D = dim
        self.n_blocks = n_blocks
        self.context_len = context_len
        self.threshold = threshold
        self.decay = decay
        self.device = device

        # Кодбук: (V, D) int8
        self.codebook = random_binary((vocab_size, dim), device)
        self.codebook_votes = torch.zeros(vocab_size, dim, device=device)

        # Блоки
        self.blocks = [BinaryBlock(dim, n_filters, device) for _ in range(n_blocks)]

        # Веса позиций для обучения кодбука
        self.pos_weights = torch.tensor(
            [decay ** i for i in range(context_len)], device=device
        )

        self.total_codebook_flips = 0
        self.total_block_flips = 0

    @torch.no_grad()
    def _cyclic_permute(self, x):
        """Cyclic positional encoding."""
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        indices = (torch.arange(D, device=x.device).unsqueeze(0) - positions.unsqueeze(1)) % D
        return x.gather(2, indices.unsqueeze(0).expand(B, -1, -1))

    @torch.no_grad()
    def forward(self, ids):
        """Forward pass с сохранением снимков.

        ids: (B, T)
        return: logits (B, V), snapshots list of (B, T, D) int8
        """
        B, T = ids.shape

        # Encode
        h = self.codebook[ids]  # (B, T, D) int8
        h = self._cyclic_permute(h)

        # Снимки: до каждого блока
        snapshots = [h.clone()]

        for block in self.blocks:
            h = block.forward(h)
            snapshots.append(h.clone())

        # Output: dot product с кодбуком
        logits = torch.matmul(h[:, -1, :].float(), self.codebook.float().T)

        return logits, snapshots

    @torch.no_grad()
    def predict(self, ids):
        """Предсказать без снимков (для eval)."""
        B, T = ids.shape
        h = self.codebook[ids]
        h = self._cyclic_permute(h)
        for block in self.blocks:
            h = block.forward(h)
        logits = torch.matmul(h[:, -1, :].float(), self.codebook.float().T)
        return logits

    @torch.no_grad()
    def train_step(self, context_ids, target_ids):
        """Один шаг обучения через snapshot blame.

        1. Forward со снимками
        2. Ошибка на выходе: какие биты неправильные
        3. Blame блоков: кто изменил биты что оказались неправильными
        4. Голосование на кодбук (weighted pairs) и блоки (blame)
        """
        B, T = context_ids.shape
        ctx_len = min(T, self.context_len)

        # === Forward со снимками ===
        logits, snapshots = self.forward(context_ids)

        # === Ошибка на выходе ===
        target_codes = self.codebook[target_ids]  # (B, D) int8
        output_codes = snapshots[-1][:, -1, :]     # (B, D) int8
        error = (output_codes != target_codes)     # (B, D) bool — где выход неправильный

        # === Blame блоков через снимки ===
        for i, block in enumerate(self.blocks):
            before = snapshots[i][:, -1, :]    # (B, D) int8
            after = snapshots[i + 1][:, -1, :] # (B, D) int8

            # Какие биты этот блок изменил
            changed = (before != after)  # (B, D) bool

            # Плохие изменения: блок перевернул бит, а он оказался неправильным
            bad_change = changed & error      # (B, D)
            # Хорошие изменения: блок перевернул бит, и он правильный
            good_change = changed & ~error    # (B, D)

            # Blame для bv_v (attention value binding):
            # Если бит j плохо изменён → bv_v[j] нужно перевернуть
            # Идеальный bv_v[j] = target[j] * input[j] (чтобы input*bv_v = target)
            ideal_v = (target_codes * before).float()  # (B, D)
            current_v = block.attention.bv_v.float()    # (D,)

            # Голос: где bad_change — давление к ideal, где good — ничего
            # Усредняем по батчу
            blame_mask = bad_change.float()  # (B, D)
            vote_v = (ideal_v * blame_mask).sum(dim=0)  # (D,)
            block.attention.votes_v += vote_v

            # Blame для controller filters:
            # Если бит j плохо изменён → фильтры должны быть "нейтральнее"
            # Простое: net blame → push filters к совпадению с входом
            net_blame = (bad_change.float() - good_change.float()).sum(dim=0)  # (D,)
            for k in range(block.controller.n_filters):
                # Где positive blame → push filter к +1 (identity)
                # Где negative blame → push filter к -1 (transform)
                block.controller.votes[k] += net_blame

        # === Обучение кодбука: weighted pairs (как в v3) ===
        for i in range(ctx_len):
            tok_ids = context_ids[:, -(i + 1)]
            tok_codes = self.codebook[tok_ids]
            weight = self.pos_weights[i].item()

            disagree = (target_codes != tok_codes)
            vote = tok_codes.float() * disagree.float() * weight
            self.codebook_votes.index_add_(0, target_ids, vote)

        # Hamming
        hamming = error.float().sum(-1).mean().item()
        return hamming

    @torch.no_grad()
    def apply_flips(self):
        """Flip bits where votes exceed threshold."""
        # Кодбук
        fp = self.codebook_votes > self.threshold
        fn = self.codebook_votes < -self.threshold
        n_cb = int(fp.sum() + fn.sum())
        if n_cb > 0:
            self.codebook[fp] = 1
            self.codebook[fn] = -1
            self.codebook_votes[fp | fn] = 0
            self.total_codebook_flips += n_cb

        # Блоки
        n_block = 0
        for block in self.blocks:
            # bv_v
            vv = block.attention.votes_v
            fp_v = vv > self.threshold
            fn_v = vv < -self.threshold
            nv = int(fp_v.sum() + fn_v.sum())
            if nv > 0:
                block.attention.bv_v[fp_v] = 1
                block.attention.bv_v[fn_v] = -1
                block.attention.votes_v[fp_v | fn_v] = 0
                n_block += nv

            # Controller filters
            for k in range(block.controller.n_filters):
                fv = block.controller.votes[k]
                fp_f = fv > self.threshold
                fn_f = fv < -self.threshold
                nf = int(fp_f.sum() + fn_f.sum())
                if nf > 0:
                    block.controller.filters[k][fp_f] = 1
                    block.controller.filters[k][fn_f] = -1
                    block.controller.votes[k][fp_f | fn_f] = 0
                    n_block += nf

        self.total_block_flips += n_block
        return n_cb, n_block

    @torch.no_grad()
    def decay_votes(self, factor=0.9):
        self.codebook_votes *= factor
        for block in self.blocks:
            block.attention.votes_v *= factor
            block.controller.votes *= factor

    @torch.no_grad()
    def generate(self, start_ids, sp, max_len=30, temperature=1.0, top_k=40):
        ids = start_ids.clone()
        for _ in range(max_len):
            logits = self.predict(ids)
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return sp.decode(ids[0].tolist())


# === Data loading ===

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
    total_top50 = 0
    total_top100 = 0
    total = 0
    total_loss = 0.0
    total_hamming = 0.0

    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, model.context_len, device)
        logits, snapshots = model.forward(x)

        # Top-K accuracy
        pred = logits.argmax(-1)
        total_correct += (pred == y).sum().item()

        for k, counter_name in [(10, 'top10'), (50, 'top50'), (100, 'top100')]:
            topk = logits.topk(k, dim=-1).indices
            hits = (topk == y.unsqueeze(1)).any(dim=1).sum().item()
            if k == 10:
                total_top10 += hits
            elif k == 50:
                total_top50 += hits
            else:
                total_top100 += hits

        total += y.shape[0]

        # Cross-entropy loss → perplexity
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_probs = log_probs[torch.arange(y.shape[0], device=device), y]
        total_loss += -target_log_probs.sum().item()

        # Hamming: реальный выход модели vs target код
        output_codes = snapshots[-1][:, -1, :]  # (B, D) — выход после всех блоков
        target_codes = model.codebook[y]          # (B, D) — код правильного токена
        hamming = (output_codes != target_codes).float().sum(-1).mean().item()
        total_hamming += hamming

    acc = total_correct / total
    t10 = total_top10 / total
    t50 = total_top50 / total
    t100 = total_top100 / total
    avg_loss = total_loss / total
    ppl = min(math.exp(avg_loss), 99999)  # cap для вывода
    avg_hamming = total_hamming / n_batches

    return acc, t10, t50, t100, ppl, avg_hamming


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR / "data_ru_wiki_50m.txt"))
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--n-filters", type=int, default=8)
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--decay", type=float, default=0.5)
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

    model = BinaryV14(
        vocab_size=vocab_size, dim=args.dim,
        n_blocks=args.n_blocks, n_filters=args.n_filters,
        context_len=args.context_len, threshold=args.threshold,
        decay=args.decay, device=device,
    )

    cb_kb = (vocab_size * args.dim) / 8 / 1024
    block_params = args.n_blocks * (3 * args.dim + args.n_filters * args.dim)
    block_kb = block_params / 8 / 1024

    print(f"\n{'='*60}")
    print(f"BINARY v14 — snapshot blame, zero backprop")
    print(f"  Codebook: {vocab_size} x {args.dim} = {cb_kb:.0f} KB")
    print(f"  Blocks: {args.n_blocks} × (attn + ctrl×{args.n_filters}) = {block_kb:.1f} KB")
    print(f"  Context: {args.context_len}, threshold: {args.threshold}")
    print(f"  Batch: {args.batch_size}, steps: {args.steps}")
    print(f"  ALL BINARY. ZERO FLOAT WEIGHTS. ZERO BACKPROP.")
    print(f"{'='*60}\n")

    t_start = time.time()
    best_acc = 0.0

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.context_len, device)
        hamming = model.train_step(x, y)

        if step % args.flip_every == 0:
            model.apply_flips()
            model.decay_votes(0.9)

        if step % args.eval_every == 0 or step == 1:
            acc, t10, t50, t100, ppl, val_hamming = evaluate(
                model, val_data, args.batch_size, device)
            elapsed = time.time() - t_start
            ex_per_sec = step * args.batch_size / elapsed

            print(f"step {step:>5}/{args.steps}  "
                  f"acc={acc:.4f}  t10={t10:.4f}  t50={t50:.4f}  t100={t100:.4f}  "
                  f"ppl={ppl:.0f}  hamming={val_hamming:.0f}  "
                  f"cb_flips={model.total_codebook_flips:,}  "
                  f"blk_flips={model.total_block_flips:,}  "
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
    print(f"Codebook flips: {model.total_codebook_flips:,}")
    print(f"Block flips: {model.total_block_flips:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
