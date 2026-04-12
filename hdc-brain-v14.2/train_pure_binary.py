"""
HDC-Brain v14.2 — полностью бинарное обучение с отпечатками.

НОЛЬ backprop. НОЛЬ градиентов. НОЛЬ float оптимизации.

Обучение через бинарное голосование с ТОЧНЫМ blame assignment:
  1. Forward: сохраняем отпечаток (sign) до и после каждого подслоя
  2. Ошибка: сравниваем выход с target
  3. Blame: для каждого подслоя — он приблизил к цели или отдалил?
  4. Голосование: только виноватые подслои получают давление
  5. Отпечатки: удаляем после использования или кешируем
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

PROJECT = Path(__file__).parent.parent
V14_DIR = PROJECT / "hdc-brain-v14"
DATA_DIR = PROJECT / "hdc-brain-v15"
SP_MODEL = V14_DIR / "bpe_ru_16k.model"


class PureBinaryModel:
    """Полностью бинарная модель с отпечатками для blame assignment."""

    def __init__(self, vocab_size, dim=1024, max_seq_len=128,
                 n_blocks=4, n_rules=32, n_filters=8,
                 max_thoughts=1, cb_threshold=5.0, block_threshold=0.5,
                 device="cpu"):
        self.V = vocab_size
        self.D = dim
        self.max_seq_len = max_seq_len
        self.n_blocks = n_blocks
        self.n_rules = n_rules
        self.n_filters = n_filters
        self.max_thoughts = max_thoughts
        self.cb_threshold = cb_threshold
        self.block_threshold = block_threshold
        self.device = device

        def rand_bipolar(*shape):
            return (torch.randint(0, 2, shape, device=device) * 2 - 1).to(torch.int8)

        # === Бинарные параметры ===
        self.codebook = rand_bipolar(vocab_size, dim)
        self.cb_pressure = torch.zeros(vocab_size, dim, device=device)

        # Per-block
        self.attn_bv_q = [rand_bipolar(dim) for _ in range(n_blocks)]
        self.attn_bv_k = [rand_bipolar(dim) for _ in range(n_blocks)]
        self.attn_bv_v = [rand_bipolar(dim) for _ in range(n_blocks)]

        self.logic_rules = [rand_bipolar(n_rules, dim) for _ in range(n_blocks)]
        self.logic_query = [rand_bipolar(dim) for _ in range(n_blocks)]

        self.ctrl_filters = [rand_bipolar(n_filters, dim) for _ in range(n_blocks)]

        # Давление — по одному буферу на каждый параметр
        self.pressure = {}
        for b in range(n_blocks):
            self.pressure[f"b{b}_bv_q"] = torch.zeros(dim, device=device)
            self.pressure[f"b{b}_bv_k"] = torch.zeros(dim, device=device)
            self.pressure[f"b{b}_bv_v"] = torch.zeros(dim, device=device)
            self.pressure[f"b{b}_logic_rules"] = torch.zeros(n_rules, dim, device=device)
            self.pressure[f"b{b}_logic_query"] = torch.zeros(dim, device=device)
            self.pressure[f"b{b}_ctrl_filters"] = torch.zeros(n_filters, dim, device=device)

        self.memory_decay = [0.9] * n_blocks
        self.total_flips = 0
        self.blame_cache = {}  # кеш отпечатков для похожих входов

    # ================================================================
    # Forward с отпечатками
    # ================================================================

    def _sign(self, x):
        s = torch.sign(x)
        return torch.where(s == 0, torch.ones_like(s), s)

    def _cyclic_position(self, x, T):
        B, _, D = x.shape
        positions = torch.arange(T, device=x.device)
        indices = (torch.arange(D, device=x.device).unsqueeze(0) - positions.unsqueeze(1)) % D
        return x.gather(2, indices.unsqueeze(0).expand(B, -1, -1))

    def _ema_memory(self, x, block_idx):
        B, T, D = x.shape
        decay = self.memory_decay[block_idx]
        powers = decay ** torch.arange(T, device=x.device, dtype=torch.float32)
        kernel = powers.flip(0).view(1, 1, T).expand(D, 1, T)
        x_t = x.transpose(1, 2)
        x_padded = F.pad(x_t, (T - 1, 0))
        return F.conv1d(x_padded, kernel, groups=D).transpose(1, 2)

    def _attention(self, x, block_idx):
        B, T, D = x.shape
        Q = x * self.attn_bv_q[block_idx].float()
        K = x * self.attn_bv_k[block_idx].float()
        V = x * self.attn_bv_v[block_idx].float()
        scores = torch.matmul(Q, K.transpose(-2, -1)) * (D ** -0.5)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, -1e9)
        attn = torch.sigmoid(scores * 4.0).masked_fill(~causal, 0.0)
        return torch.matmul(attn, V)

    def _logic(self, x, block_idx):
        B, T, D = x.shape
        rules = self.logic_rules[block_idx].float()
        qbind = self.logic_query[block_idx].float()
        query = x * qbind
        sim = torch.matmul(query, rules.T) * (D ** -0.5)
        weights = torch.sigmoid(sim * 4.0)
        bindings = x.unsqueeze(2) * rules.unsqueeze(0).unsqueeze(0)
        output = (weights.unsqueeze(-1) * bindings).sum(dim=2)
        return x * 0.7 + output * 0.3

    def _controller(self, x, block_idx):
        filters = self.ctrl_filters[block_idx].float()
        bound = x.unsqueeze(2) * filters.unsqueeze(0).unsqueeze(0)
        mixed = bound.mean(dim=2)
        return x * 0.5 + mixed * 0.5

    @torch.no_grad()
    def forward_with_fingerprints(self, idx):
        """Forward pass, сохраняя отпечаток ДО и ПОСЛЕ каждого подслоя.

        Возвращает: h, fingerprints
        fingerprints = список (block_idx, sublayer_name, before_sign, after_sign)
        """
        B, T = idx.shape
        x = self.codebook[idx].float()
        x = self._cyclic_position(x, T)

        fingerprints = []

        for b in range(self.n_blocks):
            # Memory
            before = self._sign(x).to(torch.int8)
            mem = self._ema_memory(x, b)
            x = self._sign(x + mem)
            after = self._sign(x).to(torch.int8)
            fingerprints.append((b, "memory", before, after))

            # Attention
            before = after
            attn = self._attention(x, b)
            x = self._sign(x + attn)
            after = self._sign(x).to(torch.int8)
            fingerprints.append((b, "attention", before, after))

            # Logic
            before = after
            x = self._logic(x, b)
            x = self._sign(x)
            after = self._sign(x).to(torch.int8)
            fingerprints.append((b, "logic", before, after))

            # Controller
            before = after
            x = self._controller(x, b)
            x = self._sign(x)
            after = self._sign(x).to(torch.int8)
            fingerprints.append((b, "controller", before, after))

        return x, fingerprints

    @torch.no_grad()
    def predict(self, idx):
        h, fingerprints = self.forward_with_fingerprints(idx)
        logits = torch.matmul(h, self.codebook.float().T)
        return logits, h, fingerprints

    # ================================================================
    # Blame assignment через отпечатки
    # ================================================================

    @torch.no_grad()
    def vote_codebook_only(self, h, target_ids):
        """Голосование ТОЛЬКО для кодбука. Блоки не трогаем."""
        B, T, D = h.shape
        h_sign = self._sign(h).to(torch.int8)
        target_codes = self.codebook[target_ids].to(torch.int8)
        disagree = (h_sign != target_codes)
        confidence = torch.abs(h).clamp(max=5.0)
        confidence = confidence / (confidence.mean() + 1e-8)
        vote = h_sign.float() * confidence * disagree.float()
        self.cb_pressure.index_add_(0, target_ids.reshape(-1), vote.reshape(-1, D))

    @torch.no_grad()
    def apply_codebook_flips(self):
        """Переключить только кодбук."""
        fp = self.cb_pressure > self.cb_threshold
        fn = self.cb_pressure < -self.cb_threshold
        n = int(fp.sum() + fn.sum())
        if n > 0:
            self.codebook[fp] = 1
            self.codebook[fn] = -1
            self.cb_pressure[fp | fn] = 0
            self.total_flips += n
        return n

    @torch.no_grad()
    def vote_with_blame(self, h, target_ids, fingerprints):
        """Точное голосование: каждый подслой получает голос
        пропорционально тому, насколько он ОТДАЛИЛ от цели.

        1. target_sign = sign(codebook[target]) — какими биты должны быть
        2. Для каждого подслоя: before → after
           - Если before был ближе к target чем after → подслой НАВРЕДИЛ
           - Навредившим подслоям — голос на исправление
        """
        B, T, D = h.shape

        # Целевые биты
        target_codes = self.codebook[target_ids].to(torch.int8)  # (B, T, D)

        # === Голосование для кодбука (как раньше — прямая связь) ===
        h_sign = self._sign(h).to(torch.int8)
        disagree = (h_sign != target_codes)
        confidence = torch.abs(h).clamp(max=5.0)
        confidence = confidence / (confidence.mean() + 1e-8)
        vote = h_sign.float() * confidence * disagree.float()
        self.cb_pressure.index_add_(0, target_ids.reshape(-1), vote.reshape(-1, D))

        # === Blame для каждого подслоя ===
        for block_idx, sublayer, before, after in fingerprints:
            # Побитовое сравнение: какие биты подслой ИСПОРТИЛ
            # before_match = биты before совпадающие с target
            # after_match = биты after совпадающие с target
            before_match = (before == target_codes)  # (B, T, D) bool
            after_match = (after == target_codes)     # (B, T, D) bool

            # Подслой НАВРЕДИЛ конкретному биту если:
            # before совпадал с target, а after — нет
            damaged_bits = before_match & ~after_match  # (B, T, D) bool

            # Подслой ПОМОГ конкретному биту если:
            # before не совпадал, а after совпал
            helped_bits = ~before_match & after_match  # (B, T, D) bool

            n_damaged = damaged_bits.float().sum()
            n_helped = helped_bits.float().sum()

            if n_damaged <= n_helped:
                continue  # подслой больше помог чем навредил

            # Направление коррекции: для повреждённых бит — к target
            # target_codes = что должно быть, after = что есть
            # correction = target - after: +2 если нужно из -1 в +1, -2 если наоборот
            correction = (target_codes.float() - after.float())  # (B, T, D)
            # Усредняем только по повреждённым битам
            correction = correction * damaged_bits.float()
            correction = correction.sum(dim=(0, 1)) / (damaged_bits.float().sum(dim=(0, 1)) + 1e-8)  # (D,)

            # Сила: доля повреждённых бит
            damage_ratio = n_damaged / (n_damaged + n_helped + 1e-8)

            if sublayer == "attention":
                vote = correction * damage_ratio
                self.pressure[f"b{block_idx}_bv_q"] += vote
                self.pressure[f"b{block_idx}_bv_k"] += vote
                self.pressure[f"b{block_idx}_bv_v"] += vote

            elif sublayer == "logic":
                vote = correction * damage_ratio
                self.pressure[f"b{block_idx}_logic_query"] += vote
                for r in range(self.n_rules):
                    rule_vote = self.logic_rules[block_idx][r].float() * vote
                    self.pressure[f"b{block_idx}_logic_rules"][r] += rule_vote

            elif sublayer == "controller":
                vote = correction * damage_ratio
                for f in range(self.n_filters):
                    filt_vote = self.ctrl_filters[block_idx][f].float() * vote
                    self.pressure[f"b{block_idx}_ctrl_filters"][f] += filt_vote

    @torch.no_grad()
    def apply_flips(self):
        total = 0

        # Кодбук — высокий порог (много голосов)
        fp = self.cb_pressure > self.cb_threshold
        fn = self.cb_pressure < -self.cb_threshold
        n = fp.sum() + fn.sum()
        if n > 0:
            self.codebook[fp] = 1
            self.codebook[fn] = -1
            self.cb_pressure[fp | fn] = 0
            total += int(n)

        # Все блочные параметры
        param_map = {}
        for b in range(self.n_blocks):
            param_map[f"b{b}_bv_q"] = self.attn_bv_q[b]
            param_map[f"b{b}_bv_k"] = self.attn_bv_k[b]
            param_map[f"b{b}_bv_v"] = self.attn_bv_v[b]
            param_map[f"b{b}_logic_query"] = self.logic_query[b]
            param_map[f"b{b}_logic_rules"] = self.logic_rules[b]
            param_map[f"b{b}_ctrl_filters"] = self.ctrl_filters[b]

        for key, param in param_map.items():
            press = self.pressure[key]
            fp = press > self.block_threshold
            fn = press < -self.block_threshold
            n = fp.sum() + fn.sum()
            if n > 0:
                param[fp] = 1
                param[fn] = -1
                press[fp | fn] = 0
                total += int(n)

        self.total_flips += total
        return total

    @torch.no_grad()
    def decay_pressure(self, factor=0.99):
        self.cb_pressure *= factor
        for key in self.pressure:
            self.pressure[key] *= factor

    @torch.no_grad()
    def generate(self, start_ids, sp, max_len=50, temperature=0.8, top_k=40):
        idx = start_ids.clone()
        for _ in range(max_len):
            context = idx[:, -self.max_seq_len:]
            h, _ = self.forward_with_fingerprints(context)
            logits = torch.matmul(h[:, -1:, :], self.codebook.float().T)
            logits = logits[:, 0, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return sp.decode(idx[0].tolist())

    def blame_stats(self):
        """Показать какие подслои больше всего под давлением."""
        stats = {}
        for key, press in self.pressure.items():
            stats[key] = float(press.abs().mean())
        return stats


# ================================================================
# Training loop
# ================================================================

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
    x = np.stack([data[s: s + seq_len] for s in starts])
    y = np.stack([data[s + 1: s + seq_len + 1] for s in starts])
    return (torch.from_numpy(x).long().to(device),
            torch.from_numpy(y).long().to(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR / "data_ru_wiki_50m.txt"))
    parser.add_argument("--max-tokens", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--n-rules", type=int, default=32)
    parser.add_argument("--n-filters", type=int, default=8)
    parser.add_argument("--cb-threshold", type=float, default=5.0)
    parser.add_argument("--block-threshold", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--flip-every", type=int, default=10)
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

    model = PureBinaryModel(
        vocab_size=vocab_size, dim=args.dim, max_seq_len=128,
        n_blocks=args.n_blocks, n_rules=args.n_rules,
        n_filters=args.n_filters, cb_threshold=args.cb_threshold,
        block_threshold=args.block_threshold, device=device,
    )

    n_binary = (vocab_size * args.dim +
                args.n_blocks * (3 * args.dim + args.n_rules * args.dim +
                                 args.dim + args.n_filters * args.dim))
    print(f"\n{'='*60}")
    print(f"PURE BINARY — ZERO backprop — fingerprint blame")
    print(f"  Binary params: {n_binary:,} bits = {n_binary/8/1024:.0f} KB")
    print(f"  Float params: 0")
    print(f"  D={args.dim} blocks={args.n_blocks}")
    print(f"  cb_threshold={args.cb_threshold} block_threshold={args.block_threshold} batch={args.batch_size}")
    print(f"{'='*60}\n")

    t_start = time.time()
    best_hamming = float("inf")

    # === ФАЗА 1: Кодбук (блоки заморожены) ===
    phase1_steps = args.steps // 2
    phase2_steps = args.steps - phase1_steps
    print(f"PHASE 1: Codebook only ({phase1_steps} steps, blocks frozen)")
    print(f"PHASE 2: Codebook + blocks ({phase2_steps} steps)")
    print()

    phase = 1
    for step in range(1, args.steps + 1):
        if step == phase1_steps + 1:
            phase = 2
            print(f"\n{'='*40}")
            print(f"PHASE 2: Unlocking blocks")
            print(f"{'='*40}\n")

        x, y = get_batch(train_data, args.batch_size, 128, device)

        with torch.no_grad():
            logits, h, fingerprints = model.predict(x)

            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
            pred_ids = logits.argmax(-1)
            acc = (pred_ids == y).float().mean().item()

            pred_codes = model.codebook[pred_ids]
            true_codes = model.codebook[y]
            hamming = (pred_codes != true_codes).float().sum(-1).mean().item()

            if phase == 1:
                # Только кодбук — прямое голосование
                model.vote_codebook_only(h, y)
            else:
                # Кодбук + blame для блоков
                model.vote_with_blame(h, y, fingerprints)

            del fingerprints

        if step % args.flip_every == 0:
            if phase == 1:
                model.apply_codebook_flips()
                model.cb_pressure *= 0.99
            else:
                model.apply_flips()
                model.decay_pressure()

        if step % args.eval_every == 0 or step == 1:
            elapsed = time.time() - t_start
            tok_per_sec = step * args.batch_size * 128 / elapsed
            bp_max = max(p.abs().max().item() for p in model.pressure.values()) if phase == 2 else 0
            print(f"[P{phase}] step {step:>5}/{args.steps}  "
                  f"loss={loss:.3f}  hamming={hamming:.0f}  acc={acc:.3f}  "
                  f"tok/s={tok_per_sec:.0f}  flips={model.total_flips}  "
                  f"time={elapsed:.0f}s")

            if hamming < best_hamming:
                best_hamming = hamming

        if step % args.gen_every == 0:
            if phase == 2:
                bs = model.blame_stats()
                top_blamed = sorted(bs.items(), key=lambda x: -x[1])[:3]
                print(f"  blamed: {[(k, f'{v:.3f}') for k, v in top_blamed]}")

            for prompt in ["Россия это", "В начале было"]:
                ids = sp.encode(prompt)
                start = torch.tensor([ids], dtype=torch.long, device=device)
                text = model.generate(start, sp, max_len=30)
                print(f"  [{prompt}] → {text[:100]}")
            print()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Best hamming: {best_hamming:.0f}")
    print(f"Total flips: {model.total_flips}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
