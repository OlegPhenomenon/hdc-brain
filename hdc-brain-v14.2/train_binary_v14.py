"""
Бинарный градиентный спуск на архитектуре v14.

Берём ТОЧНУЮ архитектуру v14 (Memory + Attention + Controller + ThoughtLoop).
Заменяем ТОЛЬКО метод обучения: вместо backprop → бинарный градиент.

Формула Олега:
  ideal = что параметр ДОЛЖЕН быть (исходя из входа и желаемого выхода)
  error = ideal XOR current_weights (где веса неправильные)
  накапливаем голоса, флипаем когда порог превышен

Все веса: бинарные {-1, +1}.
Промежуточные вычисления: float (matmul, sigmoid, softmax) — это ОК,
  бинарные именно ОБУЧАЕМЫЕ ПАРАМЕТРЫ.
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


# ============================================================
# Модель: точная копия v14, но все параметры бинарные
# ============================================================

class BinaryParam:
    """Обёртка для бинарного параметра с голосованием."""
    def __init__(self, shape, device="cpu"):
        self.data = (torch.randint(0, 2, shape, device=device) * 2 - 1).to(torch.int8)
        self.votes = torch.zeros(shape, device=device)
        self.total_flips = 0

    def float(self):
        return self.data.float()

    def apply_flips(self, threshold):
        fp = self.votes > threshold
        fn = self.votes < -threshold
        n = int(fp.sum() + fn.sum())
        if n > 0:
            self.data[fp] = 1
            self.data[fn] = -1
            self.votes[fp | fn] = 0
            self.total_flips += n
        return n

    def decay_votes(self, factor):
        self.votes *= factor


class BinaryV14Model:
    """Архитектура v14, все параметры бинарные.

    Структура:
      Codebook (V, D)
      N × Block:
        Memory: mass_w (D,), decay_w (D,)
        Attention: bv_q (D,), bv_k (D,), bv_v (D,)
        Controller: down (inner, D), up (D, inner)
      ThoughtLoop: thought_pos (K, D)
      Output: scale = 1/sqrt(D) (фиксированный)
    """

    def __init__(self, vocab_size, hdc_dim=1024, n_blocks=4,
                 controller_dim=256, max_thoughts=3,
                 max_seq_len=64, device="cpu"):
        self.V = vocab_size
        self.D = hdc_dim
        self.n_blocks = n_blocks
        self.controller_dim = controller_dim
        self.max_thoughts = max_thoughts
        self.max_seq_len = max_seq_len
        self.device = device
        self.scale = hdc_dim ** -0.5

        # === Кодбук ===
        self.codebook = BinaryParam((vocab_size, hdc_dim), device)

        # === Блоки ===
        self.blocks = []
        for _ in range(n_blocks):
            block = {
                # Memory
                'mass_w': BinaryParam((hdc_dim,), device),
                'decay_w': BinaryParam((hdc_dim,), device),
                # Attention
                'bv_q': BinaryParam((hdc_dim,), device),
                'bv_k': BinaryParam((hdc_dim,), device),
                'bv_v': BinaryParam((hdc_dim,), device),
                # Controller
                'down': BinaryParam((controller_dim, hdc_dim), device),
                'up': BinaryParam((hdc_dim, controller_dim), device),
            }
            self.blocks.append(block)

        # === ThoughtLoop ===
        self.thought_pos = BinaryParam((max_thoughts, hdc_dim), device)

    def _all_params(self):
        """Итератор по всем BinaryParam."""
        yield ('codebook', self.codebook)
        for i, block in enumerate(self.blocks):
            for name, param in block.items():
                yield (f'block{i}.{name}', param)
        yield ('thought_pos', self.thought_pos)

    def _cyclic_permute(self, x):
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        indices = (torch.arange(D, device=x.device).unsqueeze(0) -
                   positions.unsqueeze(1)) % D
        return x.gather(2, indices.unsqueeze(0).expand(B, -1, -1))

    @torch.no_grad()
    def _block_forward(self, x, block, audit):
        """Один блок: Memory → Attention → Controller.

        Каждая операция записывает в audit: какой параметр какие биты тронул.
        audit — список записей: (param_ref, contribution)
        contribution: (B, D) float — вклад этого параметра в каждый бит выхода
        """
        B, T, D = x.shape
        h_in = x[:, -1, :]  # (B, D) — отслеживаем последнюю позицию

        # === Memory (EMA) ===
        mass_w = block['mass_w'].float()
        decay_w = block['decay_w'].float()
        mass = torch.sigmoid((x * mass_w).sum(-1, keepdim=True))
        decay = torch.sigmoid((x * decay_w).sum(-1, keepdim=True))
        weighted = mass * x

        context = torch.zeros_like(x)
        state = torch.zeros(B, 1, D, device=self.device)
        for t in range(x.shape[1]):
            d = decay[:, t:t+1, :]
            state = state * d + weighted[:, t:t+1, :]
            context[:, t:t+1, :] = state

        mem_contribution = context[:, -1, :]  # (B, D) — что memory добавила
        # mass_w влияет на амплитуду: contribution[j] ≈ mass * x[j]
        # Записываем: mass_w тронул биты пропорционально mem_contribution
        audit.append(('mass_w', block['mass_w'], mem_contribution * mass_w))
        audit.append(('decay_w', block['decay_w'], mem_contribution * decay_w))

        x = (x + context).clamp(-50, 50)

        # === Attention ===
        bv_q = block['bv_q'].float()
        bv_k = block['bv_k'].float()
        bv_v = block['bv_v'].float()

        Q = x * bv_q
        K = x * bv_k
        V = x * bv_v

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        causal = torch.tril(torch.ones(T, T, device=self.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float('-inf'))
        attn_w = torch.sigmoid(scores * 4.0)
        attn_w = attn_w.masked_fill(~causal, 0.0)
        attn_out = torch.matmul(attn_w, V)

        # bv_v[j] напрямую контролирует бит j выхода attention
        # Если bv_v[j]=-1, бит j перевёрнут. Запись: bv_v тронул каждый бит.
        attn_last = attn_out[:, -1, :]  # (B, D)
        audit.append(('bv_v', block['bv_v'], attn_last))
        # bv_q и bv_k влияют косвенно через attention weights
        audit.append(('bv_q', block['bv_q'], attn_last * 0.3))
        audit.append(('bv_k', block['bv_k'], attn_last * 0.3))

        x = (x + attn_out).clamp(-50, 50)

        # === Controller ===
        down_w = block['down'].float()
        up_w = block['up'].float()

        h = F.gelu(F.linear(x, down_w))   # (B, T, inner)
        ctrl_out = F.linear(h, up_w)       # (B, T, D)
        ctrl_last = ctrl_out[:, -1, :]     # (B, D)

        # down[i,j]: бит j входа → нейрон i скрытого слоя → бит выхода
        # up[j,i]: нейрон i → бит j выхода
        # Для каждого бита j выхода: up[j,:] определяет вклад
        # Записываем: down и up тронули биты пропорционально ctrl_out
        audit.append(('down', block['down'], ctrl_last))
        audit.append(('up', block['up'], ctrl_last))

        x = (x + ctrl_out).clamp(-50, 50)
        return x

    @torch.no_grad()
    def forward(self, ids, collect_audit=False):
        """Forward pass с audit trail.

        ids: (B, T)
        collect_audit: собирать лог кто какие биты тронул
        return: logits (B, V), audit_log, h_final
        """
        B, T = ids.shape
        audit = []

        # Encode
        h = self.codebook.data[ids].float()  # (B, T, D)
        h = self._cyclic_permute(h)

        # Проход через блоки (каждый пишет в audit)
        for block in self.blocks:
            h = self._block_forward(h, block, audit if collect_audit else [])

        # Дополнительные thoughts
        for t in range(1, self.max_thoughts):
            thought_input = h + self.thought_pos.data[t].float()
            thought = thought_input
            for block in self.blocks:
                thought = self._block_forward(
                    thought, block, audit if collect_audit else [])
            h = h + 0.5 * (thought - h)

        # Output
        h_out = h[:, -1, :]  # (B, D)
        h_norm = h_out / (h_out.norm(dim=-1, keepdim=True) + 1e-8)
        cb_norm = self.codebook.float()
        cb_norm = cb_norm / (cb_norm.norm(dim=-1, keepdim=True) + 1e-8)
        logits = F.linear(h_norm, cb_norm) / 0.1

        return logits, audit, h_out

    @torch.no_grad()
    def train_step(self, context_ids, target_ids):
        """Бинарный градиентный спуск через audit trail.

        1. Forward собирает audit: каждый параметр записывает
           какие биты он тронул и свой вклад (contribution)
        2. На выходе: error = какие биты неправильные
        3. Идём по audit: если параметр тронул бит который оказался
           неправильным → голос на этот параметр
        """
        B = target_ids.shape[0]

        # === Forward с audit ===
        logits, audit, h_out = self.forward(context_ids, collect_audit=True)

        # === Ошибка на выходе ===
        target_codes = self.codebook.data[target_ids].float()  # (B, D)
        output_sign = torch.sign(h_out)
        output_sign = torch.where(output_sign == 0,
                                   torch.ones_like(output_sign), output_sign)
        # error[j] = True если бит j неправильный
        error = (output_sign != target_codes.sign())  # (B, D) bool
        # direction: куда нужно двигать каждый бит (+2 или -2 или 0)
        direction = target_codes - output_sign  # (B, D)

        # === Обучение кодбука: weighted pairs (проверено, работает) ===
        ctx_len = min(context_ids.shape[1], 10)
        decay = 0.5
        for i in range(ctx_len):
            tok_ids = context_ids[:, -(i + 1)]
            tok_codes = self.codebook.data[tok_ids].float()
            weight = decay ** i
            disagree = (target_codes != tok_codes)
            vote = tok_codes * disagree.float() * weight
            self.codebook.votes.index_add_(0, target_ids, vote)

        # === Обучение блоков: идём по audit trail ===
        # Каждая запись: (name, param_ref, contribution)
        # contribution: (B, D) — вкл��д этого параметра в каждый бит
        #
        # Логика: если параметр внёс вклад в бит j, и бит j неправильный,
        # то пара��етр виноват. Голос = direction[j] * sign(contribution[j])
        # Т.е. "исправь свой вклад в направлении правильного ответа"

        error_float = error.float()  # (B, D)

        for name, param, contribution in audit:
            # contribution: (B, D) — вклад в каждый бит последней позиции

            if name in ('down', 'up'):
                # Матричные пара��етры: contribution (B, D) показывает
                # общий вклад, но параметр — матрица (inner, D) или (D, inner)
                # Простое правило: голосуе�� на ВСЕ элементы параметра
                # пропорционально ошибке, н��рмализовано по размеру
                error_signal = (direction * torch.sign(contribution)).mean(dim=0)  # (D,)
                rows, cols = param.data.shape
                if cols == self.D:
                    # down: (inner, D) — error по D, broadcast по inner
                    vote = error_signal.unsqueeze(0).expand(rows, cols)
                else:
                    # up: (D, inner) — error по D, broadcast по inner
                    vote = error_signal.unsqueeze(1).expand(rows, cols)
                param.votes += vote / (rows * B)
            else:
                # Векторные параметры (bv_q, bv_k, bv_v, mass_w, decay_w):
                # param — вектор (D,)
                # Если contribution[b,j] > 0 и error[b,j] → параметр тронул бит j
                # и бит j неправильный → голос = direction[b,j]
                #
                # Если contribution[b,j] ≈ 0 → параметр не трогал бит j → нет голоса
                touched = (contribution.abs() > 0.1)  # (B, D) — параметр тронул бит?
                guilty = touched & error               # тронул И неправильный
                # Голос: в направлении исправления, только для виновных битов
                vote = (direction * guilty.float()).sum(dim=0)  # (D,)
                param.votes += vote / B

        # Hamming
        hamming = error.float().sum(-1).mean().item()
        return hamming

    @torch.no_grad()
    def apply_flips(self, threshold=5.0):
        total_cb = 0
        total_block = 0
        for name, param in self._all_params():
            n = param.apply_flips(threshold)
            if 'codebook' in name:
                total_cb += n
            else:
                total_block += n
        return total_cb, total_block

    @torch.no_grad()
    def decay_votes(self, factor=0.9):
        for _, param in self._all_params():
            param.decay_votes(factor)

    @torch.no_grad()
    def generate(self, start_ids, sp, max_len=30, temperature=1.0, top_k=40):
        ids = start_ids.clone()
        for _ in range(max_len):
            logits, _, _ = self.forward(ids[:, -self.max_seq_len:])
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return sp.decode(ids[0].tolist())

    def total_params(self):
        total = 0
        for _, param in self._all_params():
            total += param.data.numel()
        return total

    def total_flips(self):
        cb = self.codebook.total_flips
        blk = sum(p.total_flips for _, p in self._all_params() if 'codebook' not in _)
        return cb, blk


# ============================================================
# Data / Eval / Main
# ============================================================

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
        x, y = get_batch(data, batch_size, 10, device)
        logits, _, h_out = model.forward(x)

        pred = logits.argmax(-1)
        total_correct += (pred == y).sum().item()

        for k_val, name in [(10, 'top10'), (50, 'top50'), (100, 'top100')]:
            topk = logits.topk(k_val, dim=-1).indices
            hits = (topk == y.unsqueeze(1)).any(dim=1).sum().item()
            if k_val == 10:
                total_top10 += hits
            elif k_val == 50:
                total_top50 += hits
            else:
                total_top100 += hits

        total += y.shape[0]

        log_probs = torch.log_softmax(logits.clamp(-100, 100), dim=-1)
        target_log_probs = log_probs[torch.arange(y.shape[0], device=device), y]
        loss_val = -target_log_probs.sum().item()
        if not math.isnan(loss_val) and not math.isinf(loss_val):
            total_loss += loss_val
        else:
            total_loss += math.log(model.V) * y.shape[0]  # random baseline

        output_codes = torch.sign(h_out)
        target_codes = model.codebook.data[y].float()
        hamming = (output_codes != target_codes).float().sum(-1).mean().item()
        total_hamming += hamming

    acc = total_correct / total
    t10 = total_top10 / total
    t50 = total_top50 / total
    t100 = total_top100 / total
    avg_loss = total_loss / total
    ppl = min(math.exp(avg_loss), 99999)
    avg_hamming = total_hamming / n_batches

    return acc, t10, t50, t100, ppl, avg_hamming


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR / "data_ru_wiki_50m.txt"))
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--controller-dim", type=int, default=256)
    parser.add_argument("--max-thoughts", type=int, default=2)
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=5.0)
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

    model = BinaryV14Model(
        vocab_size=vocab_size, hdc_dim=args.dim,
        n_blocks=args.n_blocks, controller_dim=args.controller_dim,
        max_thoughts=args.max_thoughts, max_seq_len=args.context_len,
        device=device,
    )

    n_params = model.total_params()
    cb_params = model.codebook.data.numel()
    blk_params = n_params - cb_params

    print(f"\n{'='*60}")
    print(f"BINARY v14 — Oleg's binary gradient descent")
    print(f"  Total: {n_params:,} binary params")
    print(f"  Codebook: {cb_params:,} ({cb_params/n_params*100:.1f}%)")
    print(f"  Blocks: {blk_params:,} ({blk_params/n_params*100:.1f}%)")
    print(f"  D={args.dim} blocks={args.n_blocks} ctrl={args.controller_dim}")
    print(f"  thoughts={args.max_thoughts} context={args.context_len}")
    print(f"  threshold={args.threshold} batch={args.batch_size}")
    print(f"  ZERO BACKPROP. ALL BINARY. Oleg's formula.")
    print(f"{'='*60}\n")

    t_start = time.time()
    best_ppl = 99999.0
    steps_per_phase = args.steps // (1 + args.n_blocks)  # делим шаги между фазами

    # === Послойное обучение: одна фаза — один слой ===
    phases = ['codebook'] + [f'block{i}' for i in range(args.n_blocks)]

    for phase_idx, phase_name in enumerate(phases):
        print(f"\n{'='*60}")
        print(f"PHASE {phase_idx}: обучаем {phase_name}, остальное заморожено")
        print(f"  {steps_per_phase} шагов")
        print(f"{'='*60}")

        # Сбросить голоса перед новой фазой
        for _, param in model._all_params():
            param.votes.zero_()

        phase_start = time.time()

        for step in range(1, steps_per_phase + 1):
            global_step = phase_idx * steps_per_phase + step
            x, y = get_batch(train_data, args.batch_size, args.context_len, device)

            # Forward с audit
            logits, audit, h_out = model.forward(x, collect_audit=True)

            target_codes = model.codebook.data[y].float()
            output_sign = torch.sign(h_out)
            output_sign = torch.where(output_sign == 0,
                                       torch.ones_like(output_sign), output_sign)
            error = (output_sign != target_codes.sign())
            direction = target_codes - output_sign
            B = y.shape[0]

            # === Голосуем ТОЛЬКО на текущую фазу ===
            if phase_name == 'codebook':
                # Weighted pairs для кодбука
                ctx_len = min(x.shape[1], 10)
                for i in range(ctx_len):
                    tok_ids = x[:, -(i + 1)]
                    tok_codes = model.codebook.data[tok_ids].float()
                    weight = 0.5 ** i
                    disagree = (target_codes != tok_codes)
                    vote = tok_codes * disagree.float() * weight
                    model.codebook.votes.index_add_(0, y, vote)
            else:
                # Audit trail для конкретного блока
                block_idx = int(phase_name.replace('block', ''))
                error_float = error.float()

                for name, param, contribution in audit:
                    # Проверяем что это параметр нужного блока
                    # audit содержит записи от ВСЕХ блоков, нам нужен только block_idx
                    # Каждый блок добавляет 7 записей (mass_w, decay_w, bv_v, bv_q, bv_k, down, up)
                    # Записи блока block_idx: индексы [block_idx*7 .. block_idx*7+6]
                    # Но проще: проверить что param — это параметр нужного блока
                    if param not in [model.blocks[block_idx][k]
                                     for k in model.blocks[block_idx]]:
                        continue

                    if name in ('down', 'up'):
                        error_signal = (direction * torch.sign(contribution)).mean(dim=0)
                        rows, cols = param.data.shape
                        if cols == model.D:
                            vote = error_signal.unsqueeze(0).expand(rows, cols)
                        else:
                            vote = error_signal.unsqueeze(1).expand(rows, cols)
                        param.votes += vote / (rows * B)
                    else:
                        touched = (contribution.abs() > 0.1)
                        guilty = touched & error
                        vote = (direction * guilty.float()).sum(dim=0)
                        param.votes += vote / B

            # Flip только для текущей фазы
            if step % args.flip_every == 0:
                if phase_name == 'codebook':
                    model.codebook.apply_flips(args.threshold)
                    model.codebook.decay_votes(0.9)
                else:
                    block_idx = int(phase_name.replace('block', ''))
                    for pname, param in model.blocks[block_idx].items():
                        param.apply_flips(args.threshold)
                        param.decay_votes(0.9)

            # Eval
            if step % args.eval_every == 0 or step == 1:
                acc, t10, t50, t100, ppl, val_hamming = evaluate(
                    model, val_data, args.batch_size, device)
                elapsed = time.time() - phase_start
                ex_per_sec = step * args.batch_size / elapsed
                cb_flips, blk_flips = model.total_flips()

                print(f"  [{phase_name}] step {step:>5}/{steps_per_phase}  "
                      f"acc={acc:.4f}  t10={t10:.4f}  t50={t50:.4f}  t100={t100:.4f}  "
                      f"ppl={ppl:.0f}  hamming={val_hamming:.0f}  "
                      f"cb={cb_flips:,}  blk={blk_flips:,}  "
                      f"ex/s={ex_per_sec:.0f}")

                if ppl < best_ppl:
                    best_ppl = ppl

            # Generate
            if step % args.gen_every == 0 or step == steps_per_phase:
                for prompt in ["Россия это", "В начале было"]:
                    ids = sp.encode(prompt)
                    start = torch.tensor([ids], dtype=torch.long, device=device)
                    text = model.generate(start, sp, max_len=20, top_k=40)
                    print(f"  [{prompt}] → {text[:100]}")
                print()

    elapsed = time.time() - t_start
    cb_flips, blk_flips = model.total_flips()
    print(f"\n{'='*60}")
    print(f"Training complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Best perplexity: {best_ppl:.0f}")
    print(f"Codebook flips: {cb_flips:,}")
    print(f"Block flips: {blk_flips:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
