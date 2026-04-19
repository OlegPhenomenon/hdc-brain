"""
Честный бенчмарк: backprop vs binary GD.

Одинаковые условия:
  - Одинаковые данные (2M токенов)
  - Одинаковое время (7 минут каждый)
  - Одинаковый eval (acc, t10, t50, t100, ppl)
  - Одинаковая архитектура (D=1024, 4 блока, ctrl=256)
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(__file__).parent.parent
V14_DIR = PROJECT / "hdc-brain-v14"
DATA_DIR = PROJECT / "hdc-brain-v15"
SP_MODEL = V14_DIR / "bpe_ru_16k.model"

# Импортируем v14 модель
sys.path.insert(0, str(V14_DIR))
from hdc_brain_v14 import HDCBrainV14

# Импортируем бинарную модель
from train_binary_v14 import BinaryV14Model, BinaryParam


# ============================================================
# Общие утилиты
# ============================================================

def load_data(path, sp, max_tokens=0):
    print(f"Loading {path}...")
    text = Path(path).read_text(encoding="utf-8")
    ids = sp.encode(text)
    if max_tokens > 0:
        ids = ids[:max_tokens]
    print(f"  {len(ids):,} tokens")
    return np.array(ids, dtype=np.int32)


def get_batch_seq(data, batch_size, seq_len, device):
    """Батч для sequence модели (x → y сдвинуты на 1)."""
    max_start = len(data) - seq_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[s: s + seq_len] for s in starts])
    y = np.stack([data[s + 1: s + seq_len + 1] for s in starts])
    return (torch.from_numpy(x).long().to(device),
            torch.from_numpy(y).long().to(device))


def get_batch_next(data, batch_size, context_len, device):
    """Батч для next-token модели (context → 1 token)."""
    max_start = len(data) - context_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[s: s + context_len] for s in starts])
    y = np.array([data[s + context_len] for s in starts])
    return (torch.from_numpy(x).long().to(device),
            torch.from_numpy(y).long().to(device))


# ============================================================
# Единый eval для обоих
# ============================================================

@torch.no_grad()
def evaluate_backprop(model, data, batch_size, seq_len, device, n_batches=30):
    """Eval для backprop модели (предсказывает последний токен)."""
    model.eval()
    total_correct = 0
    total_t10 = 0
    total_t50 = 0
    total_t100 = 0
    total = 0
    total_loss = 0.0

    for _ in range(n_batches):
        x, y = get_batch_seq(data, batch_size, seq_len, device)
        logits, loss = model(x, y, n_thoughts=2)

        # Берём предсказание последнего токена
        last_logits = logits[:, -1, :]  # (B, V)
        last_target = y[:, -1]           # (B,)

        pred = last_logits.argmax(-1)
        total_correct += (pred == last_target).sum().item()

        for k in [10, 50, 100]:
            topk = last_logits.topk(k, dim=-1).indices
            hits = (topk == last_target.unsqueeze(1)).any(dim=1).sum().item()
            if k == 10: total_t10 += hits
            elif k == 50: total_t50 += hits
            else: total_t100 += hits

        total += last_target.shape[0]
        total_loss += loss.item() * last_target.shape[0]

    model.train()
    acc = total_correct / total
    ppl = min(math.exp(total_loss / total), 99999)
    return acc, total_t10/total, total_t50/total, total_t100/total, ppl


@torch.no_grad()
def evaluate_binary(model, data, batch_size, device, n_batches=30):
    """Eval для бинарной модели."""
    total_correct = 0
    total_t10 = 0
    total_t50 = 0
    total_t100 = 0
    total = 0
    total_loss = 0.0

    for _ in range(n_batches):
        x, y = get_batch_next(data, batch_size, 10, device)
        logits, _, _ = model.forward(x)

        pred = logits.argmax(-1)
        total_correct += (pred == y).sum().item()

        for k in [10, 50, 100]:
            topk = logits.topk(k, dim=-1).indices
            hits = (topk == y.unsqueeze(1)).any(dim=1).sum().item()
            if k == 10: total_t10 += hits
            elif k == 50: total_t50 += hits
            else: total_t100 += hits

        total += y.shape[0]

        log_probs = torch.log_softmax(logits.clamp(-100, 100), dim=-1)
        target_lp = log_probs[torch.arange(y.shape[0], device=device), y]
        loss_val = -target_lp.sum().item()
        if not math.isnan(loss_val) and not math.isinf(loss_val):
            total_loss += loss_val
        else:
            total_loss += math.log(model.V) * y.shape[0]

    acc = total_correct / total
    ppl = min(math.exp(total_loss / total), 99999)
    return acc, total_t10/total, total_t50/total, total_t100/total, ppl


def generate_text(model, sp, device, prompt="Россия это", max_len=20,
                  is_backprop=True):
    ids = sp.encode(prompt)
    start = torch.tensor([ids], dtype=torch.long, device=device)

    if is_backprop:
        model.eval()
        out = model.generate(start, max_len=max_len, temperature=1.0, top_k=40,
                             n_thoughts=2)
        text = sp.decode(out[0].tolist())
        model.train()
    else:
        text = model.generate(start, sp, max_len=max_len, top_k=40)

    return text[:100]


# ============================================================
# Бенчмарк 1: Backprop
# ============================================================

def run_backprop(train_data, val_data, vocab_size, device, time_limit=420):
    print(f"\n{'='*60}")
    print(f"BENCHMARK: v14 with BACKPROP")
    print(f"  Time limit: {time_limit}s ({time_limit/60:.0f} min)")
    print(f"{'='*60}\n")

    model = HDCBrainV14(
        vocab_size=vocab_size,
        hdc_dim=1024,
        max_seq_len=10,
        n_blocks=4,
        controller_dim=256,
        dropout=0.0,
        max_thoughts=2,
        use_checkpoint=False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()

    t_start = time.time()
    step = 0
    results = []

    while True:
        elapsed = time.time() - t_start
        if elapsed > time_limit:
            break

        step += 1
        x, y = get_batch_seq(train_data, 64, 10, device)
        logits, loss = model(x, y, n_thoughts=2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 500 == 0 or step == 1:
            acc, t10, t50, t100, ppl = evaluate_backprop(
                model, val_data, 64, 10, device)
            elapsed = time.time() - t_start
            print(f"  step {step:>6}  acc={acc:.4f}  t10={t10:.4f}  "
                  f"t50={t50:.4f}  t100={t100:.4f}  ppl={ppl:.0f}  "
                  f"loss={loss.item():.3f}  time={elapsed:.0f}s")
            results.append((step, acc, t10, t50, t100, ppl, elapsed))

    # Финальный eval
    acc, t10, t50, t100, ppl = evaluate_backprop(
        model, val_data, 64, 10, device)
    elapsed = time.time() - t_start
    print(f"\n  FINAL: step={step}  acc={acc:.4f}  t10={t10:.4f}  "
          f"t50={t50:.4f}  t100={t100:.4f}  ppl={ppl:.0f}  time={elapsed:.0f}s")

    # Генерация
    for prompt in ["Россия это", "В начале было"]:
        text = generate_text(model, spm.SentencePieceProcessor(
            model_file=str(SP_MODEL)), device, prompt, is_backprop=True)
        print(f"  [{prompt}] → {text}")

    return {'acc': acc, 't10': t10, 't50': t50, 't100': t100,
            'ppl': ppl, 'steps': step, 'time': elapsed}


# ============================================================
# Бенчмарк 2: Binary GD (послойный)
# ============================================================

def run_binary(train_data, val_data, vocab_size, device, time_limit=420):
    print(f"\n{'='*60}")
    print(f"BENCHMARK: v14 with BINARY GRADIENT DESCENT (layerwise)")
    print(f"  Time limit: {time_limit}s ({time_limit/60:.0f} min)")
    print(f"{'='*60}\n")

    model = BinaryV14Model(
        vocab_size=vocab_size, hdc_dim=1024,
        n_blocks=4, controller_dim=256,
        max_thoughts=2, max_seq_len=10,
        device=device,
    )

    n_params = model.total_params()
    print(f"  Params: {n_params:,} (all binary)")

    t_start = time.time()
    step = 0
    phases = ['codebook'] + [f'block{i}' for i in range(4)]
    time_per_phase = time_limit / len(phases)

    for phase_idx, phase_name in enumerate(phases):
        phase_start = time.time()
        print(f"\n  --- Phase {phase_idx}: {phase_name} ---")

        # Сбросить голоса
        for _, param in model._all_params():
            param.votes.zero_()

        while True:
            phase_elapsed = time.time() - phase_start
            if phase_elapsed > time_per_phase:
                break

            step += 1
            x, y = get_batch_next(train_data, 64, 10, device)

            # Forward с audit
            logits, audit, h_out = model.forward(x, collect_audit=True)
            target_codes = model.codebook.data[y].float()
            output_sign = torch.sign(h_out)
            output_sign = torch.where(output_sign == 0,
                                       torch.ones_like(output_sign), output_sign)
            error = (output_sign != target_codes.sign())
            direction = target_codes - output_sign
            B = y.shape[0]

            if phase_name == 'codebook':
                ctx_len = min(x.shape[1], 10)
                for i in range(ctx_len):
                    tok_ids = x[:, -(i + 1)]
                    tok_codes = model.codebook.data[tok_ids].float()
                    weight = 0.5 ** i
                    disagree = (target_codes != tok_codes)
                    vote = tok_codes * disagree.float() * weight
                    model.codebook.votes.index_add_(0, y, vote)
            else:
                block_idx = int(phase_name.replace('block', ''))
                block_params = set(model.blocks[block_idx][k]
                                   for k in model.blocks[block_idx])
                for name, param, contribution in audit:
                    if param not in block_params:
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

            if step % 5 == 0:
                if phase_name == 'codebook':
                    model.codebook.apply_flips(5.0)
                    model.codebook.decay_votes(0.9)
                else:
                    block_idx = int(phase_name.replace('block', ''))
                    for pname, param in model.blocks[block_idx].items():
                        param.apply_flips(5.0)
                        param.decay_votes(0.9)

            if step % 500 == 0 or step == 1:
                acc, t10, t50, t100, ppl = evaluate_binary(
                    model, val_data, 64, device)
                elapsed = time.time() - t_start
                print(f"  step {step:>6}  acc={acc:.4f}  t10={t10:.4f}  "
                      f"t50={t50:.4f}  t100={t100:.4f}  ppl={ppl:.0f}  "
                      f"[{phase_name}]  time={elapsed:.0f}s")

    # Финальный eval
    acc, t10, t50, t100, ppl = evaluate_binary(model, val_data, 64, device)
    elapsed = time.time() - t_start
    print(f"\n  FINAL: step={step}  acc={acc:.4f}  t10={t10:.4f}  "
          f"t50={t50:.4f}  t100={t100:.4f}  ppl={ppl:.0f}  time={elapsed:.0f}s")

    # Генерация
    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    for prompt in ["Россия это", "В начале было"]:
        text = generate_text(model, sp, device, prompt, is_backprop=False)
        print(f"  [{prompt}] → {text}")

    return {'acc': acc, 't10': t10, 't50': t50, 't100': t100,
            'ppl': ppl, 'steps': step, 'time': elapsed}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR / "data_ru_wiki_50m.txt"))
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--time-limit", type=int, default=420,
                        help="Seconds per benchmark (default 420 = 7 min)")
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

    # === Бенчмарк 1: Backprop ===
    bp_results = run_backprop(train_data, val_data, vocab_size, device,
                              args.time_limit)

    # === Бенчмарк 2: Binary GD ===
    bin_results = run_binary(train_data, val_data, vocab_size, device,
                             args.time_limit)

    # === Сравнение ===
    print(f"\n{'='*60}")
    print(f"COMPARISON (same data, same time = {args.time_limit}s)")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'Backprop':>12} {'Binary GD':>12} {'Winner':>10}")
    print(f"{'-'*46}")

    for metric in ['acc', 't10', 't50', 't100']:
        bp_val = bp_results[metric]
        bin_val = bin_results[metric]
        winner = "backprop" if bp_val > bin_val else "binary" if bin_val > bp_val else "tie"
        print(f"{metric:<12} {bp_val:>12.4f} {bin_val:>12.4f} {winner:>10}")

    bp_ppl = bp_results['ppl']
    bin_ppl = bin_results['ppl']
    winner = "backprop" if bp_ppl < bin_ppl else "binary" if bin_ppl < bp_ppl else "tie"
    print(f"{'ppl':<12} {bp_ppl:>12.0f} {bin_ppl:>12.0f} {winner:>10}")

    print(f"{'steps':<12} {bp_results['steps']:>12} {bin_results['steps']:>12}")
    print(f"{'time':<12} {bp_results['time']:>11.0f}s {bin_results['time']:>11.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
