"""
CogitLayer experiment: does HDC associative memory speed up training?

A/B comparison:
  A) Mini v14.1 — standard backprop
  B) Mini v14.1 + CogitLayer — backprop + HDC memory that updates on-the-fly

Both train on the same Alpaca instruction data, same time budget.
Eval every 10 minutes. Compare loss, accuracy, generation quality.

Mini architecture (fits MPS without overheating):
  dim=1024, 4 blocks, ctrl=512, 2 heads, ~20M params
"""
import sys
import time
import argparse
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sentencepiece as spm

V14_1_DIR = Path(__file__).parent.parent / "hdc-brain-v14.1"
sys.path.insert(0, str(V14_1_DIR))

SP_MODEL = V14_1_DIR / "bpe_en_32k.model"
ALPACA_BIN = V14_1_DIR / "alpaca.bin"
VAL_BIN = V14_1_DIR / "val.bin"


# ============================================================
# Mini v14.1 architecture (same structure, smaller dims)
# ============================================================

class MiniControllerBlock(nn.Module):
    def __init__(self, dim, inner_dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, inner_dim)
        self.up = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = F.gelu(self.down(h))
        h = self.dropout(self.up(h))
        return x + h


class MiniBindingAttention(nn.Module):
    def __init__(self, dim, n_heads=2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.bv_q = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.bv_k = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.bv_v = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)

    def forward(self, x):
        B, T, D = x.shape
        H, HD = self.n_heads, self.head_dim

        x_h = x.view(B, T, H, HD)
        Q = x_h * torch.sign(self.bv_q)
        K = x_h * torch.sign(self.bv_k)
        V = x_h * torch.sign(self.bv_v)

        Q, K, V = [t.permute(0, 2, 1, 3) for t in (Q, K, V)]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float('-inf'))
        attn = torch.sigmoid(scores * 4.0).masked_fill(~causal, 0.0)
        out = torch.matmul(attn, V)
        return out.permute(0, 2, 1, 3).contiguous().view(B, T, D)


class MiniMemory(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mass_proj = nn.Linear(dim, 1, bias=False)
        self.decay_proj = nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        mass = torch.sigmoid(self.mass_proj(x))
        decay = torch.sigmoid(self.decay_proj(x))
        weighted = mass * x

        # Simple EMA instead of parallel scan (saves memory)
        out = torch.zeros_like(x)
        state = torch.zeros(B, D, device=x.device)
        for t in range(T):
            state = decay[:, t] * state + weighted[:, t]
            out[:, t] = state
        return out


class MiniBlock(nn.Module):
    def __init__(self, dim, ctrl_dim, n_heads=2, dropout=0.1):
        super().__init__()
        self.memory = MiniMemory(dim)
        self.attention = MiniBindingAttention(dim, n_heads)
        self.controller = MiniControllerBlock(dim, ctrl_dim, dropout)
        self.ln_mem = nn.LayerNorm(dim)
        self.ln_attn = nn.LayerNorm(dim)

    def forward(self, x):
        mem = self.memory(x)
        x = self.ln_mem(x + mem)
        attn = self.attention(x)
        x = self.ln_attn(x + attn)
        x = self.controller(x)
        return x


class MiniV14(nn.Module):
    """Mini v14.1: same architecture, smaller dims."""
    def __init__(self, vocab_size, dim=1024, n_blocks=4, ctrl_dim=512,
                 n_heads=2, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.codebook = nn.Parameter(torch.randn(vocab_size, dim) * 0.02)
        self.blocks = nn.ModuleList([
            MiniBlock(dim, ctrl_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.output_ln = nn.LayerNorm(dim)
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def _encode(self, idx):
        real = self.codebook[idx]
        alpha = torch.mean(torch.abs(real), dim=-1, keepdim=True)
        hard = alpha * torch.sign(real)
        hard = torch.where(hard == 0, alpha, hard)
        return (hard - real).detach() + real

    def _position(self, x):
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device)
        dims = torch.arange(D, device=x.device)
        idx = (dims.unsqueeze(0) - pos.unsqueeze(1)) % D
        return x.gather(2, idx.unsqueeze(0).expand(B, -1, -1))

    def forward(self, idx, targets=None, cogit_hint=None):
        tokens = self._encode(idx)
        tokens = self._position(tokens)

        h = tokens

        # Если есть подсказка от CogitLayer — добавляем
        if cogit_hint is not None:
            h = h + cogit_hint * 0.3  # взвешенная подсказка

        for block in self.blocks:
            h = block(h)

        h = self.output_ln(h)
        logits = F.linear(h, self.codebook) * self.output_scale

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss


# ============================================================
# CogitLayer: HDC associative memory that learns on-the-fly
# ============================================================

class CogitLayer:
    """HDC ассоциативная память, обновляемая каждый батч.

    Не участвует в backprop. Чисто бинарные HDC операции.

    Для каждого токена хранит:
      - pos_mem[i][tok]: средний код токенов, идущих через i позиций после tok
      - Это позволяет подсказывать модели "по статистике, после X обычно идёт Y"

    Подсказка формируется как weighted bundle запросов ко всем памятям.
    """
    def __init__(self, vocab_size, dim, n_pos=5, device="cpu"):
        self.V = vocab_size
        self.D = dim
        self.n_pos = n_pos
        self.device = device

        # (n_pos, V, D) — ассоциативные памяти
        self.pos_mem = torch.zeros(n_pos, vocab_size, dim, device=device)
        self.pos_count = torch.zeros(n_pos, vocab_size, device=device)

        # Бинарный кодбук для HDC операций (обновляется из модели)
        self.cb_binary = None

    def update_codebook(self, codebook_param):
        """Синхронизировать бинарный кодбук с текущими весами модели."""
        with torch.no_grad():
            self.cb_binary = torch.sign(codebook_param.data).to(self.device)

    @torch.no_grad()
    def observe_batch(self, token_ids):
        """Записать наблюдения ��з батча в HDC-памяти.

        token_ids: (B, T) — батч последовательностей
        """
        if self.cb_binary is None:
            return

        B, T = token_ids.shape

        for pos in range(min(self.n_pos, T - 1)):
            offset = pos + 1
            ctx_ids = token_ids[:, :T - offset].reshape(-1)     # контекстные токены
            tgt_ids = token_ids[:, offset:T].reshape(-1)         # целевые токены

            tgt_codes = self.cb_binary[tgt_ids].float()  # (N, D)

            # Добавляем в память
            self.pos_mem[pos].index_add_(0, ctx_ids, tgt_codes)
            self.pos_count[pos].index_add_(0, ctx_ids, torch.ones(len(ctx_ids), device=self.device))

    @torch.no_grad()
    def get_hint(self, token_ids):
        """Сформировать подсказку для модели.

        token_ids: (B, T) — текущий контекст
        return: (B, T, D) — подсказка (что статистически следует дальше)
        """
        B, T = token_ids.shape
        D = self.D
        hint = torch.zeros(B, T, D, device=self.device)

        for t in range(T):
            vote = torch.zeros(B, D, device=self.device)
            n_votes = 0

            for pos in range(min(self.n_pos, t + 1)):
                src_t = t - pos  # позиция контекстного токена
                if src_t < 0:
                    break

                tok_ids = token_ids[:, src_t]  # (B,)

                # Среднее ожидание из памяти
                counts = self.pos_count[pos][tok_ids].unsqueeze(1).clamp(min=1)  # (B, 1)
                mem_vec = self.pos_mem[pos][tok_ids] / counts  # (B, D)

                weight = 1.0 / (pos + 1)
                vote += weight * mem_vec
                n_votes += 1

            if n_votes > 0:
                # Нормализуем подсказку
                hint[:, t] = vote / n_votes

        return hint

    @torch.no_grad()
    def decay_memories(self, factor=0.99):
        """Медленное затухание старых наблюдений."""
        self.pos_mem *= factor
        self.pos_count *= factor


# ============================================================
# Data loading
# ============================================================

def load_data(path, max_tokens=0):
    data = np.fromfile(str(path), dtype=np.uint16)
    if max_tokens > 0:
        data = data[:max_tokens]
    return data


def get_batch(data, batch_size, seq_len, device):
    max_start = len(data) - seq_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[s:s + seq_len] for s in starts])
    y = np.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return (torch.from_numpy(x.astype(np.int64)).to(device),
            torch.from_numpy(y.astype(np.int64)).to(device))


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, val_data, device, batch_size=32, seq_len=128,
             n_batches=20, cogit=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top10 = 0
    total_tokens = 0

    for _ in range(n_batches):
        x, y = get_batch(val_data, batch_size, seq_len, device)

        hint = None
        if cogit is not None:
            hint = cogit.get_hint(x)

        logits, loss = model(x, y, cogit_hint=hint)
        total_loss += loss.item() * y.numel()

        pred = logits.argmax(-1)
        total_correct += (pred == y).sum().item()

        top10 = logits.topk(10, dim=-1).indices
        total_top10 += (top10 == y.unsqueeze(-1)).any(-1).sum().item()

        total_tokens += y.numel()

    model.train()
    avg_loss = total_loss / total_tokens
    return {
        "loss": avg_loss,
        "ppl": min(torch.exp(torch.tensor(avg_loss)).item(), 100000),
        "acc": total_correct / total_tokens,
        "top10": total_top10 / total_tokens,
    }


@torch.no_grad()
def generate(model, sp, prompt, device, max_len=50, cogit=None):
    model.eval()
    ids = sp.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_len):
        ctx = idx[:, -model.max_seq_len:]

        hint = None
        if cogit is not None:
            hint = cogit.get_hint(ctx)

        logits, _ = model(ctx, cogit_hint=hint)
        logits = logits[:, -1, :] / 0.8

        # Top-k + repetition penalty
        recent = set(idx[0, -30:].tolist())
        for tid in recent:
            if logits[0, tid] > 0:
                logits[0, tid] /= 1.3
            else:
                logits[0, tid] *= 1.3

        v, _ = torch.topk(logits, 40)
        logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)

        if next_id.item() == 2:
            break

    model.train()
    return sp.decode(idx[0].tolist())


# ============================================================
# Training loop
# ============================================================

def train_model(model, train_data, val_data, sp, device, cogit=None,
                label="", time_budget=1800, eval_every_sec=600,
                batch_size=16, seq_len=128, lr=3e-4):
    """Train for a fixed time budget. Eval every eval_every_sec seconds."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=time_budget // 2, eta_min=lr * 0.1
    )

    model.train()
    t_start = time.time()
    last_eval = t_start
    step = 0
    total_loss = 0.0
    log_interval = 100

    results_log = []

    print(f"\n{'='*60}")
    print(f"TRAINING: {label}")
    print(f"  Time budget: {time_budget}s ({time_budget/60:.0f} min)")
    print(f"  Eval every: {eval_every_sec}s ({eval_every_sec/60:.0f} min)")
    print(f"  Batch: {batch_size}, Seq: {seq_len}, LR: {lr}")
    print(f"  Cogit: {'YES' if cogit else 'NO'}")
    print(f"{'='*60}\n")

    while True:
        elapsed = time.time() - t_start
        if elapsed >= time_budget:
            break

        x, y = get_batch(train_data, batch_size, seq_len, device)

        # CogitLayer: наблюдаем батч + формируем подсказку
        hint = None
        if cogit is not None:
            cogit.observe_batch(x)
            hint = cogit.get_hint(x)
            # Периодически синхронизируем кодбук и затухаем
            if step % 50 == 0:
                cogit.update_codebook(model.codebook)
            if step % 200 == 0:
                cogit.decay_memories(0.995)

        logits, loss = model(x, y, cogit_hint=hint)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        step += 1

        if step % log_interval == 0:
            avg = total_loss / log_interval
            total_loss = 0.0
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [{label}] step {step:>5}  loss={avg:.4f}  "
                  f"lr={lr_now:.6f}  time={elapsed:.0f}s")

        # Eval каждые eval_every_sec секунд
        now = time.time()
        if now - last_eval >= eval_every_sec:
            last_eval = now
            metrics = evaluate(model, val_data, device, cogit=cogit,
                               batch_size=batch_size, seq_len=seq_len)
            results_log.append({"step": step, "time": elapsed, **metrics})

            print(f"\n  >>> [{label}] EVAL at {elapsed:.0f}s (step {step}):")
            print(f"      loss={metrics['loss']:.4f}  ppl={metrics['ppl']:.1f}  "
                  f"acc={metrics['acc']:.4f}  top10={metrics['top10']:.4f}")

            # Генерация
            for prompt in ["### Instruction: Explain what is AI.\n### Response:"]:
                text = generate(model, sp, prompt, device, max_len=30, cogit=cogit)
                print(f"      Gen: {text[:150]}")
            print()

    # Финальный eval
    metrics = evaluate(model, val_data, device, cogit=cogit,
                       batch_size=batch_size, seq_len=seq_len)
    results_log.append({"step": step, "time": time.time() - t_start, **metrics})

    print(f"\n  >>> [{label}] FINAL (step {step}, {time.time()-t_start:.0f}s):")
    print(f"      loss={metrics['loss']:.4f}  ppl={metrics['ppl']:.1f}  "
          f"acc={metrics['acc']:.4f}  top10={metrics['top10']:.4f}")

    for prompt in ["### Instruction: Explain what is AI.\n### Response:",
                   "### Instruction: What is the capital of France?\n### Response:"]:
        text = generate(model, sp, prompt, device, max_len=40, cogit=cogit)
        print(f"      Gen: {text[:180]}")

    return results_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=1800,
                        help="Time budget per model in seconds (default: 1800 = 30 min)")
    parser.add_argument("--eval-every", type=int, default=600,
                        help="Eval interval in seconds (default: 600 = 10 min)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--ctrl-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mode", choices=["both", "baseline", "cogito"], default="both")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    vocab_size = sp.get_piece_size()
    print(f"Tokenizer: {vocab_size} vocab")

    # Data
    train_data = load_data(ALPACA_BIN)
    val_data = load_data(VAL_BIN)
    print(f"Train: {len(train_data):,} tokens (Alpaca instructions)")
    print(f"Val: {len(val_data):,} tokens")

    # Model config
    config = dict(
        vocab_size=vocab_size, dim=args.dim, n_blocks=args.blocks,
        ctrl_dim=args.ctrl_dim, n_heads=2, max_seq_len=args.seq_len, dropout=0.1,
    )

    # Create model
    model_a = MiniV14(**config).to(device)
    n_params = sum(p.numel() for p in model_a.parameters())
    print(f"\nMini v14.1: {n_params:,} params")
    print(f"  dim={args.dim}, blocks={args.blocks}, ctrl={args.ctrl_dim}")

    # ============================================================
    # A) Baseline — standard backprop
    # ============================================================
    if args.mode in ("both", "baseline"):
        print("\n" + "="*60)
        print("MODEL A: Standard backprop (no Cogito)")
        print("="*60)
        results_a = train_model(
            model_a, train_data, val_data, sp, device,
            cogit=None, label="BASELINE",
            time_budget=args.time, eval_every_sec=args.eval_every,
            batch_size=args.batch_size, seq_len=args.seq_len, lr=args.lr,
        )

    # ============================================================
    # B) Cogito — backprop + CogitLayer
    # ============================================================
    if args.mode in ("both", "cogito"):
        # Fresh model with same init
        torch.manual_seed(42)
        model_b = MiniV14(**config).to(device)

        # Copy init weights from model_a for fair comparison
        if args.mode == "both":
            model_b.load_state_dict(
                {k: v.clone() for k, v in model_a.state_dict().items()}
                if hasattr(model_a, 'state_dict') else model_b.state_dict()
            )
            # Re-init model_a was already trained, use fresh copy
            torch.manual_seed(42)
            model_b = MiniV14(**config).to(device)

        cogit = CogitLayer(vocab_size, args.dim, n_pos=5, device=device)
        cogit.update_codebook(model_b.codebook)

        # Pre-seed Cogito memory with a quick pass over training data
        print("\nPre-seeding Cogito memory...")
        preseed_size = min(200_000, len(train_data))
        preseed_data = torch.from_numpy(
            train_data[:preseed_size].astype(np.int64)
        ).unsqueeze(0).to(device)
        # Process in chunks
        chunk = 1000
        for i in range(0, preseed_size - chunk, chunk):
            cogit.observe_batch(preseed_data[:, i:i+chunk])
        print(f"  Pre-seeded from {preseed_size:,} tokens")

        print("\n" + "="*60)
        print("MODEL B: Backprop + CogitLayer")
        print("="*60)
        results_b = train_model(
            model_b, train_data, val_data, sp, device,
            cogit=cogit, label="COGITO",
            time_budget=args.time, eval_every_sec=args.eval_every,
            batch_size=args.batch_size, seq_len=args.seq_len, lr=args.lr,
        )

    # ============================================================
    # Comparison
    # ============================================================
    if args.mode == "both":
        print(f"\n{'='*70}")
        print(f"FINAL COMPARISON")
        print(f"{'='*70}")
        print(f"{'':>20} {'BASELINE':>12} {'COGITO':>12} {'Winner':>10}")
        print(f"{'-'*56}")

        fa = results_a[-1]
        fb = results_b[-1]
        for metric in ["loss", "ppl", "acc", "top10"]:
            va = fa[metric]
            vb = fb[metric]
            if metric in ("loss", "ppl"):
                winner = "COGITO" if vb < va else "BASELINE"
            else:
                winner = "COGITO" if vb > va else "BASELINE"
            print(f"{metric:>20} {va:>12.4f} {vb:>12.4f} {winner:>10}")

        print(f"{'steps':>20} {fa['step']:>12} {fb['step']:>12}")
        print(f"{'time':>20} {fa['time']:>11.0f}s {fb['time']:>11.0f}s")

        # Progress over time
        print(f"\n{'Progress over time':}")
        print(f"{'Time':>8} {'BL loss':>10} {'CG loss':>10} {'BL acc':>10} {'CG acc':>10}")
        for ra, rb in zip(results_a, results_b):
            print(f"{ra['time']:>7.0f}s {ra['loss']:>10.4f} {rb['loss']:>10.4f} "
                  f"{ra['acc']:>10.4f} {rb['acc']:>10.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
