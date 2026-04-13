"""
Finetune HDC-Brain v14.1 + CogitLayer on Alpaca instructions.

Takes pretrained base model (BPB 5.434) and teaches it to follow instructions.
CogitLayer provides hint for faster adaptation.

Usage:
  python3 finetune_cogito.py
"""
import json
import time
from datetime import datetime, timezone
import signal
import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from hdc_brain_v14_1 import create_model
from cogit_layer_server import CogitLayer

# === Logging ===
LOG_FILE = 'finetune.log'
EXPERIMENT_LOG = "finetune.jsonl"

def log_json(evt, data):
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": evt, **data}
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def log(msg=''):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

# === Graceful stop ===
stop_training = False
def handler(signum, frame):
    global stop_training
    print("\n[Signal] Stopping gracefully...")
    stop_training = True
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)
if os.name != 'nt':
    signal.signal(signal.SIGALRM, handler)

# === Config ===
MODEL_CONFIG = {
    'hdc_dim': 4096,
    'max_seq_len': 512,
    'n_blocks': 8,
    'controller_dim': 2560,
    'n_heads': 4,
    'dropout': 0.05,  # lower dropout for finetune
    'max_thoughts': 4,
    'use_checkpoint': True,
}

# Finetune hyperparams — much smaller than pretrain
BATCH = 8            # smaller batch for instruction data
GRAD_ACCUM = 4       # effective batch = 32 * 512 = 16K tokens/step
SEQ_LEN = 512
LR = 1e-4            # lower than pretrain (3e-4)
LR_MIN = 1e-5
WARMUP = 100
MAX_ITERS = 5000     # Alpaca is small, don't need many iters
USE_AMP = True
AMP_DTYPE = 'bfloat16'
CLIP_GRAD = 1.0
TRAIN_THOUGHTS = 3
EVAL_INTERVAL = 250
LOG_INTERVAL = 25
GENERATE_INTERVAL = 500

# CogitLayer config
COGIT_N_POS = 3
COGIT_HINT_WEIGHT = 0.1
COGIT_PRESEED = 0        # preseed from alpaca data below
COGIT_SYNC_EVERY = 25
COGIT_DECAY_EVERY = 100
COGIT_DECAY_FACTOR = 0.99

# Data files
TOKENIZER_PATH = 'bpe_en_32k.model'
ALPACA_BIN = 'alpaca.bin'
VAL_BIN = 'val.bin'         # use pretrain val for eval
BASE_CKPT = 'best_hdc_brain_v14_1.pt'  # pretrained base
FINETUNE_CKPT = 'best_finetune_v14_1.pt'
FINETUNE_LAST_CKPT = 'last_finetune_v14_1.pt'

# === Device ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

use_bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()
AMP_DTYPE = 'bfloat16' if use_bf16 else 'float16'
print(f"  Using {AMP_DTYPE}")

# === Tokenizer ===
sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
vocab_size = sp.get_piece_size()
print(f"Tokenizer: {vocab_size} vocab")

# === Data ===
print("Loading data...")
train_data = np.fromfile(ALPACA_BIN, dtype=np.uint16)
val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode='r')
print(f"  Finetune: {len(train_data):,} tokens (Alpaca instructions)")
print(f"  Val: {len(val_data):,} tokens")

tokens_per_step = BATCH * GRAD_ACCUM * SEQ_LEN
print(f"  Tokens/step: {tokens_per_step:,}")
epochs_approx = MAX_ITERS * tokens_per_step / len(train_data)
print(f"  Approximate epochs over Alpaca: {epochs_approx:.1f}")

# === Model ===
model, config = create_model(vocab_size, MODEL_CONFIG)
model = model.to(device)
n_params = sum(p.numel() for p in model.parameters())

# === Load pretrained base ===
print(f"Loading pretrained base from {BASE_CKPT}...")
ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
base_val = ckpt.get('val_loss', float('inf'))
base_iter = ckpt.get('iter', 0)
print(f"  Base model: iter {base_iter}, val {base_val:.4f}, BPB {base_val/0.6931:.3f}")
del ckpt

# === Resume finetune if exists ===
best_val = float('inf')
start_iter = 0
if os.path.exists(FINETUNE_CKPT):
    print(f"Resuming finetune from {FINETUNE_CKPT}...")
    ft_ckpt = torch.load(FINETUNE_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ft_ckpt['model'])
    best_val = ft_ckpt.get('val_loss', float('inf'))
    start_iter = ft_ckpt.get('iter', 0)
    print(f"  Resumed at iter {start_iter}, best val: {best_val:.4f}")
    del ft_ckpt

# === CogitLayer ===
print(f"\nInitializing CogitLayer...")
cogit = CogitLayer(vocab_size, config['hdc_dim'], n_pos=COGIT_N_POS, device=device)
cogit.update_codebook(model.codebook)

# Pre-seed from Alpaca data
print(f"  Pre-seeding from Alpaca data ({len(train_data):,} tokens)...")
cogit.preseed(train_data, max_tokens=len(train_data))
stats = cogit.memory_stats()
print(f"  CogitLayer: {stats['pos_mem_shape']}, {stats['total_mb']} MB")

# === Print config ===
print(f"\n{'='*60}")
print(f"FINETUNE: HDC-Brain v14.1 + CogitLayer")
print(f"  Base: BPB {base_val/0.6931:.3f} (iter {base_iter})")
print(f"  CogitLayer: {COGIT_N_POS} positions, hint_weight={COGIT_HINT_WEIGHT}")
print(f"  Batch: {BATCH} x {GRAD_ACCUM} x {SEQ_LEN} = {tokens_per_step:,} tokens/step")
print(f"  LR: {LR} -> {LR_MIN} (cosine, {WARMUP} warmup)")
print(f"  Max iters: {MAX_ITERS:,} (~{epochs_approx:.1f} epochs)")
print(f"  Data: Alpaca {len(train_data):,} tokens")
print(f"{'='*60}\n")

# === Optimizer — lower LR for finetune ===
codebook_params = [model.codebook]
other_params = [p for n, p in model.named_parameters() if n != 'codebook']

optimizer = torch.optim.AdamW([
    {'params': codebook_params, 'lr': LR * 0.1, 'weight_decay': 0.0},
    {'params': other_params, 'lr': LR, 'weight_decay': 0.01},
], betas=(0.9, 0.95))

scaler = torch.amp.GradScaler('cuda', enabled=(AMP_DTYPE == 'float16'))
print("torch.compile: OFF (incompatible with Thought Loops)")


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(d) - SEQ_LEN - 1, size=(BATCH,))
    X = torch.stack([torch.from_numpy(d[i:i+SEQ_LEN].astype(np.int64).copy()) for i in ix])
    Y = torch.stack([torch.from_numpy(d[i+1:i+SEQ_LEN+1].astype(np.int64).copy()) for i in ix])
    return X.to(device), Y.to(device)


def forward_with_cogit(model, X, Y, cogit, n_thoughts):
    """Forward pass with CogitLayer hint."""
    B, T = X.shape
    hint = cogit.get_hint_fast(X)

    tokens = model._ste_encode(X)
    tokens = model._cyclic_position(tokens)

    with torch.no_grad():
        token_scale = tokens.detach().norm(dim=-1, keepdim=True).mean()
    tokens = tokens + hint * token_scale * COGIT_HINT_WEIGHT

    h = model.thought_loop(tokens, model.blocks, n_thoughts, model.use_checkpoint)
    h = model.output_ln(h)
    logits = F.linear(h, model.codebook) * model.output_scale

    loss = F.cross_entropy(logits.view(-1, model.vocab_size), Y.view(-1))
    return logits, loss


@torch.no_grad()
def evaluate(n_batches=20):
    model.eval()
    losses = {'train': [], 'val': []}
    for split in ['train', 'val']:
        for _ in range(n_batches):
            X, Y = get_batch(split)
            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=getattr(torch, AMP_DTYPE)):
                _, loss = forward_with_cogit(model, X, Y, cogit, TRAIN_THOUGHTS)
            losses[split].append(loss.item())
    model.train()
    return np.mean(losses['train']), np.mean(losses['val'])


@torch.no_grad()
def generate_sample(prompt="### Instruction: Explain what is AI.\n### Response:"):
    """Generate with CogitLayer hint."""
    model.eval()
    ids = sp.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(150):
        ctx = idx[:, -model.max_seq_len:]
        hint = cogit.get_hint_fast(ctx)

        tokens = model._ste_encode(ctx)
        tokens = model._cyclic_position(tokens)
        with torch.no_grad():
            token_scale = tokens.norm(dim=-1, keepdim=True).mean()
        tokens = tokens + hint * token_scale * COGIT_HINT_WEIGHT

        h = model.thought_loop(tokens, model.blocks, TRAIN_THOUGHTS, False)
        h = model.output_ln(h)
        logits = F.linear(h, model.codebook) * model.output_scale
        logits = logits[:, -1, :] / 0.7  # slightly lower temp for instruction following

        # Repetition penalty
        recent = set(idx[0, -50:].tolist())
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


# === Training loop ===
print(f"Starting finetune from iter {start_iter}...")
log_json("finetune_start", {
    "model": "HDC-Brain v14.1 + CogitLayer finetune",
    "base_bpb": round(base_val / 0.6931, 4),
    "base_iter": base_iter,
    "cogit": {"n_pos": COGIT_N_POS, "hint_weight": COGIT_HINT_WEIGHT},
    "train": {"batch": BATCH, "accum": GRAD_ACCUM, "tps": tokens_per_step,
              "lr": LR, "lr_min": LR_MIN, "max_iters": MAX_ITERS},
    "data": {"alpaca_tok": int(len(train_data)), "val_tok": int(len(val_data))},
})

t0 = time.time()
running_loss = 0.0
running_count = 0

for it in range(start_iter, MAX_ITERS + 1):
    if stop_training:
        break

    # === LR schedule ===
    if it < WARMUP:
        lr = LR * (it + 1) / WARMUP
    else:
        progress = (it - WARMUP) / max(MAX_ITERS - WARMUP, 1)
        progress = min(progress, 1.0)
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))

    for i, pg in enumerate(optimizer.param_groups):
        pg['lr'] = lr * (0.1 if i == 0 else 1.0)

    # === CogitLayer maintenance ===
    if it % COGIT_SYNC_EVERY == 0:
        cogit.update_codebook(model.codebook)
    if it % COGIT_DECAY_EVERY == 0 and it > start_iter:
        cogit.decay_memories(COGIT_DECAY_FACTOR)

    # === Forward + backward ===
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for micro in range(GRAD_ACCUM):
        X, Y = get_batch('train')
        cogit.observe_batch(X)

        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=getattr(torch, AMP_DTYPE)):
            _, loss = forward_with_cogit(model, X, Y, cogit, TRAIN_THOUGHTS)
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        accum_loss += loss.item()

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
    scaler.step(optimizer)
    scaler.update()

    running_loss += accum_loss
    running_count += 1

    # === Logging ===
    if it % LOG_INTERVAL == 0 and it > start_iter:
        elapsed = time.time() - t0
        steps_done = it - start_iter
        ms_per_step = elapsed / steps_done * 1000
        avg_loss = running_loss / running_count

        log(f"iter {it:>5d} | loss {avg_loss:.4f} | "
            f"lr {lr:.2e} | grad {grad_norm:.2f} | "
            f"{ms_per_step:.0f}ms/step")

        log_json("step", {"iter": it, "loss": round(avg_loss, 6),
            "lr": round(lr, 8), "grad": round(float(grad_norm), 4),
            "ms": round(ms_per_step, 1)})
        running_loss = 0.0
        running_count = 0

    # === Evaluation ===
    if it > 0 and it % EVAL_INTERVAL == 0:
        train_loss, val_loss = evaluate()
        bpb = val_loss / 0.6931
        elapsed_h = (time.time() - t0) / 3600

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        ckpt_data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'val_loss': val_loss,
            'best_val': best_val,
            'iter': it,
            'vocab_size': vocab_size,
            'finetune': True,
            'base_iter': base_iter,
            'cogit_config': {'n_pos': COGIT_N_POS, 'hint_weight': COGIT_HINT_WEIGHT},
        }

        if is_best:
            torch.save(ckpt_data, FINETUNE_CKPT)
        torch.save(ckpt_data, FINETUNE_LAST_CKPT)

        marker = ">>> BEST!" if is_best else ""
        log(f"{'='*60}")
        log(f"[{elapsed_h:.1f}h, iter {it}] "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"BPB: {bpb:.3f} | Best: {best_val/0.6931:.3f} {marker}")

        thought_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        gates = torch.sigmoid(thought_model.thought_loop.thought_gates).tolist()
        log(f"  Thought gates: {[round(g, 3) for g in gates]}")
        log_json("eval", {"iter": it, "train": round(train_loss, 6),
            "val": round(val_loss, 6), "bpb": round(bpb, 4),
            "best": int(is_best), "hours": round(elapsed_h, 2)})
        log(f"{'='*60}")

    # === Generation samples ===
    if it > 0 and it % GENERATE_INTERVAL == 0:
        prompts = [
            "### Instruction: Explain what is AI.\n### Response:",
            "### Instruction: What is the capital of France?\n### Response:",
            "### Instruction: Write a short poem about the ocean.\n### Response:",
            "### Instruction: How does photosynthesis work?\n### Response:",
        ]
        log(f"\n--- Generation samples (iter {it}) ---")
        for prompt in prompts:
            gen = generate_sample(prompt)
            # Show only the response part
            if "### Response:" in gen:
                response = gen.split("### Response:")[-1].strip()
            else:
                response = gen
            log(f"  Q: {prompt.split(chr(10))[0][20:]}")
            log(f"  A: {response[:200]}")
            log()
        log(f"---")

# === Final save ===
print(f"\nFinetune finished at iter {it}.")
thought_model = model._orig_mod if hasattr(model, '_orig_mod') else model
torch.save({
    'model': thought_model.state_dict(),
    'config': config,
    'val_loss': best_val,
    'iter': it,
    'vocab_size': vocab_size,
    'finetune': True,
    'base_iter': base_iter,
}, FINETUNE_LAST_CKPT)
print(f"Saved {FINETUNE_LAST_CKPT}")
log_json("end", {"iter": it, "best_val": round(best_val, 6),
    "best_bpb": round(best_val / 0.6931, 4),
    "hours": round((time.time() - t0) / 3600, 2)})
print(f"Best val: {best_val:.4f} (BPB: {best_val/0.6931:.3f})")
