"""
Train HDC-Brain v14.1: English 300M.

Optimized for A100/H100:
- Mixed precision (bf16)
- Gradient checkpointing
- Gradient accumulation
- Cosine LR schedule with warmup
- Periodic evaluation and checkpointing

Usage:
  python3 train.py
  python3 train.py --resume best_hdc_brain_v14_1.pt
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

# === Unbuffered logging to file ===
LOG_FILE = 'train.log'

# === JSON experiment log (arXiv) ===
EXPERIMENT_LOG = "experiment.jsonl"

def log_json(evt, data):
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": evt, **data}
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def log(msg=''):
    """Print to both stdout and log file, flush immediately."""
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
    signal.alarm(86400)  # 24h timer — auto-stop and save

# === Config ===
MODEL_CONFIG = {
    'hdc_dim': 4096,
    'max_seq_len': 512,
    'n_blocks': 8,
    'controller_dim': 2560,
    'n_heads': 4,
    'dropout': 0.1,
    'max_thoughts': 4,
    'use_checkpoint': True,  # gradient checkpointing
}

# Training hyperparams — RTX 3090 24GB (torch.compile uses extra VRAM for CUDA graphs)
BATCH = 16
GRAD_ACCUM = 8
SEQ_LEN = 512       # full 512 context fits
LR = 3e-4           # peak learning rate
LR_MIN = 3e-5       # min LR (cosine decay target)
WARMUP = 500         # warmup steps
MAX_ITERS = 30_000   # ~24h on RTX 3090, 0.7B tokens, extend if needed
USE_AMP = True
AMP_DTYPE = 'bfloat16'  # bf16 if supported, else fp16
CLIP_GRAD = 1.0
TRAIN_THOUGHTS = 3   # v14 proved: 1=garbage, 2=plateau, 3=quality. Don't compromise.
EVAL_INTERVAL = 500
LOG_INTERVAL = 50
GENERATE_INTERVAL = 2000

# Data files
TOKENIZER_PATH = 'bpe_en_32k.model'
TRAIN_BIN = 'train.bin'
VAL_BIN = 'val.bin'
CKPT_PATH = 'best_hdc_brain_v14_1.pt'
LAST_CKPT_PATH = 'last_hdc_brain_v14_1.pt'

# === Device ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check for bf16 support
use_bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()
if use_bf16:
    AMP_DTYPE = 'bfloat16'
    print(f"  Using bfloat16 (native)")
else:
    AMP_DTYPE = 'float16'
    print(f"  Using float16 (with loss scaling)")

# === Tokenizer ===
if not os.path.exists(TOKENIZER_PATH):
    print(f"Tokenizer not found: {TOKENIZER_PATH}")
    print("Run: python3 prepare_data.py --step tokenizer")
    exit(1)

sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
vocab_size = sp.get_piece_size()
print(f"Tokenizer: {vocab_size} vocab ({TOKENIZER_PATH})")

# === Data ===
if not os.path.exists(TRAIN_BIN):
    print(f"Training data not found: {TRAIN_BIN}")
    print("Run: python3 prepare_data.py --step tokenize")
    exit(1)

print("Loading data...")
train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode='r')
val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode='r')
print(f"  Train: {len(train_data):,} tokens ({len(train_data)/1e9:.2f}B)")
print(f"  Val:   {len(val_data):,} tokens ({len(val_data)/1e6:.1f}M)")

tokens_per_step = BATCH * GRAD_ACCUM * SEQ_LEN
total_tokens = MAX_ITERS * tokens_per_step
epochs_approx = total_tokens / len(train_data)
print(f"  Tokens/step: {tokens_per_step:,}")
print(f"  Total tokens seen: {total_tokens:,} ({total_tokens/1e9:.1f}B)")
print(f"  Approximate epochs: {epochs_approx:.1f}")

# === Model ===
model, config = create_model(vocab_size, MODEL_CONFIG)
model = model.to(device)
n_params = sum(p.numel() for p in model.parameters())

# === Resume ===
best_val = float('inf')
start_iter = 0

resume_path = None
if os.path.exists(CKPT_PATH):
    resume_path = CKPT_PATH
elif os.path.exists(LAST_CKPT_PATH):
    resume_path = LAST_CKPT_PATH

if resume_path:
    print(f"Resuming from {resume_path}...")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    best_val = ckpt.get('val_loss', float('inf'))
    start_iter = ckpt.get('iter', 0)
    print(f"  Resumed at iter {start_iter}, best val: {best_val:.4f}")
else:
    print("Training from scratch")

# === Print config ===
print(f"\n{'='*60}")
print(f"HDC-Brain v14.1: English 300M")
print(f"  HDC: {config['hdc_dim']}d, STE bipolar codebook")
print(f"  Blocks: {config['n_blocks']} (Memory + {config['n_heads']}-Head Binding Attention + Controller)")
print(f"  Controller: {config['controller_dim']}d inner")
print(f"  Thought Loops: {config['max_thoughts']} max, training with {TRAIN_THOUGHTS}")
print(f"  Vocab: {vocab_size} (BPE English)")
print(f"  Batch: {BATCH} x {GRAD_ACCUM} x {SEQ_LEN} = {tokens_per_step:,} tokens/step")
print(f"  Params: {n_params:,}")
print(f"  LR: {LR} -> {LR_MIN} (cosine, {WARMUP} warmup)")
print(f"  Max iters: {MAX_ITERS:,}")
print(f"{'='*60}\n")

# === Optimizer ===
# Two param groups: codebook at lower LR (it learns via STE)
codebook_params = [model.codebook]
other_params = [p for n, p in model.named_parameters() if n != 'codebook']

optimizer = torch.optim.AdamW([
    {'params': codebook_params, 'lr': LR * 0.1, 'weight_decay': 0.0},  # codebook: 10x lower LR, no decay
    {'params': other_params, 'lr': LR, 'weight_decay': 0.05},
], betas=(0.9, 0.95))

# Loss scaling for fp16 (not needed for bf16)
scaler = torch.amp.GradScaler('cuda', enabled=(AMP_DTYPE == 'float16'))

# Restore optimizer state if available
if resume_path and os.path.exists(resume_path):
    try:
        _ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if 'optimizer' in _ckpt:
            optimizer.load_state_dict(_ckpt['optimizer'])
            print(f'  Optimizer state restored from {resume_path}')
        del _ckpt
    except Exception as e:
        print(f'  Could not restore optimizer: {e}')

# torch.compile disabled: causes infinite recompilation with 3 thoughts
# (ThoughtLoop has dynamic control flow that breaks CUDA graphs)
print("torch.compile: OFF (incompatible with Thought Loops)")


def get_batch(split):
    """Get a random batch from train or val data."""
    d = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(d) - SEQ_LEN - 1, size=(BATCH,))
    X = torch.stack([torch.from_numpy(d[i:i+SEQ_LEN].astype(np.int64).copy()) for i in ix])
    Y = torch.stack([torch.from_numpy(d[i+1:i+SEQ_LEN+1].astype(np.int64).copy()) for i in ix])
    return X.to(device), Y.to(device)


@torch.no_grad()
def evaluate(n_batches=30):
    """Evaluate on train and val sets."""
    model.eval()
    losses = {'train': [], 'val': []}
    for split in ['train', 'val']:
        for _ in range(n_batches):
            X, Y = get_batch(split)
            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=getattr(torch, AMP_DTYPE)):
                _, loss = model(X, Y, n_thoughts=TRAIN_THOUGHTS)
            losses[split].append(loss.item())
    model.train()
    return np.mean(losses['train']), np.mean(losses['val'])


@torch.no_grad()
def generate_sample(prompt="The capital of France is"):
    """Generate text sample for monitoring."""
    model.eval()
    ids = sp.encode(prompt)
    ids_tensor = torch.tensor([ids], device=device)
    out = model.generate(ids_tensor, max_len=100, temperature=0.8, top_k=40,
                         rep_penalty=1.3, n_thoughts=TRAIN_THOUGHTS)
    model.train()
    return sp.decode(out[0].tolist())


# === Training loop ===
print(f"Starting training from iter {start_iter}...")
log_json("experiment_start", {
    "model": "HDC-Brain v14.1", "n_params": n_params,
    "config": {k: (int(v) if isinstance(v, bool) else v) for k, v in config.items()},
    "train": {"batch": BATCH, "accum": GRAD_ACCUM, "tps": tokens_per_step,
              "seq": SEQ_LEN, "lr": LR, "lr_min": LR_MIN, "warmup": WARMUP,
              "max_iters": MAX_ITERS, "amp": AMP_DTYPE, "clip": CLIP_GRAD,
              "thoughts": TRAIN_THOUGHTS, "wd": 0.05, "betas": [0.9, 0.95]},
    "data": {"train_tok": int(len(train_data)), "val_tok": int(len(val_data)),
             "vocab": vocab_size, "tokenizer": TOKENIZER_PATH},
    "hw": {"gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
           "vram": round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if device == "cuda" else 0},
    "resume": resume_path, "start_iter": start_iter,
})
t0 = time.time()
running_loss = 0.0
running_count = 0

for it in range(start_iter, MAX_ITERS + 1):
    if stop_training:
        break

    # === LR schedule: warmup + cosine decay ===
    if it < WARMUP:
        lr = LR * (it + 1) / WARMUP
    else:
        progress = (it - WARMUP) / max(MAX_ITERS - WARMUP, 1)
        progress = min(progress, 1.0)
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))

    for i, pg in enumerate(optimizer.param_groups):
        pg['lr'] = lr * (0.1 if i == 0 else 1.0)  # codebook stays at 10x lower

    # === Forward + backward with gradient accumulation ===
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for micro in range(GRAD_ACCUM):
        X, Y = get_batch('train')
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=getattr(torch, AMP_DTYPE)):
            _, loss = model(X, Y, n_thoughts=TRAIN_THOUGHTS)
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
        tokens_seen = steps_done * tokens_per_step
        avg_loss = running_loss / running_count

        log(f"iter {it:>6d} | loss {avg_loss:.4f} | "
            f"lr {lr:.2e} | grad {grad_norm:.2f} | "
            f"{ms_per_step:.0f}ms/step | "
            f"{tokens_seen/1e6:.0f}M tok")

        log_json("step", {"iter": it, "loss": round(avg_loss, 6), "lr": round(lr, 8),
            "grad": round(float(grad_norm), 4), "ms": round(ms_per_step, 1), "tok": tokens_seen})
        running_loss = 0.0
        running_count = 0

    # === Evaluation ===
    if it > 0 and it % EVAL_INTERVAL == 0:
        train_loss, val_loss = evaluate()
        bpb = val_loss / 0.6931
        gap = train_loss - val_loss
        elapsed_h = (time.time() - t0) / 3600

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        # Save checkpoint
        ckpt_data = {
            'model': model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'val_loss': val_loss,
            'best_val': best_val,
            'iter': it,
            'vocab_size': vocab_size,
        }

        if is_best:
            torch.save(ckpt_data, CKPT_PATH)
        torch.save(ckpt_data, LAST_CKPT_PATH)

        marker = ">>> BEST!" if is_best else ""
        log(f"{'='*60}")
        log(f"[{elapsed_h:.1f}h, iter {it}] "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"BPB: {bpb:.3f} | Best BPB: {best_val/0.6931:.3f} | "
            f"Gap: {gap:.3f} {marker}")

        # Thought gate values
        thought_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        gates = torch.sigmoid(thought_model.thought_loop.thought_gates).tolist()
        log(f"  Thought gates: {[round(g, 3) for g in gates]}")

        # Context mass stats (block 0)
        masses = thought_model.blocks[0].memory.mass_proj.weight.squeeze()
        log(f"  Mass proj norm: {masses.norm():.4f}")
        log_json("eval", {"iter": it, "train": round(train_loss, 6), "val": round(val_loss, 6),
            "bpb": round(bpb, 4), "best_bpb": round(best_val / 0.6931, 4), "gap": round(gap, 4),
            "best": int(is_best), "hours": round(elapsed_h, 2),
            "gates": [round(g, 4) for g in gates], "mass": round(float(masses.norm()), 4),
            "vram_mb": torch.cuda.max_memory_allocated() // (1024*1024) if device == "cuda" else 0})
        log(f"{'='*60}")

    # === Generation samples ===
    if it > 0 and it % GENERATE_INTERVAL == 0:
        prompts = [
            "The capital of France is",
            "Machine learning is a field of",
            "In the year 2050, humanity",
            "The meaning of life is",
        ]
        log(f"\n--- Generation samples (iter {it}) ---")
        for prompt in prompts:
            gen = generate_sample(prompt)
            log(f"  [{prompt[:30]}]: '{gen[:200]}'")
        log(f"---")

# === Final save ===
print(f"\nTraining finished at iter {it}.")
thought_model = model._orig_mod if hasattr(model, '_orig_mod') else model
torch.save({
    'model': thought_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': config,
    'val_loss': best_val,
    'iter': it,
    'vocab_size': vocab_size,
}, LAST_CKPT_PATH)
print(f"Saved {LAST_CKPT_PATH}")
log_json("end", {"iter": it, "best_val": round(best_val, 6), "best_bpb": round(best_val / 0.6931, 4),
    "hours": round((time.time() - t0) / 3600, 2), "reason": "signal" if stop_training else "done"})
print(f"Best val: {best_val:.4f} (BPB: {best_val/0.6931:.3f})")
