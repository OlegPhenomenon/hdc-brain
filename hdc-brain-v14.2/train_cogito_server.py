"""
Train HDC-Brain v14.1 + CogitLayer: English 300M.

Same as train.py but with CogitLayer integration:
- CogitLayer observes each batch and provides hints
- Hint is added to token embeddings before blocks
- CogitLayer memory lives on CPU (~2.4 GB), only hint goes to GPU
- Pre-seeds from first 500K tokens of training data

Usage:
  python3 train_cogito_server.py
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

# === Unbuffered logging to file ===
LOG_FILE = 'train.log'
EXPERIMENT_LOG = "experiment.jsonl"

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
    'dropout': 0.1,
    'max_thoughts': 4,
    'use_checkpoint': True,
}

# Training hyperparams
BATCH = 16
GRAD_ACCUM = 8
SEQ_LEN = 512
LR = 3e-4
LR_MIN = 3e-5
WARMUP = 500
MAX_ITERS = 120_000
USE_AMP = True
AMP_DTYPE = 'bfloat16'
CLIP_GRAD = 1.0
TRAIN_THOUGHTS = 3
EVAL_INTERVAL = 500
LOG_INTERVAL = 50
GENERATE_INTERVAL = 2000

# CogitLayer config
COGIT_N_POS = 3          # positional memories (1-gram to 3-gram, saves VRAM)
COGIT_HINT_WEIGHT = 0.1  # how much hint influences tokens (conservative start)
COGIT_PRESEED = 500_000  # tokens to pre-seed from training data
COGIT_SYNC_EVERY = 50    # sync codebook every N steps
COGIT_DECAY_EVERY = 200  # decay memories every N steps
COGIT_DECAY_FACTOR = 0.995

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

use_bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()
AMP_DTYPE = 'bfloat16' if use_bf16 else 'float16'
print(f"  Using {AMP_DTYPE}")

# === Tokenizer ===
sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
vocab_size = sp.get_piece_size()
print(f"Tokenizer: {vocab_size} vocab ({TOKENIZER_PATH})")

# === Data ===
print("Loading data...")
train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode='r')
val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode='r')
print(f"  Train: {len(train_data):,} tokens ({len(train_data)/1e9:.2f}B)")
print(f"  Val:   {len(val_data):,} tokens ({len(val_data)/1e6:.1f}M)")

tokens_per_step = BATCH * GRAD_ACCUM * SEQ_LEN
total_tokens = MAX_ITERS * tokens_per_step
print(f"  Tokens/step: {tokens_per_step:,}")
print(f"  Total tokens seen: {total_tokens:,} ({total_tokens/1e9:.1f}B)")

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

# === CogitLayer ===
print(f"\nInitializing CogitLayer...")
cogit = CogitLayer(vocab_size, config['hdc_dim'], n_pos=COGIT_N_POS, device=device)
cogit.update_codebook(model.codebook)
cogit.preseed(train_data, max_tokens=COGIT_PRESEED)
stats = cogit.memory_stats()
print(f"  CogitLayer: {stats['pos_mem_shape']}, {stats['total_mb']} MB on CPU")

# === Print config ===
print(f"\n{'='*60}")
print(f"HDC-Brain v14.1 + CogitLayer: English 300M")
print(f"  HDC: {config['hdc_dim']}d, STE bipolar codebook")
print(f"  Blocks: {config['n_blocks']} (Memory + {config['n_heads']}-Head Binding Attention + Controller)")
print(f"  Controller: {config['controller_dim']}d inner")
print(f"  Thought Loops: {config['max_thoughts']} max, training with {TRAIN_THOUGHTS}")
print(f"  CogitLayer: {COGIT_N_POS} positions, hint_weight={COGIT_HINT_WEIGHT}")
print(f"  Vocab: {vocab_size} (BPE English)")
print(f"  Batch: {BATCH} x {GRAD_ACCUM} x {SEQ_LEN} = {tokens_per_step:,} tokens/step")
print(f"  Params: {n_params:,} (CogitLayer has 0 trainable params)")
print(f"  LR: {LR} -> {LR_MIN} (cosine, {WARMUP} warmup)")
print(f"  Max iters: {MAX_ITERS:,}")
print(f"{'='*60}\n")

# === Optimizer ===
codebook_params = [model.codebook]
other_params = [p for n, p in model.named_parameters() if n != 'codebook']

optimizer = torch.optim.AdamW([
    {'params': codebook_params, 'lr': LR * 0.1, 'weight_decay': 0.0},
    {'params': other_params, 'lr': LR, 'weight_decay': 0.05},
], betas=(0.9, 0.95))

scaler = torch.amp.GradScaler('cuda', enabled=(AMP_DTYPE == 'float16'))

# Restore optimizer
if resume_path:
    try:
        _ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if 'optimizer' in _ckpt:
            optimizer.load_state_dict(_ckpt['optimizer'])
            print(f'  Optimizer state restored')
        del _ckpt
    except Exception as e:
        print(f'  Could not restore optimizer: {e}')

print("torch.compile: OFF (incompatible with Thought Loops)")


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(d) - SEQ_LEN - 1, size=(BATCH,))
    X = torch.stack([torch.from_numpy(d[i:i+SEQ_LEN].astype(np.int64).copy()) for i in ix])
    Y = torch.stack([torch.from_numpy(d[i+1:i+SEQ_LEN+1].astype(np.int64).copy()) for i in ix])
    return X.to(device), Y.to(device)


def forward_with_cogit(model, X, Y, cogit, n_thoughts):
    """Forward pass with CogitLayer hint injected."""
    B, T = X.shape

    # Get hint from CogitLayer (computed on CPU, moved to GPU)
    # hint is already L2-normalized per token
    hint = cogit.get_hint_fast(X)

    # Encode tokens (same as model._ste_encode + _cyclic_position)
    tokens = model._ste_encode(X)
    tokens = model._cyclic_position(tokens)

    # Scale hint to match token magnitude
    # tokens have norm ~ alpha * sqrt(D), hint is unit-norm
    # We scale hint by mean token norm * hint_weight
    with torch.no_grad():
        token_scale = tokens.detach().norm(dim=-1, keepdim=True).mean()
    tokens = tokens + hint * token_scale * COGIT_HINT_WEIGHT

    # Process with ThoughtLoop (bypass model.forward, use internals)
    h = model.thought_loop(tokens, model.blocks, n_thoughts, model.use_checkpoint)

    # Output
    h = model.output_ln(h)
    logits = F.linear(h, model.codebook) * model.output_scale

    loss = F.cross_entropy(logits.view(-1, model.vocab_size), Y.view(-1))
    return logits, loss


@torch.no_grad()
def evaluate(n_batches=30):
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
def generate_sample(prompt="The capital of France is"):
    """Generate text with CogitLayer hint at each step."""
    model.eval()
    ids = sp.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(100):
        ctx = idx[:, -model.max_seq_len:]

        # Get CogitLayer hint for current context
        hint = cogit.get_hint_fast(ctx)

        # Encode + position + hint
        tokens = model._ste_encode(ctx)
        tokens = model._cyclic_position(tokens)
        with torch.no_grad():
            token_scale = tokens.norm(dim=-1, keepdim=True).mean()
        tokens = tokens + hint * token_scale * COGIT_HINT_WEIGHT

        # ThoughtLoop + output
        h = model.thought_loop(tokens, model.blocks, TRAIN_THOUGHTS, False)
        h = model.output_ln(h)
        logits = F.linear(h, model.codebook) * model.output_scale
        logits = logits[:, -1, :] / 0.8

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
print(f"Starting training from iter {start_iter}...")
log_json("experiment_start", {
    "model": "HDC-Brain v14.1 + CogitLayer", "n_params": n_params,
    "cogit": {"n_pos": COGIT_N_POS, "hint_weight": COGIT_HINT_WEIGHT,
              "preseed": COGIT_PRESEED, "memory_mb": stats['total_mb']},
    "config": {k: (int(v) if isinstance(v, bool) else v) for k, v in config.items()},
    "train": {"batch": BATCH, "accum": GRAD_ACCUM, "tps": tokens_per_step,
              "seq": SEQ_LEN, "lr": LR, "lr_min": LR_MIN, "warmup": WARMUP,
              "max_iters": MAX_ITERS, "amp": AMP_DTYPE, "clip": CLIP_GRAD,
              "thoughts": TRAIN_THOUGHTS},
    "data": {"train_tok": int(len(train_data)), "val_tok": int(len(val_data)),
             "vocab": vocab_size},
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
        pg['lr'] = lr * (0.1 if i == 0 else 1.0)

    # === CogitLayer maintenance ===
    if it % COGIT_SYNC_EVERY == 0:
        cogit.update_codebook(model.codebook)
    if it % COGIT_DECAY_EVERY == 0 and it > start_iter:
        cogit.decay_memories(COGIT_DECAY_FACTOR)

    # === Forward + backward with gradient accumulation ===
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for micro in range(GRAD_ACCUM):
        X, Y = get_batch('train')

        # CogitLayer observes this batch
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

        ckpt_data = {
            'model': model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'val_loss': val_loss,
            'best_val': best_val,
            'iter': it,
            'vocab_size': vocab_size,
            'cogit_config': {'n_pos': COGIT_N_POS, 'hint_weight': COGIT_HINT_WEIGHT},
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

        thought_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        gates = torch.sigmoid(thought_model.thought_loop.thought_gates).tolist()
        log(f"  Thought gates: {[round(g, 3) for g in gates]}")

        masses = thought_model.blocks[0].memory.mass_proj.weight.squeeze()
        log(f"  Mass proj norm: {masses.norm():.4f}")

        # CogitLayer stats
        cstats = cogit.memory_stats()
        log(f"  CogitLayer: {cstats['total_mb']}MB, active={cstats['active_tokens']}")

        log_json("eval", {"iter": it, "train": round(train_loss, 6), "val": round(val_loss, 6),
            "bpb": round(bpb, 4), "best_bpb": round(best_val / 0.6931, 4), "gap": round(gap, 4),
            "best": int(is_best), "hours": round(elapsed_h, 2),
            "gates": [round(g, 4) for g in gates], "mass": round(float(masses.norm()), 4),
            "cogit_mb": cstats['total_mb'],
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
    'cogit_config': {'n_pos': COGIT_N_POS, 'hint_weight': COGIT_HINT_WEIGHT},
}, LAST_CKPT_PATH)
print(f"Saved {LAST_CKPT_PATH}")
log_json("end", {"iter": it, "best_val": round(best_val, 6), "best_bpb": round(best_val / 0.6931, 4),
    "hours": round((time.time() - t0) / 3600, 2), "reason": "signal" if stop_training else "done"})
print(f"Best val: {best_val:.4f} (BPB: {best_val/0.6931:.3f})")
