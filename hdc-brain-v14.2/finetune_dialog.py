"""Finetune v14.1 on dialog data with CogitLayer (preseed only, no live observe)."""
import json, time, signal, os, math
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from hdc_brain_v14_1 import create_model
from cogit_layer_server import CogitLayer

LOG_FILE = "finetune.log"

def log(msg=""):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
        f.flush()

stop = False

def handler(s, f):
    global stop
    stop = True

signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

# Config
BATCH, ACCUM, SEQ = 4, 2, 512
LR, LR_MIN, WARMUP = 1e-4, 1e-5, 200
MAX_ITERS = 15000
THOUGHTS = 3
COGIT_HINT_WEIGHT = 0.1
device = "cuda"

sp = spm.SentencePieceProcessor(model_file="bpe_en_32k.model")
V = sp.get_piece_size()

train_data = np.fromfile("dialog_train.bin", dtype=np.uint16)
val_data = np.fromfile("dialog_val.bin", dtype=np.uint16)
log(f"Data: train={len(train_data):,} val={len(val_data):,}")

model, cfg = create_model(V, {
    "hdc_dim": 4096, "max_seq_len": 512, "n_blocks": 8,
    "controller_dim": 2560, "n_heads": 4, "dropout": 0.05,
    "max_thoughts": 4, "use_checkpoint": False,
})
model = model.to(device)

ckpt = torch.load("best_hdc_brain_v14_1.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
log(f"Loaded base: BPB {ckpt.get('val_loss', 0) / 0.6931:.3f}")
del ckpt

# CogitLayer — preseed from training data, NO live observe (avoids CUDA hang)
log("Initializing CogitLayer...")
cogit = CogitLayer(V, 4096, n_pos=3, device=device)
cogit.update_codebook(model.codebook)
cogit.preseed(train_data, max_tokens=min(len(train_data), 2_000_000))
stats = cogit.memory_stats()
log(f"  CogitLayer: {stats['total_mb']}MB, active={stats['active_tokens']}")

opt = torch.optim.AdamW([
    {"params": [model.codebook], "lr": LR * 0.1, "weight_decay": 0.0},
    {"params": [p for n, p in model.named_parameters() if n != "codebook"],
     "lr": LR, "weight_decay": 0.01},
], betas=(0.9, 0.95))


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(d) - SEQ - 1, size=(BATCH,))
    X = torch.stack([torch.from_numpy(d[i:i + SEQ].astype(np.int64).copy()) for i in ix]).to(device)
    Y = torch.stack([torch.from_numpy(d[i + 1:i + SEQ + 1].astype(np.int64).copy()) for i in ix]).to(device)
    return X, Y


def forward_cogit(X, Y):
    """Forward with CogitLayer hint."""
    hint = cogit.get_hint_fast(X)
    tokens = model._ste_encode(X)
    tokens = model._cyclic_position(tokens)
    with torch.no_grad():
        ts = tokens.detach().norm(dim=-1, keepdim=True).mean()
    tokens = tokens + hint * ts * COGIT_HINT_WEIGHT
    h = model.thought_loop(tokens, model.blocks, THOUGHTS, False)
    h = model.output_ln(h)
    logits = F.linear(h, model.codebook) * model.output_scale
    loss = F.cross_entropy(logits.view(-1, V), Y.view(-1))
    return logits, loss


@torch.no_grad()
def evaluate():
    model.eval()
    losses = {"train": [], "val": []}
    for s in ["train", "val"]:
        for _ in range(20):
            X, Y = get_batch(s)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = forward_cogit(X, Y)
            losses[s].append(loss.item())
    model.train()
    return np.mean(losses["train"]), np.mean(losses["val"])


@torch.no_grad()
def gen(prompt):
    model.eval()
    ids = sp.encode(prompt)
    idx = torch.tensor([ids], device=device)
    for _ in range(150):
        ctx = idx[:, -512:]
        hint = cogit.get_hint_fast(ctx)
        tokens = model._ste_encode(ctx)
        tokens = model._cyclic_position(tokens)
        ts = tokens.norm(dim=-1, keepdim=True).mean()
        tokens = tokens + hint * ts * COGIT_HINT_WEIGHT
        h = model.thought_loop(tokens, model.blocks, THOUGHTS, False)
        h = model.output_ln(h)
        logits = F.linear(h, model.codebook) * model.output_scale
        logits = logits[:, -1, :] / 0.7
        recent = set(idx[0, -50:].tolist())
        for tid in recent:
            if logits[0, tid] > 0:
                logits[0, tid] /= 1.3
            else:
                logits[0, tid] *= 1.3
        v, _ = torch.topk(logits, 40)
        logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nid = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nid], dim=1)
        if nid.item() == 2:
            break
    model.train()
    return sp.decode(idx[0].tolist())


best_val = float("inf")
t0 = time.time()
rl, rc = 0.0, 0

log(f"\nStarting finetune+CogitLayer: {MAX_ITERS} iters, LR={LR}")
for it in range(MAX_ITERS + 1):
    if stop:
        break

    if it < WARMUP:
        lr = LR * (it + 1) / WARMUP
    else:
        p = min((it - WARMUP) / max(MAX_ITERS - WARMUP, 1), 1.0)
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * p))

    for i, pg in enumerate(opt.param_groups):
        pg["lr"] = lr * (0.1 if i == 0 else 1.0)

    # Sync codebook periodically
    if it % 100 == 0:
        cogit.update_codebook(model.codebook)

    opt.zero_grad(set_to_none=True)
    al = 0.0
    for _ in range(ACCUM):
        X, Y = get_batch("train")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = forward_cogit(X, Y)
            loss = loss / ACCUM
        loss.backward()
        al += loss.item()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    rl += al
    rc += 1

    if it % 25 == 0 and it > 0:
        ms = (time.time() - t0) / it * 1000
        log(f"iter {it:>5d} | loss {rl / rc:.4f} | lr {lr:.2e} | grad {gn:.2f} | {ms:.0f}ms/step")
        rl, rc = 0.0, 0

    if it > 0 and it % 500 == 0:
        tl, vl = evaluate()
        bpb = vl / 0.6931
        best = vl < best_val
        if best:
            best_val = vl
            torch.save({
                "model": model.state_dict(), "config": cfg,
                "val_loss": vl, "iter": it, "vocab_size": V, "finetune": True,
            }, "best_finetune_v14_1.pt")
        torch.save({
            "model": model.state_dict(), "config": cfg,
            "val_loss": vl, "iter": it, "vocab_size": V, "finetune": True,
        }, "last_finetune_v14_1.pt")
        log(f"{'=' * 60}")
        log(f"[{(time.time() - t0) / 3600:.1f}h, iter {it}] Train:{tl:.4f} Val:{vl:.4f} BPB:{bpb:.3f} {'>>> BEST!' if best else ''}")
        log(f"{'=' * 60}")

    if it > 0 and it % 1000 == 0:
        log(f"\n--- Gen (iter {it}) ---")
        for pr in [
            "### Instruction: Explain what is artificial intelligence.\n### Response:",
            "### Instruction: What is the capital of France?\n### Response:",
            "### Instruction: Write a short poem about the ocean.\n### Response:",
            "### Instruction: How does a car engine work?\n### Response:",
        ]:
            r = gen(pr)
            resp = r.split("### Response:")[-1].strip() if "### Response:" in r else r
            log(f"  Q: {pr.split(chr(10))[0][20:]}")
            log(f"  A: {resp[:250]}\n")
        log("---")

log(f"\nDone at iter {it}. Best BPB: {best_val / 0.6931:.3f}")
torch.save({
    "model": model.state_dict(), "config": cfg,
    "val_loss": best_val, "iter": it, "vocab_size": V,
}, "last_finetune_v14_1.pt")
