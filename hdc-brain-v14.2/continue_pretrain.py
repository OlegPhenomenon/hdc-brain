"""
Continue pretraining HDC-Brain v14.1 on FineWeb-Edu for another ~3B tokens.

Goal: push base BPB from 5.434 toward 4.8-5.0 by giving the model more
language exposure (Chinchilla-optimal for 300M ≈ 6B tokens, we were at 3B).

Strategy:
- Load best_hdc_brain_v14_1.pt
- Fresh optimizer (don't inherit possibly-stale AdamW state)
- Cosine LR 1e-4 -> 1e-5 with 500-step warmup
- 60,000 iters (~24h on RTX 3090, ~3.9B tokens seen)
- Save to best_hdc_brain_v14_1_ext.pt (don't overwrite original!)
"""
import json, time, signal, os, math
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from hdc_brain_v14_1 import create_model

LOG_FILE = "continue_pretrain.log"
JSONL = "continue_pretrain.jsonl"


def log(msg=""):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def log_json(evt, data):
    with open(JSONL, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                            "event": evt, **data}) + "\n")


stop = False


def handler(s, f):
    global stop
    stop = True


signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

# Config — tuned for speed (no gradient checkpoint, smaller batch)
BATCH, ACCUM, SEQ = 8, 8, 512  # 32768 tokens/step (half but 2x faster)
LR, LR_MIN, WARMUP = 1e-4, 1e-5, 500
MAX_ITERS = 30000  # at ~3-4s/step ≈ 24-30h
THOUGHTS = 3
EVAL_INTERVAL = 500
LOG_INTERVAL = 50
device = "cuda"

sp = spm.SentencePieceProcessor(model_file="bpe_en_32k.model")
V = sp.get_piece_size()

log("Loading FineWeb-Edu 3B...")
train_data = np.memmap("data_3b/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data_3b/val.bin", dtype=np.uint16, mode="r")
log(f"  Train: {len(train_data):,}  Val: {len(val_data):,}")

model, cfg = create_model(V, {
    "hdc_dim": 4096, "max_seq_len": 512, "n_blocks": 8,
    "controller_dim": 2560, "n_heads": 4, "dropout": 0.1,
    "max_thoughts": 4, "use_checkpoint": False,  # disabled for speed (2x faster)
})
model = model.to(device)

ckpt = torch.load("best_hdc_brain_v14_1.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
starting_bpb = ckpt.get("val_loss", 3.77) / 0.6931
log(f"Loaded base: BPB {starting_bpb:.3f} (from iter {ckpt.get('iter', '?')})")
del ckpt

opt = torch.optim.AdamW([
    {"params": [model.codebook], "lr": LR * 0.1, "weight_decay": 0.0},
    {"params": [p for n, p in model.named_parameters() if n != "codebook"],
     "lr": LR, "weight_decay": 0.05},
], betas=(0.9, 0.95))


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(d) - SEQ - 1, size=(BATCH,))
    X = torch.stack([torch.from_numpy(d[i:i + SEQ].astype(np.int64).copy()) for i in ix]).to(device, non_blocking=True)
    Y = torch.stack([torch.from_numpy(d[i + 1:i + SEQ + 1].astype(np.int64).copy()) for i in ix]).to(device, non_blocking=True)
    return X, Y


@torch.no_grad()
def evaluate():
    model.eval()
    losses = {"train": [], "val": []}
    for s in ["train", "val"]:
        for _ in range(20):
            X, Y = get_batch(s)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(X, targets=Y, n_thoughts=THOUGHTS)
            losses[s].append(loss.item())
    model.train()
    return float(np.mean(losses["train"])), float(np.mean(losses["val"]))


best_val = float("inf")
t0 = time.time()
rl, rc = 0.0, 0

log(f"\nCONTINUE PRETRAIN: {MAX_ITERS} iters, LR={LR}->{LR_MIN}")
log_json("start", {"max_iters": MAX_ITERS, "lr_peak": LR, "starting_bpb": starting_bpb})

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

    opt.zero_grad(set_to_none=True)
    al = 0.0
    for _ in range(ACCUM):
        X, Y = get_batch("train")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(X, targets=Y, n_thoughts=THOUGHTS)
            loss = loss / ACCUM
        loss.backward()
        al += loss.item()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    rl += al
    rc += 1

    if it % LOG_INTERVAL == 0 and it > 0:
        ms = (time.time() - t0) / it * 1000
        log(f"iter {it:>5d} | loss {rl / rc:.4f} | lr {lr:.2e} | grad {gn:.2f} | {ms:.0f}ms/step")
        rl, rc = 0.0, 0

    if it > 0 and it % EVAL_INTERVAL == 0:
        tl, vl = evaluate()
        bpb = vl / 0.6931
        best = vl < best_val
        if best:
            best_val = vl
            torch.save({
                "model": model.state_dict(), "config": cfg,
                "val_loss": vl, "iter": it, "vocab_size": V,
                "phase": "continue_pretrain",
            }, "best_hdc_brain_v14_1_ext.pt")
        torch.save({
            "model": model.state_dict(), "config": cfg,
            "val_loss": vl, "iter": it, "vocab_size": V,
            "phase": "continue_pretrain",
        }, "last_hdc_brain_v14_1_ext.pt")
        log("=" * 60)
        log(f"[{(time.time() - t0) / 3600:.1f}h, iter {it}] Train:{tl:.4f} Val:{vl:.4f} BPB:{bpb:.3f} {'>>> BEST!' if best else ''}")
        log("=" * 60)
        log_json("eval", {"iter": it, "train": tl, "val": vl, "bpb": bpb, "best": best})

log(f"\nDone at iter {it}. Best BPB: {best_val / 0.6931:.3f} (started from {starting_bpb:.3f})")
torch.save({
    "model": model.state_dict(), "config": cfg,
    "val_loss": best_val, "iter": it, "vocab_size": V, "phase": "continue_pretrain",
}, "last_hdc_brain_v14_1_ext.pt")
log_json("done", {"iter": it, "best_bpb": best_val / 0.6931})
