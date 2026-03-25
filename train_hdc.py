"""Train HDC-AM v6.0 on Russian char-level."""
import time, signal, os, numpy as np, torch, torch.nn.functional as F
from hdc_am import create_hdc_am
from hoffman_agent import FixedCharTokenizer

stop_training = False
def handler(signum, frame):
    global stop_training
    print("\n[Timer] Done.")
    stop_training = True
if os.name != 'nt':
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(7200)

CONFIG = {
    'hdc_dim': 4096,
    'codebook_size': 4096,
    'nav_hidden': 512,
    'nav_layers': 4,
    'decay': 0.95,
    'dropout': 0.1,
}
BATCH = 24
GRAD_ACCUM = 5
SEQ_LEN = 256
LR = 3e-4
WARMUP = 500
USE_AMP = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

tok = FixedCharTokenizer()
print(f"Vocab: {tok.vocab_size}")

print("Loading data...")
all_text = ""
for f in sorted(os.listdir('.')):
    if f.startswith('data_ru_') and f.endswith('.txt') and os.path.getsize(f) > 0:
        with open(f, 'r', encoding='utf-8') as fh:
            t = fh.read()
            all_text += t + "\n\n"
            print(f"  {f}: {len(t):,}")

all_ids = np.array(tok.encode(all_text), dtype=np.uint8)
n = len(all_ids)
train_ids = all_ids[:int(n * 0.9)]
val_ids = all_ids[int(n * 0.9):]
train_ids.tofile('train_hdc.bin')
val_ids.tofile('val_hdc.bin')
print(f"  Train: {len(train_ids):,}, Val: {len(val_ids):,}")

def get_batch(split):
    data = np.fromfile(f'{split}_hdc.bin', dtype=np.uint8)
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH,))
    x = torch.stack([torch.from_numpy(data[i:i+SEQ_LEN].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+SEQ_LEN+1].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

model, config = create_hdc_am(tok.vocab_size, CONFIG)
model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())

BEST = 'best_hdc_am.pt'
LAST = 'last_hdc_am.pt'

def save_clean(m, path):
    sd = {k.replace('_orig_mod.', ''): v for k, v in m.state_dict().items()}
    torch.save(sd, path)

loaded = False
for cp in [BEST, LAST]:
    if os.path.exists(cp):
        try:
            sd = torch.load(cp, map_location=DEVICE, weights_only=True)
            sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
            missing, _ = model.load_state_dict(sd, strict=False)
            if missing: print(f"  New: {len(missing)}")
            print(f"Loaded: {cp}")
            loaded = True
            break
        except Exception as e:
            print(f"Skip {cp}: {e}")
if not loaded:
    print("From scratch")

scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and DEVICE == 'cuda'))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200000, eta_min=1e-5)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(20)
        for k in range(20):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out['train'], out['val'], out['val'] / np.log(2)

def generate(prompt="Россия ", n=200):
    model.eval()
    ids = tok.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    for _ in range(n):
        if ctx.size(1) > SEQ_LEN: ctx = ctx[:, -SEQ_LEN:]
        logits, _ = model(ctx)
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        idx = torch.multinomial(probs, 1)
        ctx = torch.cat([ctx, idx], dim=1)
    model.train()
    return tok.decode(ctx[0].cpu().tolist())

start = time.time()
step = 0
best_bpb = float('inf')

print(f"\n{'='*60}")
print(f"HDC-AM v6.0: Hyperdimensional Associative Manifold")
print(f"  HDC: {config['hdc_dim']}d, Codebook: {config['codebook_size']}")
print(f"  Navigator: {config['nav_hidden']}d x {config['nav_layers']} layers")
print(f"  {n_params:,} params | Data: {len(all_text):,} chars")
print(f"  Ratio: {len(train_ids)/n_params:.1f}x")
print(f"{'='*60}\n")

while not stop_training:
    t0 = time.time()
    if step < WARMUP:
        lr = LR * (step + 1) / WARMUP
        for pg in optimizer.param_groups: pg['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    acc_loss = 0
    for _ in range(GRAD_ACCUM):
        X, Y = get_batch('train')
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            _, loss = model(X, Y)
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        acc_loss += loss.item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    if step >= WARMUP: scheduler.step()

    dt = time.time() - t0
    if step % 50 == 0:
        print(f"iter {step}: loss {acc_loss:.4f}, {dt*1000:.0f}ms, lr {optimizer.param_groups[0]['lr']:.2e}")

    if step % 500 == 0 and step > 0:
        tl, vl, bpb = estimate_loss()
        sample = generate()
        if bpb < best_bpb:
            best_bpb = bpb
            save_clean(model, BEST)
            print("  >>> RECORD!")
        elapsed = int(time.time() - start)
        print("-" * 50)
        print(f"[{elapsed//60}min] Train: {tl:.4f} | Val: {vl:.4f} | BPB: {bpb:.4f} | Best: {best_bpb:.4f} | Gap: {tl-vl:.3f}")
        print(f"  Gen: '{sample[:200]}'")
        print("-" * 50)

    step += 1

tl, vl, bpb = estimate_loss()
sample = generate()
if bpb < best_bpb:
    best_bpb = bpb
    save_clean(model, BEST)
save_clean(model, LAST)
print(f"\n{'='*50}")
print(f"FINAL: {(time.time()-start)/60:.1f}min, {step} iter")
print(f"  BPB: {bpb:.4f} | Best: {best_bpb:.4f}")
print(f"  '{sample[:200]}'")
print("=" * 50)
