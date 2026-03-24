"""
Training Conscious Swarm v5.0 на русском языке (посимвольно).
FixedCharTokenizer (vocab=170). Чекпоинт никогда не сломается.
"""
import time, signal, os, numpy as np, torch, torch.nn.functional as F
from conscious_swarm import create_conscious_swarm
from hoffman_agent import FixedCharTokenizer

stop_training = False
def handler(signum, frame):
    global stop_training
    print("\n[Таймер] Время вышло.")
    stop_training = True
if os.name != 'nt':
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(7200)

CONFIG = {
    'n_agents': 32,
    'state_dim': 128,
    'hdc_dim': 64,
    'n_sensory': 8,
    'seq_len': 256,
    'dropout': 0.1,
}
MICRO_BATCH = 48
GRAD_ACCUM_STEPS = 4
BATCH_SIZE = MICRO_BATCH
LEARNING_RATE = 3e-4
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.02
EVAL_BATCHES = 20
USE_AMP = True
SEQ_LEN = CONFIG['seq_len']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Устройство: {DEVICE}")

tokenizer = FixedCharTokenizer()
VOCAB_SIZE = tokenizer.vocab_size
print(f"FixedCharTokenizer: vocab={VOCAB_SIZE}")

print("Подготовка корпуса...")
all_text = ""
for f in sorted(os.listdir('.')):
    if f.startswith('data_ru_') and f.endswith('.txt'):
        if os.path.getsize(f) > 0:
            with open(f, 'r', encoding='utf-8') as fh:
                t = fh.read()
                all_text += t + "\n\n"
                print(f"  {f}: {len(t):,}")

print(f"  Итого: {len(all_text):,} символов")
all_ids = np.array(tokenizer.encode(all_text), dtype=np.uint8)
unk = (all_ids == 0).sum()
print(f"  UNK: {unk:,} ({100*unk/len(all_ids):.1f}%)")
n = len(all_ids)
train_ids = all_ids[:int(n * 0.9)]
val_ids = all_ids[int(n * 0.9):]
train_ids.tofile('train_conscious.bin')
val_ids.tofile('val_conscious.bin')
print(f"  Train: {len(train_ids):,}, Val: {len(val_ids):,}")

def get_batch(split):
    data = np.fromfile(f'{split}_conscious.bin', dtype=np.uint8)
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i+SEQ_LEN].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+SEQ_LEN+1].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

model, config = create_conscious_swarm(VOCAB_SIZE, CONFIG)
model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())

BEST = 'best_conscious.pt'
LAST = 'last_conscious.pt'

def clean_sd(sd):
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
def save_clean(m, path):
    torch.save(clean_sd(m.state_dict()), path)

loaded = False
for cp in [BEST, LAST]:
    if os.path.exists(cp) and not loaded:
        try:
            sd = torch.load(cp, map_location=DEVICE, weights_only=True)
            sd = clean_sd(sd)
            missing, _ = model.load_state_dict(sd, strict=False)
            if missing: print(f"  New params: {len(missing)}")
            print(f"Loaded: {cp}")
            loaded = True
        except Exception as e:
            print(f"Skip {cp}: {e}")
if not loaded:
    print("From scratch")

scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and DEVICE == 'cuda'))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200000, eta_min=1e-5)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_BATCHES)
        for k in range(EVAL_BATCHES):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out['train'], out['val'], out['val'] / np.log(2)

def generate(prompt="Россия ", max_tokens=200):
    model.eval()
    ids = tokenizer.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    for _ in range(max_tokens):
        if ctx.size(1) > SEQ_LEN: ctx = ctx[:, -SEQ_LEN:]
        logits, _ = model(ctx)
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        idx = torch.multinomial(probs, 1)
        ctx = torch.cat([ctx, idx], dim=1)
    model.train()
    return tokenizer.decode(ctx[0].cpu().tolist())

start_time = time.time()
iter_num = 0
best_bpb = float('inf')
VERSION = 'v5.0-conscious-swarm'

print(f"\n{'='*60}")
print(f"{VERSION}: Conscious Swarm Intelligence")
print(f"  {config['n_agents']} independent agents, {config['state_dim']}d")
print(f"  {n_params:,} params, vocab={VOCAB_SIZE}")
print(f"  Data: {len(all_text):,} chars")
print(f"  Ratio: {len(train_ids)/n_params:.1f}x")
print(f"{'='*60}\n")

while not stop_training:
    t0 = time.time()
    if iter_num < WARMUP_STEPS:
        lr = LEARNING_RATE * (iter_num + 1) / WARMUP_STEPS
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for _ in range(GRAD_ACCUM_STEPS):
        X, Y = get_batch('train')
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            logits, loss = model(X, Y)
            loss = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
        accum_loss += loss.item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    if iter_num >= WARMUP_STEPS:
        scheduler.step()

    dt = time.time() - t0
    if iter_num % 50 == 0:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"iter {iter_num}: loss {accum_loss:.4f}, {dt*1000:.0f}ms, lr {lr_now:.2e}")

    if iter_num % 500 == 0 and iter_num > 0:
        train_loss, val_loss, val_bpb = estimate_loss()
        sample = generate()
        if val_bpb < best_bpb:
            best_bpb = val_bpb
            save_clean(model, BEST)
            print(f"  >>> RECORD!")
        elapsed = int(time.time() - start_time)
        gap = train_loss - val_loss
        print("-" * 50)
        print(f"[{elapsed//60}min] Train: {train_loss:.4f} | Val: {val_loss:.4f} | BPB: {val_bpb:.4f} | Best: {best_bpb:.4f} | Gap: {gap:.3f}")
        print(f"  Gen: '{sample[:200]}'")
        print("-" * 50)

    iter_num += 1

train_loss, val_loss, val_bpb = estimate_loss()
sample = generate()
if val_bpb < best_bpb:
    best_bpb = val_bpb
    save_clean(model, BEST)
save_clean(model, LAST)
print(f"\n{'='*50}")
print(f"FINAL {VERSION}: {(time.time()-start_time)/60:.1f}min, {iter_num} iter")
print(f"  BPB: {val_bpb:.4f} | Best: {best_bpb:.4f}")
print(f"  '{sample[:200]}'")
print("=" * 50)
