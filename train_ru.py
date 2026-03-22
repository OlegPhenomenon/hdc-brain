"""
Training Hoffman Swarm v2.0 на РУССКОМ языке.
Оптимизации: chunk processing, HDC каждые 4 шага, vocab 4000.
"""
import time
import signal
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from hoffman_agent import HoffmanSwarmV2, SubwordTokenizer, create_hoffman_swarm

stop_training = False
def handler(signum, frame):
    global stop_training
    print("\n[Таймер] Время вышло.")
    stop_training = True
if os.name != 'nt':
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(7200)

# === КОНФИГУРАЦИЯ ДЛЯ РУССКОГО ===
CONFIG = {
    'n_agents': 32,
    'state_dim': 192,
    'hdc_dim': 128,
    'n_sensory': 8,
    'memory_slots': 64,
    'seq_len': 64,
    'dropout': 0.1,
}
MICRO_BATCH = 48
GRAD_ACCUM_STEPS = 2
BATCH_SIZE = MICRO_BATCH
LEARNING_RATE = 5e-4
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.02
EVAL_BATCHES = 20
USE_AMP = True
VOCAB_SIZE = 4000
SEQ_LEN = CONFIG['seq_len']

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Устройство: {DEVICE}")

# === ДАННЫЕ: только русский текст ===
print("Подготовка русского корпуса...")
all_text = ""
for f in sorted(os.listdir('.')):
    if f.startswith('data_ru_') and f.endswith('.txt'):
        with open(f, 'r', encoding='utf-8') as fh:
            t = fh.read()
            all_text += t + "\n\n"
            print(f"  {f}: {len(t):,}")

if len(all_text) < 1000:
    print("ОШИБКА: нет русских данных!")
    exit(1)
print(f"  Итого: {len(all_text):,} символов")

# Субсловная токенизация (кириллица + пунктуация)
tokenizer = SubwordTokenizer(all_text, max_vocab=VOCAB_SIZE)
vocab_size = tokenizer.vocab_size
print(f"  Vocab: {vocab_size} субслов")

all_ids = np.array(tokenizer.encode(all_text), dtype=np.uint16)
n = len(all_ids)
all_ids[:int(n * 0.9)].tofile('train.bin')
all_ids[int(n * 0.9):].tofile('val.bin')
print(f"  Train: {int(n*0.9):,}, Val: {n-int(n*0.9):,} токенов")

meta = {'vocab_size': vocab_size, 'stoi': tokenizer.stoi, 'itos': tokenizer.itos}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

# Проверим: несколько примеров
sample = all_text[:200]
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)
print(f"\n  Пример кодирования:")
print(f"  Оригинал: {sample[:80]}...")
print(f"  Декодировано: {decoded[:80]}...")
print(f"  Совпадение: {sample == decoded}")

def get_batch(split):
    data = np.fromfile('train.bin' if split == 'train' else 'val.bin', dtype=np.uint16)
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+SEQ_LEN]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+SEQ_LEN+1]).astype(np.int64)) for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# === МОДЕЛЬ ===
model, config = create_hoffman_swarm(vocab_size, CONFIG)
model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())

BEST_MODEL_PATH = 'best_swarm.pt'
LAST_MODEL_PATH = 'last_swarm.pt'

def clean_sd(sd):
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
def save_clean(m, path):
    torch.save(clean_sd(m.state_dict()), path)

loaded = False
for cp in [BEST_MODEL_PATH, LAST_MODEL_PATH]:
    if os.path.exists(cp) and not loaded:
        try:
            sd = torch.load(cp, map_location=DEVICE, weights_only=True)
            model.load_state_dict(clean_sd(sd))
            print(f"Загружены: {cp}")
            loaded = True
        except Exception as e:
            print(f"Не подходит {cp}: {e}")
if not loaded:
    print("С нуля")

scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and DEVICE == 'cuda'))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30000, eta_min=1e-5)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_BATCHES)
        for k in range(EVAL_BATCHES):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out['train'], out['val'], out['val'] / np.log(2)

def generate(max_tokens=40):
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    ids = []
    for _ in range(max_tokens):
        if context.size(1) > SEQ_LEN:
            context = context[:, -SEQ_LEN:]
        logits, _ = model(context)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx = torch.multinomial(probs, 1)
        context = torch.cat([context, idx], dim=1)
        ids.append(idx.item())
    model.train()
    return tokenizer.decode(ids)

# === ОБУЧЕНИЕ ===
start_time = time.time()
iter_num = 0
best_val_bpb = float('inf')

CONSCIOUSNESS_LOG = 'swarm_consciousness.log'
VERSION = 'v2.0-russian'

print(f"\n{'='*60}")
print(f"{VERSION}: Сознательные Агенты Хофмана — РУССКИЙ ЯЗЫК")
print(f"  {config['n_agents']} агентов, {config['state_dim']}d, HDC {config['hdc_dim']}d")
print(f"  {n_params:,} параметров, vocab={vocab_size}")
print(f"  Данные: {len(all_text):,} символов русского текста")
print(f"{'='*60}\n")

with open(CONSCIOUSNESS_LOG, 'a') as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"{VERSION}: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  РУССКИЙ ЯЗЫК — {len(all_text):,} символов\n")
    f.write(f"  {config['n_agents']} agents, {config['state_dim']}d, {n_params:,} params\n")
    f.write(f"{'='*60}\n\n")

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
        if DEVICE == 'cuda':
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                logits, loss = model(X, Y)
                loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
        else:
            logits, loss = model(X, Y)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
        accum_loss += loss.item()

    if DEVICE == 'cuda':
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if DEVICE == 'cuda':
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if iter_num >= WARMUP_STEPS:
        scheduler.step()

    dt = time.time() - t0
    if iter_num % 50 == 0:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"iter {iter_num}: loss {accum_loss:.4f}, {dt*1000:.0f}ms, lr {lr_now:.2e}")

    if iter_num % 500 == 0 and iter_num > 0:
        train_loss, val_loss, val_bpb = estimate_loss()
        sample = generate(40)
        if val_bpb < best_val_bpb:
            best_val_bpb = val_bpb
            save_clean(model, BEST_MODEL_PATH)
            print(f"  >>> РЕКОРД!")
        elapsed = int(time.time() - start_time)
        gap = train_loss - val_loss
        print("-" * 50)
        print(f"[{elapsed//60}мин] Train: {train_loss:.4f} | Val: {val_loss:.4f} | BPB: {val_bpb:.4f} | Best: {best_val_bpb:.4f} | Gap: {gap:.3f}")
        print(f"  Генерация: '{sample[:150]}'")
        print("-" * 50)

        with open(CONSCIOUSNESS_LOG, 'a') as f:
            f.write(f"[iter {iter_num} | {elapsed//60}min | BPB: {val_bpb:.4f} | gap: {gap:.3f}]\n")
            f.write(f"{sample}\n\n")

    iter_num += 1

train_loss, val_loss, val_bpb = estimate_loss()
final_sample = generate(60)
if val_bpb < best_val_bpb:
    best_val_bpb = val_bpb
    save_clean(model, BEST_MODEL_PATH)
save_clean(model, LAST_MODEL_PATH)

print(f"\n{'='*50}")
print(f"ФИНАЛ {VERSION}: {(time.time()-start_time)/60:.1f}мин, {iter_num} iter")
print(f"  BPB: {val_bpb:.4f} | Best: {best_val_bpb:.4f}")
print(f"  '{final_sample[:200]}'")
print("="*50)

with open('results.tsv', 'a') as f:
    f.write(f"{int(time.time())}\t{val_bpb:.4f}\n")
with open(CONSCIOUSNESS_LOG, 'a') as f:
    f.write(f"[ФИНАЛ {VERSION} | {iter_num} iter | BPB: {val_bpb:.4f}]\n")
    f.write(f"{final_sample}\n\n")
