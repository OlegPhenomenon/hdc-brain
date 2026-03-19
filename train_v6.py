import time
import signal
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Глобальный флаг для остановки по таймеру
stop_training = False

def handler(signum, frame):
    global stop_training
    print("\n[Таймер] Время вышло. Останавливаем обучение и сохраняем результаты...")
    stop_training = True

if os.name != 'nt':
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(14400)  # 4 часа

# === ГИПЕРПАРАМЕТРЫ (v8 — Long Training + Light Regularization) ===
# Стратегия: простая архитектура v6 + длинное обучение + label smoothing
N_AGENTS = 64           # 64 агента — максимум ёмкости
STATE_DIM = 256
N_SENSORY = 16
N_CHANNELS = 2
N_INTERACTION_STEPS = 1 # 1 раунд — быстрее итерации, больше итераций за час
MICRO_BATCH = 16
GRAD_ACCUM_STEPS = 8    # Эффективный батч = 128
BATCH_SIZE = MICRO_BATCH
SEQ_LEN = 128
LEARNING_RATE = 5e-4
WARMUP_STEPS = 200
GRAD_CLIP = 1.0
DROPOUT = 0.12
WEIGHT_DECAY = 0.02
LABEL_SMOOTHING = 0.05
EVAL_BATCHES = 30

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Используется устройство: {DEVICE}")

# Загрузка метаданных
if not os.path.exists('meta.pkl'):
    print("Ошибка: Запустите сначала python prepare.py")
    exit(1)
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
itos = meta['itos']

def get_batch(split):
    filename = 'train.bin' if split == 'train' else 'val.bin'
    data = np.fromfile(filename, dtype=np.uint16)
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+SEQ_LEN]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+SEQ_LEN+1]).astype(np.int64)) for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


# === Архитектура v8 — Простой Multi-Channel Swarm + регуляризация ===

class SwarmChannel(nn.Module):
    """Один канал коммуникации роя: граф связей + GRU-гейтинг"""
    def __init__(self, n_agents, state_dim, connectivity, rewire_prob):
        super().__init__()
        self.agent_network = nn.Parameter(
            self._init_graph(n_agents, connectivity, rewire_prob)
        )
        self.state_gate = nn.Linear(state_dim * 2, state_dim)
        self.state_candidate = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(state_dim)

    def _init_graph(self, n, k, rewire_prob):
        """Small-world граф (Watts-Strogatz)"""
        W = torch.zeros(n, n)
        for i in range(n):
            for j in range(1, k // 2 + 1):
                W[i, (i + j) % n] = 1.0 / k
                W[i, (i - j) % n] = 1.0 / k
        mask = torch.rand(n, n) < rewire_prob
        W[mask] = torch.randn(mask.sum()) * 0.1
        return W

    def forward(self, agents_states):
        interaction = torch.matmul(self.agent_network, agents_states)
        candidate = self.state_candidate(interaction)
        gate = torch.sigmoid(self.state_gate(
            torch.cat([agents_states, interaction], dim=-1)
        ))
        return self.layer_norm(gate * candidate + (1 - gate) * agents_states)


class HofmanSwarm(nn.Module):
    def __init__(self, vocab_size, n_agents, state_dim):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        self.perception = nn.Embedding(vocab_size, state_dim)
        self.pos_embedding = nn.Embedding(SEQ_LEN, state_dim)
        self.sensory_proj = nn.Linear(state_dim, N_SENSORY * state_dim)

        self.channels = nn.ModuleList([
            SwarmChannel(n_agents, state_dim, connectivity=10, rewire_prob=0.05),
            SwarmChannel(n_agents, state_dim, connectivity=4, rewire_prob=0.25),
        ])

        self.channel_mix = nn.Parameter(torch.ones(N_CHANNELS) / N_CHANNELS)
        self.dropout = nn.Dropout(DROPOUT)
        self.agent_importance = nn.Parameter(torch.zeros(n_agents))

        self.action = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(state_dim * 2, vocab_size),
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        pos = torch.arange(T, device=DEVICE)
        percepts = self.perception(idx) + self.pos_embedding(pos)
        percepts = self.dropout(percepts)

        agents_states = torch.zeros(B, self.n_agents, self.state_dim, device=DEVICE)
        logits_seq = []
        importance_weights = F.softmax(self.agent_importance, dim=0)
        channel_weights = F.softmax(self.channel_mix, dim=0)

        for t in range(T):
            current_percept = percepts[:, t, :]

            sensory_input = self.sensory_proj(current_percept)
            sensory_input = sensory_input.view(B, N_SENSORY, self.state_dim)
            sensory_full = torch.zeros_like(agents_states)
            sensory_full[:, :N_SENSORY, :] = sensory_input
            agents_states = agents_states + sensory_full

            for _ in range(N_INTERACTION_STEPS):
                channel_outputs = [ch(agents_states) for ch in self.channels]
                agents_states = sum(
                    w * out for w, out in zip(channel_weights, channel_outputs)
                )

            agents_states = self.dropout(agents_states)

            swarm_consensus = (agents_states * importance_weights.view(1, -1, 1)).sum(dim=1)
            logits = self.action(swarm_consensus)
            logits_seq.append(logits)

        logits = torch.stack(logits_seq, dim=1)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                label_smoothing=LABEL_SMOOTHING
            )

        return logits, loss


model = HofmanSwarm(vocab_size, N_AGENTS, STATE_DIM)
model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())

BEST_MODEL_PATH = 'best_swarm.pt'
LAST_MODEL_PATH = 'last_swarm.pt'

loaded = False
for checkpoint_path in [BEST_MODEL_PATH, LAST_MODEL_PATH]:
    if os.path.exists(checkpoint_path) and not loaded:
        try:
            state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Загружены веса из {checkpoint_path} — продолжаем дообучение")
            loaded = True
        except Exception as e:
            print(f"Не удалось загрузить {checkpoint_path}: {e}")
if not loaded:
    print("Чекпоинты не найдены — начинаем с нуля")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15000, eta_min=1e-5)

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
    val_bpb = out['val'] / np.log(2)
    return out['train'], out['val'], val_bpb

def generate(max_new_tokens=100):
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    chars = []
    for _ in range(max_new_tokens):
        if context.size(1) > SEQ_LEN:
            context = context[:, -SEQ_LEN:]
        logits, _ = model(context)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, idx_next), dim=1)
        chars.append(itos[idx_next.item()])
    model.train()
    return "".join(chars)

# Цикл обучения
start_time = time.time()
iter_num = 0

best_val_bpb = float('inf')
if os.path.exists('results.tsv'):
    with open('results.tsv', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                try:
                    bpb = float(parts[1])
                    if bpb < best_val_bpb:
                        best_val_bpb = bpb
                except ValueError:
                    pass
    if best_val_bpb < float('inf'):
        print(f"Исторический лучший BPB: {best_val_bpb:.4f}")
    else:
        best_val_bpb = float('inf')

CONSCIOUSNESS_LOG = 'swarm_consciousness.log'
VERSION = 'v8'

print(f"Старт обучения Роя Субагентов ({VERSION} — Long Training)...")
print(f"Конфигурация: {N_AGENTS} агентов, {STATE_DIM}d, {N_SENSORY} сенсорных, {N_CHANNELS} каналов, {N_INTERACTION_STEPS} раунд")
print(f"  Label smoothing={LABEL_SMOOTHING}, dropout={DROPOUT}, weight_decay={WEIGHT_DECAY}")
print(f"Микро-батч: {MICRO_BATCH}, accum: {GRAD_ACCUM_STEPS}, эфф. батч: {MICRO_BATCH * GRAD_ACCUM_STEPS}")
print(f"Параметров: {n_params:,}")
print(f"Таймер: 4 часа")
print("-" * 30)

with open(CONSCIOUSNESS_LOG, 'a') as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"СЕССИЯ {VERSION}: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Hoffman Swarm {VERSION}: agents={N_AGENTS}, state_dim={STATE_DIM}\n")
    f.write(f"  Каналы: {N_CHANNELS}, сенсорных: {N_SENSORY}, раундов: {N_INTERACTION_STEPS}\n")
    f.write(f"  Dropout: {DROPOUT}, WD: {WEIGHT_DECAY}, LS: {LABEL_SMOOTHING}\n")
    f.write(f"  Микро-батч: {MICRO_BATCH}, accum: {GRAD_ACCUM_STEPS}\n")
    f.write(f"  Параметров: {n_params:,}\n")
    f.write(f"  Таймер: 4 часа\n")
    f.write(f"{'='*60}\n\n")

while not stop_training:
    t0 = time.time()

    if iter_num < WARMUP_STEPS:
        lr = LEARNING_RATE * (iter_num + 1) / WARMUP_STEPS
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for micro_step in range(GRAD_ACCUM_STEPS):
        X, Y = get_batch('train')
        logits, loss = model(X, Y)
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        accum_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    if iter_num >= WARMUP_STEPS:
        scheduler.step()

    t1 = time.time()
    dt = t1 - t0

    if iter_num % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Итерация {iter_num}: loss {accum_loss:.4f}, время {dt*1000:.2f}ms, lr {current_lr:.2e}")

    if iter_num % 200 == 0 and iter_num > 0:
        t_eval = time.time()
        train_loss, val_loss, val_bpb = estimate_loss()
        sample = generate(100)
        if val_bpb < best_val_bpb:
            best_val_bpb = val_bpb
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  >>> НОВЫЙ РЕКОРД! Веса сохранены в {BEST_MODEL_PATH}")
        elapsed = int(t_eval - start_time)
        gap = train_loss - val_loss
        print("-" * 30)
        print(f"ОЦЕНКА (Прошло {elapsed//60}мин):")
        print(f"  Train: {train_loss:.4f} | Val: {val_loss:.4f} | BPB: {val_bpb:.4f} | Best: {best_val_bpb:.4f} | Gap: {gap:.4f}")
        print(f"  '{sample}'")
        print("-" * 30)

        with open(CONSCIOUSNESS_LOG, 'a') as f:
            f.write(f"[iter {iter_num} | {elapsed//60}min | BPB: {val_bpb:.4f} | gap: {gap:.4f}]\n")
            f.write(f"{sample}\n\n")

    iter_num += 1

# Финальная оценка
train_loss, val_loss, val_bpb = estimate_loss()
final_sample = generate(200)

if val_bpb < best_val_bpb:
    best_val_bpb = val_bpb
    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f">>> ФИНАЛЬНЫЙ РЕКОРД! Веса сохранены в {BEST_MODEL_PATH}")

torch.save(model.state_dict(), LAST_MODEL_PATH)
print(f"Последний чекпоинт сохранён в {LAST_MODEL_PATH}")

print("\n" + "="*50)
print(f"ФИНАЛЬНЫЙ ОТЧЕТ ({VERSION}):")
print(f"Время: {(time.time() - start_time)/60:.1f} мин | Итераций: {iter_num}")
print(f"BPB: {val_bpb:.4f} | Best: {best_val_bpb:.4f}")
print(f"'{final_sample}'")
print("="*50)

with open('results.tsv', 'a') as f:
    f.write(f"{int(time.time())}\t{val_bpb:.4f}\n")

with open(CONSCIOUSNESS_LOG, 'a') as f:
    f.write(f"[ФИНАЛ {VERSION} | {iter_num} iter | BPB: {val_bpb:.4f}]\n")
    f.write(f"{final_sample}\n\n")
