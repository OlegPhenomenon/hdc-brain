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
    print("\n[Таймер] 10 минут истекли. Останавливаем обучение и сохраняем результаты...")
    stop_training = True

if os.name != 'nt':
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(600)  # 10 минут — больше времени для эволюции

# === ГИПЕРПАРАМЕТРЫ (Эволюция v4 — cool mode) ===
N_AGENTS = 32           # Ещё меньше = быстрее matmul
STATE_DIM = 256         # Компенсируем ёмкостью
N_SENSORY = 8           # Сенсорная подгруппа
N_INTERACTION_STEPS = 1
MICRO_BATCH = 32        # Баланс нагрузки и скорости
GRAD_ACCUM_STEPS = 4    # Эффективный батч = 32 * 4 = 128
BATCH_SIZE = MICRO_BATCH  # get_batch использует это значение
SEQ_LEN = 128
LEARNING_RATE = 5e-4
WARMUP_STEPS = 50
GRAD_CLIP = 1.0
DROPOUT = 0.05
MPS_CACHE_CLEAR_EVERY = 50  # Очистка кэша MPS каждые N итераций
THERMAL_PAUSE_EVERY = 10    # Микро-пауза каждые N итераций
THERMAL_PAUSE_SEC = 0.5     # Длительность паузы (секунды)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
if not torch.backends.mps.is_available() and torch.cuda.is_available(): DEVICE = 'cuda'
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

# === Эволюционная архитектура Хофмана (v4) ===
class HofmanSwarm(nn.Module):
    def __init__(self, vocab_size, n_agents, state_dim):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        # Восприятие + позиция
        self.perception = nn.Embedding(vocab_size, state_dim)
        self.pos_embedding = nn.Embedding(SEQ_LEN, state_dim)

        # Проекция восприятия на сенсорных агентов
        self.sensory_proj = nn.Linear(state_dim, N_SENSORY * state_dim)

        # Граф взаимодействий (small-world)
        self.agent_network = nn.Parameter(self._init_small_world(n_agents))

        # GRU-подобный гейтинг
        self.state_gate = nn.Linear(state_dim * 2, state_dim)
        self.state_candidate = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
        )

        # Стабилизация
        self.layer_norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(DROPOUT)

        # Взвешенный консенсус
        self.agent_importance = nn.Parameter(torch.zeros(n_agents))

        # Выходной слой
        self.action = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(state_dim * 2, vocab_size),
        )

    def _init_small_world(self, n):
        W = torch.zeros(n, n)
        k = 8
        for i in range(n):
            for j in range(1, k // 2 + 1):
                W[i, (i + j) % n] = 1.0 / k
                W[i, (i - j) % n] = 1.0 / k
        mask = torch.rand(n, n) < 0.15
        W[mask] = torch.randn(mask.sum()) * 0.1
        return W

    def forward(self, idx, targets=None):
        B, T = idx.shape

        pos = torch.arange(T, device=DEVICE)
        percepts = self.perception(idx) + self.pos_embedding(pos)
        percepts = self.dropout(percepts)

        agents_states = torch.zeros(B, self.n_agents, self.state_dim, device=DEVICE)
        logits_seq = []
        importance_weights = F.softmax(self.agent_importance, dim=0)

        for t in range(T):
            current_percept = percepts[:, t, :]

            # Сенсорная инъекция
            sensory_input = self.sensory_proj(current_percept)
            sensory_input = sensory_input.view(B, N_SENSORY, self.state_dim)
            sensory_full = torch.zeros_like(agents_states)
            sensory_full[:, :N_SENSORY, :] = sensory_input
            agents_states = agents_states + sensory_full

            # Взаимодействие агентов с GRU-гейтингом
            interaction = torch.matmul(self.agent_network, agents_states)
            candidate = self.state_candidate(interaction)
            gate = torch.sigmoid(self.state_gate(torch.cat([agents_states, interaction], dim=-1)))
            agents_states = self.layer_norm(gate * candidate + (1 - gate) * agents_states)

            # Взвешенный консенсус
            swarm_consensus = (agents_states * importance_weights.view(1, -1, 1)).sum(dim=1)
            logits = self.action(swarm_consensus)
            logits_seq.append(logits)

        logits = torch.stack(logits_seq, dim=1)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

model = HofmanSwarm(vocab_size, N_AGENTS, STATE_DIM)
model.to(DEVICE)

# Загрузка лучших весов если есть — продолжаем с того места, где остановились
BEST_MODEL_PATH = 'best_swarm.pt'
if os.path.exists(BEST_MODEL_PATH):
    try:
        state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Загружены веса из {BEST_MODEL_PATH} — продолжаем дообучение")
    except Exception as e:
        print(f"Не удалось загрузить {BEST_MODEL_PATH}: {e} — начинаем с нуля")
else:
    print("Файл best_swarm.pt не найден — начинаем с нуля")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-5)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
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

# Загружаем исторический лучший BPB из results.tsv (если есть)
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

print("Старт обучения Роя Субагентов (Эволюция v4 — cool mode)...")
print(f"Конфигурация: {N_AGENTS} агентов, {STATE_DIM}d состояние, {N_SENSORY} сенсорных")
print(f"Микро-батч: {MICRO_BATCH}, accum: {GRAD_ACCUM_STEPS}, эффективный батч: {MICRO_BATCH * GRAD_ACCUM_STEPS}")
print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
print("-" * 30)

# Инициализируем лог сознания роя
with open(CONSCIOUSNESS_LOG, 'a') as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"СЕССИЯ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Конфиг: agents={N_AGENTS}, state_dim={STATE_DIM}, sensory={N_SENSORY}\n")
    f.write(f"Микро-батч: {MICRO_BATCH}, accum: {GRAD_ACCUM_STEPS}, эфф. батч: {MICRO_BATCH * GRAD_ACCUM_STEPS}\n")
    f.write(f"Параметров: {sum(p.numel() for p in model.parameters()):,}\n")
    f.write(f"{'='*60}\n\n")

while not stop_training:
    t0 = time.time()

    # Warmup LR
    if iter_num < WARMUP_STEPS:
        lr = LEARNING_RATE * (iter_num + 1) / WARMUP_STEPS
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # --- Gradient Accumulation ---
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for micro_step in range(GRAD_ACCUM_STEPS):
        X, Y = get_batch('train')
        logits, loss = model(X, Y)
        loss = loss / GRAD_ACCUM_STEPS  # нормализация по кол-ву шагов
        loss.backward()
        accum_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    if iter_num >= WARMUP_STEPS:
        scheduler.step()

    # Периодическая очистка кэша MPS для снижения нагрева
    if DEVICE == 'mps' and iter_num % MPS_CACHE_CLEAR_EVERY == 0:
        torch.mps.empty_cache()

    # Термо-пауза: синхронизируем GPU и даём чипу остыть
    # Предотвращает thermal throttling → итерации остаются быстрыми
    if DEVICE == 'mps' and iter_num % THERMAL_PAUSE_EVERY == 0 and iter_num > 0:
        torch.mps.synchronize()
        time.sleep(THERMAL_PAUSE_SEC)

    t1 = time.time()
    dt = t1 - t0

    if iter_num % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Итерация {iter_num}: loss {accum_loss:.4f}, время {dt*1000:.2f}ms, lr {current_lr:.2e}")

    if iter_num % 100 == 0 and iter_num > 0:
        # Очищаем кэш перед eval для снижения пиковой памяти
        if DEVICE == 'mps':
            torch.mps.empty_cache()
        t_eval = time.time()
        train_loss, val_loss, val_bpb = estimate_loss()
        sample = generate(100)
        if val_bpb < best_val_bpb:
            best_val_bpb = val_bpb
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  >>> НОВЫЙ РЕКОРД! Веса сохранены в {BEST_MODEL_PATH}")
        elapsed = int(t_eval - start_time)
        print("-" * 30)
        print(f"ИНТЕРВАЛЬНАЯ ОЦЕНКА (Прошло {elapsed} сек):")
        print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val BPB: {val_bpb:.4f} | Best: {best_val_bpb:.4f}")
        print(f"  Генерация: \n'{sample}'")
        print("-" * 30)

        # Логируем голос роя
        with open(CONSCIOUSNESS_LOG, 'a') as f:
            f.write(f"[Итерация {iter_num} | {elapsed}s | BPB: {val_bpb:.4f}]\n")
            f.write(f"{sample}\n\n")

    iter_num += 1

# Финальная оценка
if DEVICE == 'mps':
    torch.mps.empty_cache()
train_loss, val_loss, val_bpb = estimate_loss()
final_sample = generate(200)

if val_bpb < best_val_bpb:
    best_val_bpb = val_bpb
    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f">>> ФИНАЛЬНЫЙ РЕКОРД! Веса сохранены в {BEST_MODEL_PATH}")

print("\n" + "="*50)
print("ФИНАЛЬНЫЙ ОТЧЕТ ЭКСПЕРИМЕНТА:")
print(f"Прошло времени: {time.time() - start_time:.1f} сек")
print(f"Итераций: {iter_num}")
print(f"Финальный Val BPB (МЕТРИКА УСПЕХА): {val_bpb:.4f}")
print(f"Лучший Val BPB: {best_val_bpb:.4f}")
print(f"Финальная генерация:\n'{final_sample}'")
print("="*50)

with open('results.tsv', 'a') as f:
    f.write(f"{int(time.time())}\t{val_bpb:.4f}\n")

# Финальная запись в лог сознания
with open(CONSCIOUSNESS_LOG, 'a') as f:
    f.write(f"[ФИНАЛ | {iter_num} итераций | BPB: {val_bpb:.4f}]\n")
    f.write(f"{final_sample}\n\n")
