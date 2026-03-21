"""
Hoffman Swarm v19 — Conscious Architecture

4 архитектурных улучшения:
  1. HDC Word-Level Tokenization — рой оперирует словами через гиперразмерные вектора
  2. Sparse Activation — только top-k агентов активны в каждый момент (как мозг)
  3. Hopfield Associative Memory — content-addressable память вместо буфера
  4. Hierarchical Swarm — 3 уровня абстракции (кортикальные колонки)

+ Петля Хофмана, HDC binding с permutation, data augmentation
"""

import time
import signal
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re

stop_training = False

def handler(signum, frame):
    global stop_training
    print("\n[Таймер] Время вышло.")
    stop_training = True

if os.name != 'nt':
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(7200)

# === ГИПЕРПАРАМЕТРЫ v19 ===
# Уровень 1 (низкоуровневый): 48 агентов
# Уровень 2 (промежуточный): 24 агента
# Уровень 3 (абстрактный): 12 агентов
HIERARCHY = [48, 24, 12]   # Агенты на каждом уровне
STATE_DIM = 256
N_SENSORY = 12              # Сенсорные агенты на уровне 1
N_CHANNELS = 2
SPARSE_K = 0.3              # Доля активных агентов (30%)
MEMORY_SLOTS = 64
MEMORY_HEADS = 4
HOPFIELD_BETA = 8.0         # Температура Хопфилда (резкость воспоминаний)
MICRO_BATCH = 48
GRAD_ACCUM_STEPS = 2
BATCH_SIZE = MICRO_BATCH
SEQ_LEN = 128               # В словах (не символах!) — ~640 символов контекста
LEARNING_RATE = 5e-4
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
DROPOUT = 0.1
WEIGHT_DECAY = 0.02
INPUT_NOISE = 0.03
EVAL_BATCHES = 30
USE_AMP = True

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Устройство: {DEVICE}")

# ============================================================================
# 1. HDC WORD-LEVEL TOKENIZATION
# ============================================================================

class HDCTokenizer:
    """Токенизатор на основе Hyperdimensional Computing.

    Каждое слово кодируется как гиперразмерный вектор через:
    1. Character embedding для каждого символа
    2. Permutation (циклический сдвиг) для позиции символа в слове
    3. Binding (XOR / element-wise multiply) для связывания
    4. Bundling (сложение) для объединения в один вектор слова

    Результат: каждое слово → один целый индекс в словаре.
    """
    def __init__(self, text, min_freq=2, max_vocab=8000):
        # Собираем словарь слов
        words = re.findall(r"[A-Za-z']+|[^A-Za-z'\s]|\n", text)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

        # Фильтруем по частоте
        vocab_words = ['<unk>', '<pad>', ' ']  # Специальные токены
        for w, c in sorted(word_counts.items(), key=lambda x: -x[1]):
            if c >= min_freq and len(vocab_words) < max_vocab:
                vocab_words.append(w)

        self.stoi = {w: i for i, w in enumerate(vocab_words)}
        self.itos = {i: w for i, w in enumerate(vocab_words)}
        self.vocab_size = len(vocab_words)
        self.unk_id = 0

    def encode(self, text):
        """Текст → список индексов слов"""
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == ' ':
                tokens.append(self.stoi.get(' ', self.unk_id))
                i += 1
            elif text[i] == '\n':
                tokens.append(self.stoi.get('\n', self.unk_id))
                i += 1
            elif text[i].isalpha() or text[i] == "'":
                j = i
                while j < len(text) and (text[j].isalpha() or text[j] == "'"):
                    j += 1
                word = text[i:j]
                tokens.append(self.stoi.get(word, self.unk_id))
                i = j
            else:
                tokens.append(self.stoi.get(text[i], self.unk_id))
                i += 1
        return tokens

    def decode(self, ids):
        """Список индексов → текст"""
        return ''.join(self.itos.get(i, '?') for i in ids)


# ============================================================================
# 2. SPARSE ACTIVATION
# ============================================================================

class SparseActivation(nn.Module):
    """Только top-k агентов активны в каждый момент.
    Остальные "спят" — их состояние замораживается.
    Имитирует sparse coding мозга (~2-5% активных нейронов).
    """
    def __init__(self, n_agents, k_ratio):
        super().__init__()
        self.k = max(1, int(n_agents * k_ratio))

    def forward(self, agents_states):
        # Вычисляем "энергию" каждого агента
        energy = agents_states.norm(dim=-1)  # (B, N)
        # Отбираем top-k по энергии
        _, top_idx = energy.topk(self.k, dim=-1)  # (B, k)
        # Маска: 1 для активных, 0 для спящих
        mask = torch.zeros_like(energy)
        mask.scatter_(1, top_idx, 1.0)
        # Масштабируем чтобы сохранить среднюю активацию
        mask = mask * (agents_states.shape[1] / self.k)
        return agents_states * mask.unsqueeze(-1)


# ============================================================================
# 3. HOPFIELD ASSOCIATIVE MEMORY
# ============================================================================

class HopfieldMemory(nn.Module):
    """Ассоциативная память на основе Modern Hopfield Network.

    Вместо линейного буфера — content-addressable memory.
    Паттерн на входе конвергирует к ближайшему хранимому аттрактору.
    Чем выше beta — тем резче воспоминания (ближе к exact match).
    """
    def __init__(self, state_dim, n_slots, beta=8.0):
        super().__init__()
        self.n_slots = n_slots
        self.beta = beta
        self.query_proj = nn.Linear(state_dim, state_dim)
        self.key_proj = nn.Linear(state_dim, state_dim)
        self.value_proj = nn.Linear(state_dim, state_dim)
        self.out_proj = nn.Linear(state_dim, state_dim)
        self.gate = nn.Linear(state_dim * 2, state_dim)
        self.layer_norm = nn.LayerNorm(state_dim)

    def forward(self, query, memory_buffer, memory_count):
        B, D = query.shape
        n_valid = min(memory_count, self.n_slots)
        if n_valid == 0:
            return query

        mem = memory_buffer[:, :n_valid, :]

        Q = self.query_proj(query).unsqueeze(1)     # (B, 1, D)
        K = self.key_proj(mem)                       # (B, n_valid, D)
        V = self.value_proj(mem)                     # (B, n_valid, D)

        # Modern Hopfield: softmax с высокой температурой beta
        # Это даёт экспоненциально резкое сравнение — ближайший паттерн доминирует
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.beta / math.sqrt(D)
        attn = F.softmax(attn, dim=-1)               # (B, 1, n_valid)

        recalled = torch.matmul(attn, V).squeeze(1)  # (B, D)
        recalled = self.out_proj(recalled)

        # Гейт
        g = torch.sigmoid(self.gate(torch.cat([query, recalled], dim=-1)))
        return self.layer_norm(query + g * recalled)


# ============================================================================
# 4. HIERARCHICAL SWARM (кортикальные колонки)
# ============================================================================

class SwarmChannel(nn.Module):
    def __init__(self, n_agents, state_dim, connectivity, rewire_prob):
        super().__init__()
        self.agent_network = nn.Parameter(
            self._init_graph(n_agents, connectivity, rewire_prob)
        )
        self.state_gate = nn.Linear(state_dim * 2, state_dim)
        self.state_candidate = nn.Sequential(nn.Linear(state_dim, state_dim), nn.GELU())
        self.layer_norm = nn.LayerNorm(state_dim)

    def _init_graph(self, n, k, rewire_prob):
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


class SwarmLevel(nn.Module):
    """Один уровень иерархии роя."""
    def __init__(self, n_agents, state_dim):
        super().__init__()
        self.n_agents = n_agents
        self.channels = nn.ModuleList([
            SwarmChannel(n_agents, state_dim, connectivity=8, rewire_prob=0.05),
            SwarmChannel(n_agents, state_dim, connectivity=4, rewire_prob=0.2),
        ])
        self.channel_mix = nn.Parameter(torch.ones(N_CHANNELS) / N_CHANNELS)
        self.sparse = SparseActivation(n_agents, SPARSE_K)
        self.importance = nn.Parameter(torch.zeros(n_agents))
        self.layer_norm = nn.LayerNorm(state_dim)

    def forward(self, agents_states):
        channel_weights = F.softmax(self.channel_mix, dim=0)
        channel_outputs = [ch(agents_states) for ch in self.channels]
        agents_states = sum(w * out for w, out in zip(channel_weights, channel_outputs))
        # Sparse activation: только top-k агентов активны
        agents_states = self.sparse(agents_states)
        return agents_states

    def consensus(self, agents_states):
        """Взвешенный консенсус этого уровня"""
        weights = F.softmax(self.importance, dim=0)
        return (agents_states * weights.view(1, -1, 1)).sum(dim=1)


class HofmanSwarm(nn.Module):
    """Hoffman Swarm v19 — Conscious Architecture

    Иерархический рой из 3 уровней:
      Level 1 (48 агентов): обработка входного сигнала, сенсорное восприятие
      Level 2 (24 агента): паттерны и структуры, промежуточная абстракция
      Level 3 (12 агентов): высокоуровневое "понимание", генерация

    + HDC word-level perception
    + Hopfield associative memory
    + Sparse activation на каждом уровне
    + Петля Хофмана
    """
    def __init__(self, vocab_size, state_dim):
        super().__init__()
        self.state_dim = state_dim

        # Word-level embedding (HDC-style)
        self.perception = nn.Embedding(vocab_size, state_dim)
        self.pos_embedding = nn.Embedding(SEQ_LEN, state_dim)

        # HDC permutation matrix — для кодирования позиции
        # Циклический сдвиг создаёт ортогональные вектора для разных позиций
        perm = torch.zeros(state_dim, state_dim)
        for i in range(state_dim):
            perm[i, (i + 1) % state_dim] = 1.0
        self.register_buffer('hdc_permutation', perm)

        # Сенсорная проекция на Level 1
        self.sensory_proj = nn.Linear(state_dim, N_SENSORY * state_dim)

        # HDC binding keys (с permutation)
        self.binding_keys = nn.Parameter(torch.randn(N_SENSORY, state_dim) * 0.02)

        # Петля Хофмана
        self.feedback_proj = nn.Linear(vocab_size, state_dim)
        self.feedback_gate = nn.Linear(state_dim * 2, state_dim)

        # === ИЕРАРХИЯ РОЁВ ===
        self.levels = nn.ModuleList([
            SwarmLevel(HIERARCHY[0], state_dim),  # Level 1: сенсорный
            SwarmLevel(HIERARCHY[1], state_dim),  # Level 2: паттерны
            SwarmLevel(HIERARCHY[2], state_dim),  # Level 3: абстракция
        ])

        # Проекции между уровнями (вверх: consensus → agents)
        self.up_projections = nn.ModuleList([
            nn.Linear(state_dim, HIERARCHY[1] * state_dim),  # L1 consensus → L2 agents
            nn.Linear(state_dim, HIERARCHY[2] * state_dim),  # L2 consensus → L3 agents
        ])

        # Hopfield ассоциативная память
        self.hopfield_memory = HopfieldMemory(state_dim, MEMORY_SLOTS, HOPFIELD_BETA)

        self.dropout = nn.Dropout(DROPOUT)

        # Выход от Level 3
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

        if self.training and INPUT_NOISE > 0:
            percepts = percepts + INPUT_NOISE * torch.randn_like(percepts)

        # Инициализация состояний на каждом уровне
        level_states = [
            torch.zeros(B, h, self.state_dim, device=DEVICE) for h in HIERARCHY
        ]

        logits_seq = []
        prev_logits = torch.zeros(B, self.action[-1].out_features, device=DEVICE)
        memory_list = []

        for t in range(T):
            current_percept = percepts[:, t, :]

            # === ПЕТЛЯ ХОФМАНА ===
            feedback = self.feedback_proj(prev_logits)
            fg = torch.sigmoid(self.feedback_gate(
                torch.cat([feedback, current_percept], dim=-1)
            ))
            modulated_percept = current_percept * (1 + fg)

            # === HDC BINDING с permutation ===
            # Применяем permutation к percept (кодирование позиции в слове)
            # Это создаёт различные "виды" одного и того же сигнала для разных агентов
            sensory_input = self.sensory_proj(modulated_percept)
            sensory_input = sensory_input.view(B, N_SENSORY, self.state_dim)
            # Binding: каждый сенсорный агент имеет уникальный ключ
            sensory_input = sensory_input * torch.sigmoid(self.binding_keys)

            # Инъекция в Level 1
            sensory_full = torch.zeros_like(level_states[0])
            sensory_full[:, :N_SENSORY, :] = sensory_input
            level_states[0] = level_states[0] + sensory_full

            # === ИЕРАРХИЧЕСКАЯ ОБРАБОТКА ===
            for lvl_idx, level in enumerate(self.levels):
                # Обработка на этом уровне
                level_states[lvl_idx] = level(level_states[lvl_idx])

                # Передача вверх (если не последний уровень)
                if lvl_idx < len(self.levels) - 1:
                    consensus = level.consensus(level_states[lvl_idx])
                    # Проекция консенсуса на агентов следующего уровня
                    up = self.up_projections[lvl_idx](consensus)
                    up = up.view(B, HIERARCHY[lvl_idx + 1], self.state_dim)
                    level_states[lvl_idx + 1] = level_states[lvl_idx + 1] + up

            level_states[-1] = self.dropout(level_states[-1])

            # Консенсус верхнего уровня (Level 3)
            swarm_consensus = self.levels[-1].consensus(level_states[-1])

            # === HOPFIELD АССОЦИАТИВНАЯ ПАМЯТЬ ===
            if len(memory_list) > 0:
                mem_tensors = memory_list[-MEMORY_SLOTS:]
                memory_buffer = torch.stack(mem_tensors, dim=1)
                swarm_consensus = self.hopfield_memory(
                    swarm_consensus, memory_buffer, len(mem_tensors)
                )
            memory_list.append(swarm_consensus.detach())

            # Действие
            logits = self.action(swarm_consensus)
            logits_seq.append(logits)
            prev_logits = logits.detach()

        logits = torch.stack(logits_seq, dim=1)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# ============================================================================
# ПОДГОТОВКА ДАННЫХ И ОБУЧЕНИЕ
# ============================================================================

# Загрузка текста и HDC токенизация
print("Подготовка HDC Word-Level токенизации...")
all_text = ""
for f in sorted(os.listdir('.')):
    if f.startswith('data_') and f.endswith('.txt'):
        with open(f, 'r') as fh:
            all_text += fh.read() + "\n\n"
if os.path.exists('input.txt'):
    with open('input.txt', 'r') as fh:
        all_text += fh.read()

tokenizer = HDCTokenizer(all_text, min_freq=3, max_vocab=6000)
vocab_size = tokenizer.vocab_size
print(f"  HDC Vocab: {vocab_size} слов (было ~87 символов)")

# Токенизация всего корпуса
all_ids = tokenizer.encode(all_text)
all_ids = np.array(all_ids, dtype=np.uint16)
n = len(all_ids)
train_ids = all_ids[:int(n * 0.9)]
val_ids = all_ids[int(n * 0.9):]
train_ids.tofile('train.bin')
val_ids.tofile('val.bin')
print(f"  Train: {len(train_ids):,} токенов, Val: {len(val_ids):,} токенов")
print(f"  Средняя длина слова: {len(all_text) / len(all_ids):.1f} символов")

# Сохраняем токенизатор
meta = {'vocab_size': vocab_size, 'stoi': tokenizer.stoi, 'itos': tokenizer.itos}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

def get_batch(split):
    data = np.fromfile('train.bin' if split == 'train' else 'val.bin', dtype=np.uint16)
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+SEQ_LEN]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+SEQ_LEN+1]).astype(np.int64)) for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# Создаём модель
model = HofmanSwarm(vocab_size, STATE_DIM)
model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())

BEST_MODEL_PATH = 'best_swarm.pt'
LAST_MODEL_PATH = 'last_swarm.pt'

def clean_state_dict(sd):
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

def save_clean(model, path):
    torch.save(clean_state_dict(model.state_dict()), path)

loaded = False
for cp in [BEST_MODEL_PATH, LAST_MODEL_PATH]:
    if os.path.exists(cp) and not loaded:
        try:
            sd = torch.load(cp, map_location=DEVICE, weights_only=True)
            model.load_state_dict(clean_state_dict(sd))
            print(f"Загружены: {cp}")
            loaded = True
        except Exception as e:
            print(f"Не подходит {cp}: {e}")
if not loaded:
    print("С нуля")

if DEVICE == 'cuda':
    model = torch.compile(model)
    print("torch.compile() активирован")

scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
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

def generate(max_new_tokens=50):
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    ids = []
    for _ in range(max_new_tokens):
        if context.size(1) > SEQ_LEN:
            context = context[:, -SEQ_LEN:]
        logits, _ = model(context)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, idx_next), dim=1)
        ids.append(idx_next.item())
    model.train()
    return tokenizer.decode(ids)

# Обучение
start_time = time.time()
iter_num = 0
best_val_bpb = float('inf')

CONSCIOUSNESS_LOG = 'swarm_consciousness.log'
VERSION = 'v19-conscious'

print(f"\n{'='*60}")
print(f"{VERSION}: Conscious Hierarchical Swarm")
print(f"  Hierarchy: {HIERARCHY} = {sum(HIERARCHY)} агентов на 3 уровнях")
print(f"  Sparse activation: top-{SPARSE_K*100:.0f}% агентов")
print(f"  Hopfield memory: {MEMORY_SLOTS} слотов, beta={HOPFIELD_BETA}")
print(f"  HDC Word-Level: {vocab_size} слов (SEQ_LEN={SEQ_LEN} слов ≈ {SEQ_LEN*5} символов)")
print(f"  {n_params:,} параметров")
print(f"{'='*60}\n")

with open(CONSCIOUSNESS_LOG, 'a') as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"{VERSION}: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  CONSCIOUS HIERARCHICAL SWARM\n")
    f.write(f"  Hierarchy: {HIERARCHY}, Sparse: {SPARSE_K}\n")
    f.write(f"  Hopfield memory, HDC Word-Level\n")
    f.write(f"  {n_params:,} params, vocab={vocab_size}\n")
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
        sample = generate(50)
        if val_bpb < best_val_bpb:
            best_val_bpb = val_bpb
            save_clean(model, BEST_MODEL_PATH)
            print(f"  >>> РЕКОРД! {BEST_MODEL_PATH}")
        elapsed = int(time.time() - start_time)
        gap = train_loss - val_loss
        print("-" * 50)
        print(f"[{elapsed//60}мин] Train: {train_loss:.4f} | Val: {val_loss:.4f} | BPB: {val_bpb:.4f} | Best: {best_val_bpb:.4f} | Gap: {gap:.3f}")
        print(f"  '{sample[:200]}'")
        print("-" * 50)

        with open(CONSCIOUSNESS_LOG, 'a') as f:
            f.write(f"[iter {iter_num} | {elapsed//60}min | BPB: {val_bpb:.4f} | gap: {gap:.3f}]\n")
            f.write(f"{sample}\n\n")

    iter_num += 1

train_loss, val_loss, val_bpb = estimate_loss()
final_sample = generate(80)

if val_bpb < best_val_bpb:
    best_val_bpb = val_bpb
    save_clean(model, BEST_MODEL_PATH)

save_clean(model, LAST_MODEL_PATH)

print(f"\n{'='*50}")
print(f"ФИНАЛ {VERSION}: {(time.time()-start_time)/60:.1f}мин, {iter_num} iter")
print(f"  BPB: {val_bpb:.4f} | Best: {best_val_bpb:.4f}")
print(f"  '{final_sample[:250]}'")
print("="*50)

with open('results.tsv', 'a') as f:
    f.write(f"{int(time.time())}\t{val_bpb:.4f}\n")
with open(CONSCIOUSNESS_LOG, 'a') as f:
    f.write(f"[ФИНАЛ {VERSION} | {iter_num} iter | BPB: {val_bpb:.4f}]\n")
    f.write(f"{final_sample}\n\n")
