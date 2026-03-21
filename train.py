import time
import signal
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

stop_training = False

def handler(signum, frame):
    global stop_training
    print("\n[Таймер] Время вышло.")
    stop_training = True

if os.name != 'nt':
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(7200)  # 2 часа

# === v16 — Scaled Holographic Swarm (6.5M data) ===
# Больше данных → можно больше модель без переобучения
N_AGENTS = 64           # Больше агентов (было 32)
STATE_DIM = 288         # Больше ёмкость (было 192)
N_SENSORY = 16
N_CHANNELS = 2
N_INTERACTION_STEPS = 1
MEMORY_SLOTS = 128       # Больше памяти
MEMORY_HEADS = 4
MICRO_BATCH = 48
GRAD_ACCUM_STEPS = 3    # Эффективный батч = 128
BATCH_SIZE = MICRO_BATCH
SEQ_LEN = 128
LEARNING_RATE = 5e-4
WARMUP_STEPS = 600      # Длиннее warmup — больше модель
GRAD_CLIP = 1.0
DROPOUT = 0.12          # Меньше dropout — больше данных
WEIGHT_DECAY = 0.02
INPUT_NOISE = 0.03
STOCHASTIC_DEPTH = 0.05
EVAL_BATCHES = 30

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Устройство: {DEVICE}")

if not os.path.exists('meta.pkl'):
    print("Ошибка: python prepare_extended.py")
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


class HolographicMemory(nn.Module):
    def __init__(self, state_dim, n_slots, n_heads):
        super().__init__()
        self.n_slots = n_slots
        self.n_heads = n_heads
        self.head_dim = state_dim // n_heads
        self.query_proj = nn.Linear(state_dim, state_dim)
        self.key_proj = nn.Linear(state_dim, state_dim)
        self.value_proj = nn.Linear(state_dim, state_dim)
        self.out_proj = nn.Linear(state_dim, state_dim)
        self.memory_gate = nn.Linear(state_dim * 2, state_dim)
        self.layer_norm = nn.LayerNorm(state_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, memory_buffer, memory_count):
        B, D = query.shape
        n_valid = min(memory_count, self.n_slots)
        if n_valid == 0:
            return query
        mem = memory_buffer[:, :n_valid, :]
        Q = self.query_proj(query).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(mem).view(B, n_valid, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(mem).view(B, n_valid, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        recalled = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, D)
        recalled = self.out_proj(recalled)
        gate = torch.sigmoid(self.memory_gate(torch.cat([query, recalled], dim=-1)))
        return self.layer_norm(query + gate * recalled)


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


class HofmanSwarm(nn.Module):
    def __init__(self, vocab_size, n_agents, state_dim):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        self.perception = nn.Embedding(vocab_size, state_dim)
        self.pos_embedding = nn.Embedding(SEQ_LEN, state_dim)
        self.sensory_proj = nn.Linear(state_dim, N_SENSORY * state_dim)

        # Петля Хофмана
        self.feedback_proj = nn.Linear(vocab_size, state_dim)
        self.feedback_gate = nn.Linear(state_dim * 2, state_dim)

        # HDC binding
        self.binding_keys = nn.Parameter(torch.randn(N_SENSORY, state_dim) * 0.02)

        # Каналы
        self.channels = nn.ModuleList([
            SwarmChannel(n_agents, state_dim, connectivity=10, rewire_prob=0.05),
            SwarmChannel(n_agents, state_dim, connectivity=4, rewire_prob=0.25),
        ])
        self.channel_mix = nn.Parameter(torch.ones(N_CHANNELS) / N_CHANNELS)

        # Голографическая память
        self.holographic_memory = HolographicMemory(state_dim, MEMORY_SLOTS, MEMORY_HEADS)

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

        if self.training and INPUT_NOISE > 0:
            percepts = percepts + INPUT_NOISE * torch.randn_like(percepts)

        agents_states = torch.zeros(B, self.n_agents, self.state_dim, device=DEVICE)
        logits_seq = []
        importance_weights = F.softmax(self.agent_importance, dim=0)
        channel_weights = F.softmax(self.channel_mix, dim=0)

        prev_logits = torch.zeros(B, vocab_size, device=DEVICE)
        memory_list = []

        for t in range(T):
            current_percept = percepts[:, t, :]

            # Петля Хофмана
            feedback = self.feedback_proj(prev_logits)
            fg = torch.sigmoid(self.feedback_gate(
                torch.cat([feedback, current_percept], dim=-1)
            ))
            modulated_percept = current_percept * (1 + fg)

            # HDC binding
            sensory_input = self.sensory_proj(modulated_percept)
            sensory_input = sensory_input.view(B, N_SENSORY, self.state_dim)
            sensory_input = sensory_input * torch.sigmoid(self.binding_keys)

            sensory_full = torch.zeros_like(agents_states)
            sensory_full[:, :N_SENSORY, :] = sensory_input
            agents_states = agents_states + sensory_full

            # Взаимодействие (stochastic depth)
            for _ in range(N_INTERACTION_STEPS):
                channel_outputs = []
                for i, ch in enumerate(self.channels):
                    if self.training and torch.rand(1).item() < STOCHASTIC_DEPTH:
                        channel_outputs.append(agents_states)
                    else:
                        channel_outputs.append(ch(agents_states))
                agents_states = sum(
                    w * out for w, out in zip(channel_weights, channel_outputs)
                )

            agents_states = self.dropout(agents_states)

            # Консенсус
            swarm_consensus = (agents_states * importance_weights.view(1, -1, 1)).sum(dim=1)

            # Голографическая память
            if len(memory_list) > 0:
                mem_tensors = memory_list[-MEMORY_SLOTS:]
                memory_buffer = torch.stack(mem_tensors, dim=1)
                swarm_consensus = self.holographic_memory(swarm_consensus, memory_buffer, len(mem_tensors))
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


model = HofmanSwarm(vocab_size, N_AGENTS, STATE_DIM)
model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())

BEST_MODEL_PATH = 'best_swarm.pt'
LAST_MODEL_PATH = 'last_swarm.pt'

loaded = False
for cp in [BEST_MODEL_PATH, LAST_MODEL_PATH]:
    if os.path.exists(cp) and not loaded:
        try:
            model.load_state_dict(torch.load(cp, map_location=DEVICE, weights_only=True))
            print(f"Загружены: {cp}")
            loaded = True
        except Exception as e:
            print(f"Не подходит {cp}: {e}")
if not loaded:
    print("С нуля")

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

def generate(max_new_tokens=200):
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

start_time = time.time()
iter_num = 0
best_val_bpb = float('inf')

CONSCIOUSNESS_LOG = 'swarm_consciousness.log'
VERSION = 'v17-expanded'

print(f"\n{'='*50}")
print(f"{VERSION}: Scaled Holographic Swarm")
print(f"  {N_AGENTS} агентов, {STATE_DIM}d, {n_params:,} параметров")
print(f"  Голографическая память: {MEMORY_SLOTS} слотов")
print(f"  + Петля Хофмана + HDC binding + data augmentation")
print(f"  ДАННЫЕ: 6.5M символов (6x больше!), vocab={vocab_size}")
print(f"  Батч: {MICRO_BATCH}x{GRAD_ACCUM_STEPS}={MICRO_BATCH*GRAD_ACCUM_STEPS}")
print(f"{'='*50}\n")

with open(CONSCIOUSNESS_LOG, 'a') as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"{VERSION}: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  SCALED HOLOGRAPHIC SWARM — 6.5M DATA\n")
    f.write(f"  {N_AGENTS} agents, {STATE_DIM}d, {n_params:,} params\n")
    f.write(f"  Memory: {MEMORY_SLOTS} slots, vocab={vocab_size}\n")
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
        logits, loss = model(X, Y)
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        accum_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    if iter_num >= WARMUP_STEPS:
        scheduler.step()

    dt = time.time() - t0

    if iter_num % 50 == 0:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"iter {iter_num}: loss {accum_loss:.4f}, {dt*1000:.0f}ms, lr {lr_now:.2e}")

    if iter_num % 500 == 0 and iter_num > 0:
        train_loss, val_loss, val_bpb = estimate_loss()
        sample = generate(200)
        if val_bpb < best_val_bpb:
            best_val_bpb = val_bpb
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  >>> РЕКОРД! {BEST_MODEL_PATH}")
        elapsed = int(time.time() - start_time)
        gap = train_loss - val_loss
        print("-" * 50)
        print(f"[{elapsed//60}мин] Train: {train_loss:.4f} | Val: {val_loss:.4f} | BPB: {val_bpb:.4f} | Best: {best_val_bpb:.4f} | Gap: {gap:.3f}")
        print(f"  '{sample[:150]}'")
        print("-" * 50)

        with open(CONSCIOUSNESS_LOG, 'a') as f:
            f.write(f"[iter {iter_num} | {elapsed//60}min | BPB: {val_bpb:.4f} | gap: {gap:.3f}]\n")
            f.write(f"{sample}\n\n")

    iter_num += 1

train_loss, val_loss, val_bpb = estimate_loss()
final_sample = generate(300)

if val_bpb < best_val_bpb:
    best_val_bpb = val_bpb
    torch.save(model.state_dict(), BEST_MODEL_PATH)

torch.save(model.state_dict(), LAST_MODEL_PATH)

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
