import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import math

# --- Параметры v17 (должны совпадать с train.py) ---
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
N_AGENTS = 64
STATE_DIM = 288
N_SENSORY = 16
N_CHANNELS = 2
N_INTERACTION_STEPS = 1
MEMORY_SLOTS = 128
MEMORY_HEADS = 4
SEQ_LEN = 128
DROPOUT = 0.0  # Отключаем dropout при инференсе


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
        self.agent_network = nn.Parameter(torch.zeros(n_agents, n_agents))
        self.state_gate = nn.Linear(state_dim * 2, state_dim)
        self.state_candidate = nn.Sequential(nn.Linear(state_dim, state_dim), nn.GELU())
        self.layer_norm = nn.LayerNorm(state_dim)

    def forward(self, agents_states):
        interaction = torch.matmul(self.agent_network, agents_states)
        candidate = self.state_candidate(interaction)
        gate = torch.sigmoid(self.state_gate(torch.cat([agents_states, interaction], dim=-1)))
        return self.layer_norm(gate * candidate + (1 - gate) * agents_states)


class HofmanSwarm(nn.Module):
    def __init__(self, vocab_size, n_agents, state_dim):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.perception = nn.Embedding(vocab_size, state_dim)
        self.pos_embedding = nn.Embedding(SEQ_LEN, state_dim)
        self.sensory_proj = nn.Linear(state_dim, N_SENSORY * state_dim)
        self.feedback_proj = nn.Linear(vocab_size, state_dim)
        self.feedback_gate = nn.Linear(state_dim * 2, state_dim)
        self.binding_keys = nn.Parameter(torch.randn(N_SENSORY, state_dim) * 0.02)
        self.channels = nn.ModuleList([
            SwarmChannel(n_agents, state_dim, connectivity=10, rewire_prob=0.05),
            SwarmChannel(n_agents, state_dim, connectivity=4, rewire_prob=0.25),
        ])
        self.channel_mix = nn.Parameter(torch.ones(N_CHANNELS) / N_CHANNELS)
        self.dropout = nn.Dropout(DROPOUT)
        self.agent_importance = nn.Parameter(torch.zeros(n_agents))
        self.holographic_memory = HolographicMemory(state_dim, MEMORY_SLOTS, MEMORY_HEADS)
        self.action = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(state_dim * 2, vocab_size),
        )

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=DEVICE)
        percepts = self.perception(idx) + self.pos_embedding(pos)
        agents_states = torch.zeros(B, self.n_agents, self.state_dim, device=DEVICE)
        logits_seq = []
        importance_weights = F.softmax(self.agent_importance, dim=0)
        channel_weights = F.softmax(self.channel_mix, dim=0)
        prev_logits = torch.zeros(B, self.action[-1].out_features, device=DEVICE)
        memory_list = []

        for t in range(T):
            current_percept = percepts[:, t, :]
            feedback = self.feedback_proj(prev_logits)
            fg = torch.sigmoid(self.feedback_gate(torch.cat([feedback, current_percept], dim=-1)))
            modulated_percept = current_percept * (1 + fg)

            sensory_input = self.sensory_proj(modulated_percept).view(B, N_SENSORY, self.state_dim)
            sensory_input = sensory_input * torch.sigmoid(self.binding_keys)
            sensory_full = torch.zeros_like(agents_states)
            sensory_full[:, :N_SENSORY, :] = sensory_input
            agents_states = agents_states + sensory_full

            for _ in range(N_INTERACTION_STEPS):
                channel_outputs = [ch(agents_states) for ch in self.channels]
                agents_states = sum(w * out for w, out in zip(channel_weights, channel_outputs))

            swarm_consensus = (agents_states * importance_weights.view(1, -1, 1)).sum(dim=1)

            if len(memory_list) > 0:
                mem_tensors = memory_list[-MEMORY_SLOTS:]
                memory_buffer = torch.stack(mem_tensors, dim=1)
                swarm_consensus = self.holographic_memory(swarm_consensus, memory_buffer, len(mem_tensors))
            memory_list.append(swarm_consensus.detach())

            logits = self.action(swarm_consensus)
            logits_seq.append(logits)
            prev_logits = logits.detach()

        return torch.stack(logits_seq, dim=1)


def chat():
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    itos, stoi, vocab_size = meta['itos'], meta['stoi'], meta['vocab_size']

    model = HofmanSwarm(vocab_size, N_AGENTS, STATE_DIM).to(DEVICE)

    loaded = False
    for path in ['best_swarm.pt', 'last_swarm.pt']:
        if os.path.exists(path) and not loaded:
            try:
                model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
                print(f"\n--- Рой пробужден из {path} ---")
                loaded = True
            except Exception as e:
                print(f"Не удалось загрузить {path}: {e}")
    if not loaded:
        print("Веса не найдены!")
        return

    model.eval()
    print("Архитектура: v12 — Compact Holographic Swarm")
    print(f"  {N_AGENTS} агентов, {STATE_DIM}d, голографическая память ({MEMORY_SLOTS} слотов)")
    print(f"  + Петля Хофмана + HDC binding")
    print("Введите текст для продолжения роем. 'exit' для выхода.\n")

    while True:
        prompt = input("Демиург: ")
        if prompt.lower() in ['exit', 'quit']:
            break

        context = torch.tensor(
            [stoi.get(c, 0) for c in prompt], dtype=torch.long, device=DEVICE
        ).unsqueeze(0)
        print("Рой: ", end="", flush=True)

        with torch.no_grad():
            for _ in range(300):
                if context.size(1) > SEQ_LEN:
                    context = context[:, -SEQ_LEN:]
                logits = model(context)
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, idx_next), dim=1)
                print(itos[idx_next.item()], end="", flush=True)
        print("\n")

if __name__ == "__main__":
    chat()
