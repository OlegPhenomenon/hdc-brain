import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

# --- ТЕ ЖЕ ПАРАМЕТРЫ, ЧТО В TRAIN.PY ---
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
N_AGENTS = 32
STATE_DIM = 256
N_SENSORY = 8
SEQ_LEN = 128
DROPOUT = 0.05

# --- КОПИРУЕМ ТОЛЬКО СТРУКТУРУ МОДЕЛИ (БЕЗ ОБУЧЕНИЯ) ---
class HofmanSwarm(nn.Module):
    def __init__(self, vocab_size, n_agents, state_dim):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.perception = nn.Embedding(vocab_size, state_dim)
        self.pos_embedding = nn.Embedding(SEQ_LEN, state_dim)
        self.sensory_proj = nn.Linear(state_dim, N_SENSORY * state_dim)
        self.agent_network = nn.Parameter(torch.zeros(n_agents, n_agents)) # Будет загружено из весов
        self.state_gate = nn.Linear(state_dim * 2, state_dim)
        self.state_candidate = nn.Sequential(nn.Linear(state_dim, state_dim), nn.GELU())
        self.layer_norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(DROPOUT)
        self.agent_importance = nn.Parameter(torch.zeros(n_agents))
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

        for t in range(T):
            current_percept = percepts[:, t, :]
            sensory_input = self.sensory_proj(current_percept).view(B, N_SENSORY, self.state_dim)
            sensory_full = torch.zeros_like(agents_states)
            sensory_full[:, :N_SENSORY, :] = sensory_input
            agents_states = agents_states + sensory_full
            interaction = torch.matmul(self.agent_network, agents_states)
            candidate = self.state_candidate(interaction)
            gate = torch.sigmoid(self.state_gate(torch.cat([agents_states, interaction], dim=-1)))
            agents_states = self.layer_norm(gate * candidate + (1 - gate) * agents_states)
            swarm_consensus = (agents_states * importance_weights.view(1, -1, 1)).sum(dim=1)
            logits_seq.append(self.action(swarm_consensus))
        return torch.stack(logits_seq, dim=1)

def chat():
    # Загрузка метаданных
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    itos, stoi, vocab_size = meta['itos'], meta['stoi'], meta['vocab_size']

    # Загрузка весов
    model = HofmanSwarm(vocab_size, N_AGENTS, STATE_DIM).to(DEVICE)
    if os.path.exists('best_swarm.pt'):
        model.load_state_dict(torch.load('best_swarm.pt', map_location=DEVICE, weights_only=True))
        print("\n--- Рой пробужден. На связи Шекспир v4 (BPB 2.24) ---")
    else:
        print("Файл best_swarm.pt не найден!")
        return

    model.eval()
    while True:
        prompt = input("\nДемиург: ")
        if prompt.lower() in ['exit', 'quit']: break
        
        context = torch.tensor([stoi.get(c, 0) for c in prompt], dtype=torch.long, device=DEVICE).unsqueeze(0)
        print("Рой:", end="", flush=True)
        
        with torch.no_grad():
            for _ in range(150): # Генерация 150 знаков
                if context.size(1) > SEQ_LEN: context = context[:, -SEQ_LEN:]
                logits = model(context)
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1) # Температура 0.8
                idx_next = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, idx_next), dim=1)
                print(itos[idx_next.item()], end="", flush=True)
        print()

if __name__ == "__main__":
    chat()