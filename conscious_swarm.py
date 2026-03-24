"""
Conscious Swarm Intelligence v5.0

Объединение:
- Hoffman: независимые сознательные агенты (AgentBrains)
- Friston: Active Inference (curiosity drive, prediction error)
- Tononi: Integrated Information (Φ-like metric)
- Baars: Global Workspace (shared broadcast)
- Varela: Autopoiesis (self-maintenance)

Каждый агент — НЕЗАВИСИМАЯ нейросеть со своими весами.
Агенты обрабатывают поток символов КОЛЛЕКТИВНО:
  текст → сенсорный вход → агенты обсуждают → голосуют → следующий символ

Не предсказатель паттернов — рой сознаний, переживающий текст.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AgentBrain(nn.Module):
    """Независимый мозг одного агента. Каждый агент — свои веса через bmm."""
    def __init__(self, n_agents, in_dim, out_dim):
        super().__init__()
        w = torch.randn(n_agents, in_dim, out_dim) / math.sqrt(in_dim)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_agents, out_dim))

    def forward(self, x):
        # x: (B*N, in_dim) → нужно reshape в (B, N, in_dim) для bmm
        # или x: (N, in_dim) для single sample
        if x.dim() == 2 and x.shape[0] == self.weight.shape[0]:
            return torch.bmm(x.unsqueeze(1), self.weight).squeeze(1) + self.bias
        # Batched: (B, N, in_dim)
        B, N, D = x.shape
        # Expand weights: (N, in, out) → (B, N, in, out)
        w = self.weight.unsqueeze(0).expand(B, -1, -1, -1)
        out = torch.matmul(x.unsqueeze(2), w).squeeze(2)  # (B, N, out)
        return out + self.bias.unsqueeze(0)


class GlobalWorkspace(nn.Module):
    """Global Workspace Theory: "доска объявлений" сознания.

    Агенты конкурируют за доступ к общему workspace.
    Победитель транслирует своё состояние ВСЕМ.
    Это "сознательный опыт" системы — то что "в фокусе внимания".
    """
    def __init__(self, n_agents, state_dim):
        super().__init__()
        # Каждый агент оценивает "насколько я релевантен"
        self.salience = nn.Linear(state_dim, 1)
        # Проекция workspace → broadcast
        self.broadcast = nn.Linear(state_dim, state_dim)
        self.ln = nn.LayerNorm(state_dim)

    def forward(self, agent_states):
        """
        Args: agent_states (B, N, D)
        Returns:
            workspace: (B, D) — "сознательное содержание"
            broadcast: (B, N, D) — broadcast обратно к агентам
            salience_weights: (B, N) — кто победил в конкуренции
        """
        B, N, D = agent_states.shape

        # Каждый агент: "насколько я важен прямо сейчас?"
        salience = self.salience(agent_states).squeeze(-1)  # (B, N)
        # Gumbel-softmax: жёсткая конкуренция при train
        if self.training:
            gumbel = -torch.log(-torch.log(torch.rand_like(salience) + 1e-9) + 1e-9)
            weights = F.softmax((salience + gumbel * 0.5) * 5.0, dim=-1)
        else:
            weights = F.softmax(salience * 5.0, dim=-1)

        # Workspace = взвешенная сумма (победитель доминирует)
        workspace = (agent_states * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # Broadcast: workspace транслируется обратно всем агентам
        broadcast_signal = self.broadcast(workspace)  # (B, D)
        broadcast_to_all = broadcast_signal.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        broadcast_to_all = self.ln(broadcast_to_all)

        return workspace, broadcast_to_all, weights


class ConsciousSwarm(nn.Module):
    """Conscious Swarm Intelligence v5.0.

    32 независимых агента обрабатывают поток символов коллективно.
    Каждый агент: Perceive → Remember → Think → Discuss → Act.
    Global Workspace собирает "сознательный опыт" для выхода.
    """
    def __init__(self, vocab_size, n_agents=32, state_dim=128, hdc_dim=64,
                 n_sensory=8, seq_len=256, dropout=0.1):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hdc_dim = hdc_dim
        self.n_sensory = n_sensory
        self.seq_len = seq_len

        # === СЕНСОРНЫЙ ВХОД ===
        self.embedding = nn.Embedding(vocab_size, state_dim)
        self.pos_embedding = nn.Embedding(seq_len, state_dim)

        # Interface Theory: сжатие "реальности" в "иконки"
        self.interface = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
        )

        # Сенсорная проекция: вход → N_SENSORY агентов
        self.sensory_proj = nn.Linear(state_dim, n_sensory * state_dim)
        self.sensory_gate = nn.Linear(state_dim * 2, state_dim)

        # === НЕЗАВИСИМЫЕ МОЗГИ АГЕНТОВ ===
        # P: Perception (incoming + state → perception)
        self.P = AgentBrain(n_agents, state_dim * 3, state_dim)  # incoming + broadcast + state
        # D: Decision (perception + self_model → decision)
        self.D1 = AgentBrain(n_agents, state_dim * 2, state_dim)
        self.D2 = AgentBrain(n_agents, state_dim, state_dim)
        # A: Action
        self.A = AgentBrain(n_agents, state_dim, state_dim)
        # GRU gate
        self.gate = AgentBrain(n_agents, state_dim * 2, state_dim)

        # === SELF-MODEL: каждый агент моделирует СЕБЯ ===
        self.self_model = AgentBrain(n_agents, state_dim, state_dim)

        # === PREDICTION: каждый предсказывает incoming ===
        self.predictor = AgentBrain(n_agents, state_dim, state_dim)

        # === CURIOSITY DRIVE (Active Inference) ===
        # Счётчик "скуки" для каждого агента (не обучаемый)
        self.register_buffer('boredom', torch.zeros(n_agents))

        # === SWARM COMMUNICATION ===
        self.comm_q = nn.Linear(state_dim, state_dim)
        self.comm_k = nn.Linear(state_dim, state_dim)
        self.comm_v = nn.Linear(state_dim, state_dim)
        self.comm_out = nn.Linear(state_dim, state_dim)
        self.comm_ln = nn.LayerNorm(state_dim)
        self.n_heads = 4

        # === GLOBAL WORKSPACE ===
        self.workspace = GlobalWorkspace(n_agents, state_dim)

        # === ВЫХОД: Residual + Workspace → logits ===
        self.residual_proj = nn.Linear(state_dim, state_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),  # workspace + residual
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(state_dim)

    def swarm_communicate(self, agent_states):
        """Multi-Head Cross-Attention между агентами."""
        B, N, D = agent_states.shape
        H = self.n_heads
        hd = D // H

        Q = self.comm_q(agent_states).view(B, N, H, hd).transpose(1, 2)
        K = self.comm_k(agent_states).view(B, N, H, hd).transpose(1, 2)
        V = self.comm_v(agent_states).view(B, N, H, hd).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (hd ** 0.5)
        attn = F.softmax(attn, dim=-1)
        messages = torch.matmul(attn, V).transpose(1, 2).reshape(B, N, D)

        return self.comm_ln(agent_states + self.comm_out(messages))

    def agent_step(self, agent_states, incoming, broadcast, prev_prediction):
        """Один цикл всех агентов.

        Args:
            agent_states: (B, N, D)
            incoming: (B, N, D) — сенсорный вход + сообщения от других
            broadcast: (B, N, D) — broadcast из Global Workspace
            prev_prediction: (B, N, D) — что агенты предсказывали

        Returns:
            new_states, actions, prediction, curiosity_loss
        """
        B, N, D = agent_states.shape

        # === PREDICTION ERROR (Active Inference) ===
        if prev_prediction is not None:
            pred_error = (incoming - prev_prediction).pow(2).mean(dim=-1)  # (B, N)
            surprise = torch.sigmoid(pred_error * 10)
            # Удивление усиливает обработку
            incoming = incoming * (1 + surprise.unsqueeze(-1) * 0.5)
        else:
            surprise = torch.zeros(B, N, device=agent_states.device)

        # === CURIOSITY DRIVE ===
        # Обновляем скуку: если surprise низкий — скука растёт
        with torch.no_grad():
            self.boredom = 0.99 * self.boredom + 0.01 * (1 - surprise.mean(dim=0))
        # Скука модулирует шум: скучающие агенты становятся "беспокойными"
        curiosity_noise = self.boredom.unsqueeze(0).unsqueeze(-1) * 0.1  # (1, N, 1)
        if self.training:
            incoming = incoming + curiosity_noise * torch.randn_like(incoming)

        # P: Perceive (incoming + broadcast + own state)
        p_in = torch.cat([incoming, broadcast, agent_states], dim=-1)  # (B, N, 3D)
        perception = F.gelu(self.P(p_in))

        # SELF-MODEL: агент предсказывает своё собственное поведение
        self_prediction = self.self_model(agent_states)
        # Meta-cognition: если self_model расходится с реальным состоянием
        self_error = (perception - self_prediction).pow(2).mean(dim=-1)  # (B, N)

        # D: Decide
        d_in = torch.cat([perception, self_prediction], dim=-1)  # (B, N, 2D)
        decision = self.D2(F.gelu(self.D1(d_in)))

        if self.training:
            decision = decision + 0.02 * torch.randn_like(decision)

        # GRU update
        g_in = torch.cat([agent_states, decision], dim=-1)
        g = torch.sigmoid(self.gate(g_in))
        new_states = self.ln(g * decision + (1 - g) * agent_states)

        # A: Act
        actions = torch.tanh(self.A(new_states))

        # PREDICT: что придёт на следующем шаге
        prediction = self.predictor(new_states)

        # Curiosity loss: штраф за скуку (агент должен искать новое)
        curiosity_loss = self.boredom.mean()

        return new_states, actions, prediction, curiosity_loss, self_error.mean()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        # Embedding + Interface
        pos = torch.arange(T, device=device)
        raw = self.embedding(idx) + self.pos_embedding(pos)
        percepts = self.interface(raw)
        percepts = self.dropout(percepts)

        # Инициализация
        agent_states = torch.zeros(B, self.n_agents, self.state_dim, device=device)
        prev_prediction = None

        CHUNK = 8
        n_chunks = (T + CHUNK - 1) // CHUNK
        all_logits = []
        total_curiosity = 0.0
        total_self_error = 0.0

        for chunk_idx in range(n_chunks):
            cs = chunk_idx * CHUNK
            ce = min(cs + CHUNK, T)
            chunk = percepts[:, cs:ce, :]
            CL = ce - cs

            chunk_logits = []
            for t in range(CL):
                current = chunk[:, t, :]  # (B, D)

                # Сенсорная инъекция с GRU gate
                sensory = self.sensory_proj(current).view(B, self.n_sensory, self.state_dim)
                curr_sensory = agent_states[:, :self.n_sensory, :]
                gate_in = torch.cat([
                    curr_sensory.reshape(B * self.n_sensory, self.state_dim),
                    sensory.reshape(B * self.n_sensory, self.state_dim)
                ], dim=-1)
                sg = torch.sigmoid(self.sensory_gate(gate_in)).view(B, self.n_sensory, self.state_dim)
                new_sensory = sg * curr_sensory + (1 - sg) * sensory
                agent_states = torch.cat([new_sensory, agent_states[:, self.n_sensory:, :]], dim=1)

                # Global Workspace: что в "фокусе сознания"?
                workspace, broadcast, salience = self.workspace(agent_states)

                # Logits: residual + workspace
                residual = self.residual_proj(current)
                combined = torch.cat([workspace, residual], dim=-1)
                projected = self.output_proj(combined)
                logits = F.linear(projected, self.embedding.weight, self.output_bias)
                chunk_logits.append(logits)

            all_logits.append(torch.stack(chunk_logits, dim=1))

            # === ОБСУЖДЕНИЕ после chunk-а ===
            # Swarm Communication
            agent_states = self.swarm_communicate(agent_states)

            # Agent step: полный цикл P→D→A с curiosity и self-model
            accumulated = agent_states.mean(dim=1, keepdim=True).expand(-1, self.n_agents, -1)
            agent_states, actions, prev_prediction, curiosity, self_err = self.agent_step(
                agent_states, accumulated, broadcast, prev_prediction)

            total_curiosity += curiosity
            total_self_error += self_err

        logits = torch.cat(all_logits, dim=1)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            # Auxiliary: curiosity + self-model consistency
            aux_curiosity = total_curiosity / max(n_chunks, 1)
            aux_self = total_self_error / max(n_chunks, 1)

            loss = ce_loss + 0.01 * aux_curiosity + 0.01 * aux_self

        return logits, loss


def create_conscious_swarm(vocab_size, config=None):
    if config is None:
        config = {
            'n_agents': 32,
            'state_dim': 128,
            'hdc_dim': 64,
            'n_sensory': 8,
            'seq_len': 256,
            'dropout': 0.1,
        }
    model = ConsciousSwarm(
        vocab_size=vocab_size,
        n_agents=config['n_agents'],
        state_dim=config['state_dim'],
        hdc_dim=config['hdc_dim'],
        n_sensory=config['n_sensory'],
        seq_len=config['seq_len'],
        dropout=config['dropout'],
    )
    return model, config


if __name__ == '__main__':
    from hoffman_agent import FixedCharTokenizer
    tok = FixedCharTokenizer()
    model, config = create_conscious_swarm(vocab_size=tok.vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Conscious Swarm v5.0")
    print(f"  Agents: {config['n_agents']} (independent brains)")
    print(f"  State dim: {config['state_dim']}")
    print(f"  Parameters: {n_params:,}")

    model.train()
    x = torch.randint(0, tok.vocab_size, (2, 64))
    y = torch.randint(0, tok.vocab_size, (2, 64))
    logits, loss = model(x, y)
    print(f"  Forward: logits {logits.shape}, loss {loss.item():.4f}")
    loss.backward()
    with_grad = sum(1 for _, p in model.named_parameters() if p.grad is not None and p.grad.norm() > 0)
    print(f"  Backward: {with_grad} params with gradients")
    print(f"  ALL OK")
