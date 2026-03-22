"""
Hoffman Swarm v2.0 — True Conscious Agent Architecture

Полная реализация теории сознательных агентов Дональда Хофмана:
  C = (X, G, P, D, A, N)
  X = пространство восприятий
  G = пространство действий
  P = ядро восприятия (World → X)
  D = ядро решения (X → G)
  A = ядро действия (G → World)
  N = композиция агентов

Ключевые отличия от v1.x:
  1. Каждый агент имеет собственный цикл: Perceive → Remember → Decide → Act → Store
  2. HDC память: episodic (XOR bind, краткосрочная) + crystallized (долгосрочная)
  3. Осознанные каналы: action агента A = perception агента B
  4. Динамический граф: агенты решают с кем общаться
  5. Сознание фундаментальнее: связи вычисляются из состояний, не наоборот
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re


# ============================================================================
# HDC WORD-LEVEL TOKENIZER
# ============================================================================

class HDCTokenizer:
    """Word-level токенизатор для Hoffman Swarm."""
    def __init__(self, text, min_freq=2, max_vocab=8000):
        words = re.findall(r"[A-Za-z']+|[^A-Za-z'\s]|\n", text)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        vocab_words = ['<unk>', '<pad>', ' ']
        for w, c in sorted(word_counts.items(), key=lambda x: -x[1]):
            if c >= min_freq and len(vocab_words) < max_vocab:
                vocab_words.append(w)
        self.stoi = {w: i for i, w in enumerate(vocab_words)}
        self.itos = {i: w for i, w in enumerate(vocab_words)}
        self.vocab_size = len(vocab_words)
        self.unk_id = 0

    def encode(self, text):
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
        return ''.join(self.itos.get(i, '?') for i in ids)


# ============================================================================
# HDC MEMORY ENGINE
# ============================================================================

class HDCMemory(nn.Module):
    """Hyperdimensional Computing память для агентов.

    Использует bipolar вектора (+1/-1) и операции:
    - Bind (XOR для bipolar = element-wise multiply): связывание двух концептов
    - Bundle (majority vote / сложение + sign): суперпозиция нескольких паттернов
    - Permute (циклический сдвиг): кодирование последовательности

    Одни вектор хранит ТЫСЯЧИ паттернов через суперпозицию.
    Поиск — через cosine similarity (для bipolar = нормализованный dot product).
    """
    def __init__(self, hdc_dim, n_codebook=256):
        super().__init__()
        self.hdc_dim = hdc_dim
        # Кодовая книга: случайные bipolar вектора для кодирования позиций
        # Не обучаемые — фиксированная структура HDC
        codebook = torch.sign(torch.randn(n_codebook, hdc_dim))
        codebook[codebook == 0] = 1.0
        self.register_buffer('position_codes', codebook)

    def bind(self, a, b):
        """Связывание двух HDC векторов (аналог XOR для bipolar = element-wise multiply)"""
        return a * b

    def bundle(self, vectors, weights=None):
        """Суперпозиция: складываем и берём знак (majority vote)"""
        if weights is not None:
            stacked = torch.stack(vectors) * weights.unsqueeze(-1)
        else:
            stacked = torch.stack(vectors)
        summed = stacked.sum(dim=0)
        return torch.sign(summed + 1e-8)  # +epsilon чтобы избежать 0

    def permute(self, v, shift=1):
        """Циклический сдвиг — кодирует порядок в последовательности"""
        return torch.roll(v, shifts=shift, dims=-1)

    def encode_position(self, position):
        """Получить HDC код для данной позиции"""
        return self.position_codes[position % self.position_codes.shape[0]]

    def write(self, memory, item, position):
        """Записать item в позицию position (bind + bundle с decay)"""
        pos_code = self.encode_position(position)
        bound = self.bind(item, pos_code)
        # Bundle с exponential decay: старые воспоминания затухают
        # Без этого вектор растёт неограниченно
        return 0.95 * memory + 0.05 * bound

    def read(self, memory, query):
        """Прочитать из памяти по запросу (cosine similarity)"""
        # Нормализованный dot product
        sim = F.cosine_similarity(memory, query, dim=-1)
        return sim

    def to_hdc(self, float_vector):
        """Конвертация float вектора в bipolar HDC.
        Используем tanh (мягкий sign) чтобы сохранить gradient flow.
        При инференсе можно заменить на hard sign для скорости.
        """
        return torch.tanh(float_vector * 3.0)  # Мягкий sign: ≈±1 но дифференцируемый

    def from_hdc(self, hdc_vector):
        """HDC → float"""
        return hdc_vector


# ============================================================================
# CONSCIOUS AGENT — один агент Хофмана
# ============================================================================

class ConsciousAgent(nn.Module):
    """Один сознательный агент по теории Хофмана.

    Имеет собственный внутренний цикл:
      Perceive → Remember → Decide → Act → Store

    Каждый агент — самостоятельная единица с:
    - Собственным ядром решения (decision kernel)
    - Episodic memory (HDC, краткосрочная)
    - Crystallized memory (HDC, долгосрочная)
    - Perception / Decision / Action состояниями
    """
    def __init__(self, state_dim, hdc_dim, agent_id=0):
        super().__init__()
        self.state_dim = state_dim
        self.hdc_dim = hdc_dim
        self.agent_id = agent_id

        # === ЯДРО ВОСПРИЯТИЯ (P): входной сигнал → perception_state ===
        self.perception_kernel = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),  # input + incoming actions
            nn.GELU(),
        )

        # === ЯДРО РЕШЕНИЯ (D): perception + memory → decision_state ===
        # Это собственные "мозги" агента — маленькая сеть
        self.decision_kernel = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),  # perception + memory_recall
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )

        # === GRU-гейт для обновления состояния (стабилизация) ===
        self.update_gate = nn.Linear(state_dim * 2, state_dim)

        # === ЯДРО ДЕЙСТВИЯ (A): decision_state → action_state ===
        self.action_kernel = nn.Linear(state_dim, state_dim)

        # === HDC проекции: float ↔ HDC пространство ===
        self.to_hdc_proj = nn.Linear(state_dim, hdc_dim)
        self.from_hdc_proj = nn.Linear(hdc_dim, state_dim)

        self.layer_norm = nn.LayerNorm(state_dim)

    def forward(self, incoming_signal, incoming_actions, episodic_mem,
                crystallized_mem, prev_state, hdc_engine, step):
        """
        Один цикл сознательного агента.

        Args:
            incoming_signal: (B, state_dim) — сенсорный вход (если сенсорный агент)
            incoming_actions: (B, state_dim) — действия соседей
            episodic_mem: (B, hdc_dim) — HDC краткосрочная память
            crystallized_mem: (B, hdc_dim) — HDC долгосрочная память
            prev_state: (B, state_dim) — предыдущее состояние агента
            hdc_engine: HDCMemory — движок HDC операций
            step: int — текущий шаг (для позиционного кодирования)

        Returns:
            action_state: (B, state_dim) — действие агента (для соседей)
            new_state: (B, state_dim) — обновлённое состояние
            new_episodic: (B, hdc_dim) — обновлённая episodic memory
            experience: (B, hdc_dim) — текущий опыт для crystallized memory
        """
        B = incoming_signal.shape[0]

        # 1. PERCEIVE: входной сигнал + действия соседей → perception
        perception_input = torch.cat([incoming_signal, incoming_actions], dim=-1)
        perception = self.perception_kernel(perception_input)

        # 2. REMEMBER: запрос к HDC памяти
        # Конвертируем perception в HDC пространство для запроса
        hdc_query = hdc_engine.to_hdc(self.to_hdc_proj(perception))

        # Запрос к episodic memory
        episodic_sim = hdc_engine.read(episodic_mem, hdc_query)  # (B,) скаляр
        # Запрос к crystallized memory
        crystal_sim = hdc_engine.read(crystallized_mem, hdc_query)  # (B,)

        # Комбинируем: используем similarity как "уверенность в воспоминании"
        # и сам HDC вектор как контент воспоминания
        memory_strength = torch.sigmoid(episodic_sim + crystal_sim).unsqueeze(-1)  # (B, 1)
        memory_recall = self.from_hdc_proj(episodic_mem + crystallized_mem) * memory_strength

        # 3. DECIDE: perception + memory → decision
        decision_input = torch.cat([perception, memory_recall], dim=-1)
        decision = self.decision_kernel(decision_input)

        # GRU-подобное обновление: сколько нового решения принять
        gate = torch.sigmoid(self.update_gate(
            torch.cat([prev_state, decision], dim=-1)
        ))
        new_state = self.layer_norm(gate * decision + (1 - gate) * prev_state)

        # 4. ACT: decision → action (для передачи соседям)
        action_state = self.action_kernel(new_state)

        # 5. STORE: записать текущий опыт в episodic memory
        experience_hdc = hdc_engine.to_hdc(self.to_hdc_proj(new_state))
        new_episodic = hdc_engine.write(episodic_mem, experience_hdc, step)

        return action_state, new_state, new_episodic, experience_hdc


# ============================================================================
# SHARED KERNELS — оптимизация: агенты делят веса ядер
# ============================================================================

class SharedConsciousAgent(nn.Module):
    """Оптимизированная версия: N агентов делят веса ядер.

    Вместо N отдельных nn.Linear (что взорвёт параметры),
    агенты используют общие ядра + индивидуальные модуляторы.

    Это как "базовый мозг" + "личность" каждого агента.
    Общий мозг: ~90% вычислений
    Личность: ~10% — маленький вектор модулирующий поведение
    """
    def __init__(self, n_agents, state_dim, hdc_dim):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hdc_dim = hdc_dim

        # Общие ядра (shared across agents)
        self.perception_kernel = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.GELU(),
        )
        self.decision_kernel = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )
        self.update_gate = nn.Linear(state_dim * 2, state_dim)
        self.action_kernel = nn.Linear(state_dim, state_dim)

        # HDC проекции (shared)
        self.to_hdc_proj = nn.Linear(state_dim, hdc_dim)
        self.from_hdc_proj = nn.Linear(hdc_dim, state_dim)

        # === ИНДИВИДУАЛЬНОСТЬ каждого агента ===
        # Каждый агент имеет свой "характер" — вектор модулирующий ядра
        self.agent_personality = nn.Parameter(torch.randn(n_agents, state_dim) * 0.1)

        # Индивидуальные HDC кодовые книги (каждый агент "видит" мир по-своему)
        self.agent_hdc_key = nn.Parameter(torch.sign(torch.randn(n_agents, hdc_dim)))

        self.layer_norm = nn.LayerNorm(state_dim)

    def forward_all(self, incoming_signal, incoming_actions,
                    episodic_mem, crystallized_mem, prev_states, hdc_engine, step):
        """
        Цикл ВСЕХ агентов одновременно (батчированно).

        Args:
            incoming_signal: (B, N, D) — входной сигнал для каждого агента
            incoming_actions: (B, N, D) — действия соседей для каждого агента
            episodic_mem: (B, N, hdc_dim) — HDC память каждого агента
            crystallized_mem: (B, N, hdc_dim) — долгосрочная HDC память
            prev_states: (B, N, D) — предыдущие состояния
            hdc_engine: HDCMemory
            step: int
        Returns:
            actions, new_states, new_episodics, experiences — всё (B, N, ...)
        """
        B, N, D = incoming_signal.shape

        # Индивидуальность модулирует сигнал для каждого агента
        personality = self.agent_personality  # (N, D)
        modulated_signal = incoming_signal * (1 + 0.1 * personality.unsqueeze(0))

        # 1. PERCEIVE — все агенты параллельно
        # Reshape: (B, N, D) → (B*N, D) для прохода через shared linear
        BN = B * N
        perception_input = torch.cat([
            modulated_signal.view(BN, D),
            incoming_actions.view(BN, D)
        ], dim=-1)  # (BN, 2D)
        perception = self.perception_kernel(perception_input)  # (BN, D)

        # 2. REMEMBER (HDC) — батчированно
        hdc_query = hdc_engine.to_hdc(self.to_hdc_proj(perception))  # (BN, hdc_dim)
        # Персонализация: каждый агент имеет уникальный ключ
        agent_keys = self.agent_hdc_key.unsqueeze(0).expand(B, -1, -1).reshape(BN, -1)  # (BN, hdc_dim)
        personal_query = hdc_engine.bind(hdc_query, agent_keys)

        ep_flat = episodic_mem.view(BN, -1)
        cr_flat = crystallized_mem.view(BN, -1)
        episodic_sim = hdc_engine.read(ep_flat, personal_query)  # (BN,)
        crystal_sim = hdc_engine.read(cr_flat, personal_query)   # (BN,)
        memory_strength = torch.sigmoid(episodic_sim + crystal_sim).unsqueeze(-1)  # (BN, 1)
        memory_recall = self.from_hdc_proj(ep_flat + cr_flat) * memory_strength  # (BN, D)

        # 3. DECIDE — все агенты параллельно
        # По Хофману: decision kernel = марковское ядро P(action|perception)
        # Добавляем стохастичность при обучении (exploration)
        decision_input = torch.cat([perception, memory_recall], dim=-1)  # (BN, 2D)
        decision = self.decision_kernel(decision_input)  # (BN, D)
        # Модуляция личностью
        personality_flat = personality.unsqueeze(0).expand(B, -1, -1).reshape(BN, D)
        decision = decision * (1 + 0.1 * personality_flat)
        # Стохастическое ядро: добавляем шум при обучении (exploration/exploitation)
        if self.training:
            decision = decision + 0.02 * torch.randn_like(decision)

        # GRU update
        prev_flat = prev_states.view(BN, D)
        gate = torch.sigmoid(self.update_gate(
            torch.cat([prev_flat, decision], dim=-1)
        ))
        new_states = self.layer_norm(gate * decision + (1 - gate) * prev_flat)

        # 4. ACT
        actions = self.action_kernel(new_states)  # (BN, D)

        # 5. STORE
        experience_hdc = hdc_engine.to_hdc(self.to_hdc_proj(new_states))  # (BN, hdc_dim)
        personal_exp = hdc_engine.bind(experience_hdc, agent_keys)
        new_episodics = hdc_engine.write(ep_flat, personal_exp, step)  # (BN, hdc_dim)

        # Reshape обратно: (BN, ...) → (B, N, ...)
        return (
            actions.view(B, N, D),
            new_states.view(B, N, D),
            new_episodics.view(B, N, -1),
            experience_hdc.view(B, N, -1),
        )


# ============================================================================
# DYNAMIC GRAPH — агенты решают с кем общаться
# ============================================================================

class DynamicGraph(nn.Module):
    """Динамический граф связей.

    Вместо фиксированного agent_network (nn.Parameter),
    связи вычисляются из ACTION состояний агентов.

    Сознание → материя: топология сети = результат коллективного сознания.
    """
    def __init__(self, state_dim, top_k_neighbors=8):
        super().__init__()
        self.top_k = top_k_neighbors
        self.query_proj = nn.Linear(state_dim, state_dim // 4)
        self.key_proj = nn.Linear(state_dim, state_dim // 4)
        self.scale = math.sqrt(state_dim // 4)

    def compute_graph(self, action_states):
        """
        Вычисляет граф связей из действий агентов.

        Args:
            action_states: (B, N, state_dim) — действия всех агентов
        Returns:
            adjacency: (B, N, N) — soft adjacency matrix
        """
        Q = self.query_proj(action_states)  # (B, N, D//4)
        K = self.key_proj(action_states)    # (B, N, D//4)

        # Attention-подобные scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, N, N)

        # Sparse: каждый агент общается только с top-k соседями
        if self.top_k < action_states.shape[1]:
            topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
            scores = scores * mask + (1 - mask) * (-1e9)

        adjacency = F.softmax(scores, dim=-1)
        return adjacency


# ============================================================================
# AGENT COMBINATION (N) — порождение мета-агентов
# ============================================================================

class AgentCombination(nn.Module):
    """Simplified Combination: агенты с похожими действиями усиливают друг друга.

    По Хофману, два сознательных агента могут объединиться в
    мета-агента. Полная реализация (создание/удаление агентов)
    несовместима с фиксированным batch training.

    Компромисс: soft combination — агенты с высокой корреляцией
    формируют "коалиции", внутри которых состояния усредняются.
    Это мягкая версия composition: не удаляем агентов, но создаём
    эмерджентные группы.
    """
    def __init__(self, n_agents, state_dim, coalition_threshold=0.7):
        super().__init__()
        self.threshold = coalition_threshold
        self.coalition_proj = nn.Linear(state_dim, state_dim // 4)

    def forward(self, agent_states, agent_actions):
        """
        Мягкая коалиция: агенты с похожими действиями обмениваются состояниями.
        """
        B, N, D = agent_states.shape

        # Compute similarity между действиями агентов
        proj = self.coalition_proj(agent_actions)  # (B, N, D//4)
        proj_norm = F.normalize(proj, dim=-1)
        sim = torch.matmul(proj_norm, proj_norm.transpose(-2, -1))  # (B, N, N)

        # Мягкая коалиция: high similarity → обмен состояниями
        # Используем sim как mixing weights (только выше порога)
        coalition_mask = (sim > self.threshold).float()
        # Нормализуем: каждый агент усредняет состояния своей коалиции
        coalition_weights = coalition_mask / (coalition_mask.sum(dim=-1, keepdim=True) + 1e-8)

        # Обмен: состояние = среднее по коалиции
        combined_states = torch.matmul(coalition_weights, agent_states)

        # Мягкий blend: 80% своё, 20% коалиция
        return 0.8 * agent_states + 0.2 * combined_states


# ============================================================================
# HOFFMAN SWARM v2.0 — Полная модель
# ============================================================================

class HoffmanSwarmV2(nn.Module):
    """Полная реализация теории сознательных агентов Хофмана.

    Архитектура:
    1. HDC Word-Level Tokenization → вход
    2. Сенсорная проекция на subset агентов
    3. Каждый агент проходит свой цикл (Perceive→Remember→Decide→Act→Store)
    4. Динамический граф: agent.action → neighbor.perception
    5. Консенсус → Hopfield long-term memory → выход
    6. Петля Хофмана: выход → модуляция следующего восприятия

    HDC память:
    - episodic_memory: один HDC вектор на агента, обнуляется между текстами
    - crystallized_memory: один HDC вектор на агента, никогда не обнуляется
    """
    def __init__(self, vocab_size, n_agents, state_dim, hdc_dim,
                 n_sensory, memory_slots, seq_len, dropout=0.1):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hdc_dim = hdc_dim
        self.n_sensory = n_sensory
        self.memory_slots = memory_slots
        self.seq_len = seq_len

        # HDC Engine
        self.hdc = HDCMemory(hdc_dim)

        # Perception: слова → state_dim
        self.word_embedding = nn.Embedding(vocab_size, state_dim)
        self.pos_embedding = nn.Embedding(seq_len, state_dim)

        # === INTERFACE THEORY (Хофман): восприятие ≠ реальность ===
        # Интерфейсный слой сжимает и фильтрует "реальность" (embedding)
        # в полезное представление для агентов. Агенты видят не слово,
        # а его "иконку" — упрощённое но полезное для выживания представление.
        self.perception_interface = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
        )

        # Сенсорная проекция: вход → N_SENSORY агентов (каждый видит по-своему)
        self.sensory_proj = nn.Linear(state_dim, n_sensory * state_dim)
        # HDC binding keys для сенсорных агентов
        self.binding_keys = nn.Parameter(torch.randn(n_sensory, state_dim) * 0.02)

        # Петля Хофмана: feedback через weight tying (экономия 1.5M параметров!)
        # Вместо отдельного Linear(vocab_size, state_dim) используем embedding.weight.T
        # logits → softmax → взвешенная сумма embedding = "смысл" предыдущего действия
        self.feedback_gate = nn.Linear(state_dim * 2, state_dim)

        # === АГЕНТЫ с shared kernels ===
        self.agents = SharedConsciousAgent(n_agents, state_dim, hdc_dim)

        # === ДИНАМИЧЕСКИЙ ГРАФ ===
        self.dynamic_graph = DynamicGraph(state_dim, top_k_neighbors=min(8, n_agents))

        # === COMPOSITION (N): мягкое объединение агентов ===
        self.agent_combination = AgentCombination(n_agents, state_dim)

        # === ДОЛГОСРОЧНАЯ ПАМЯТЬ (Hopfield для консенсуса) ===
        self.consensus_memory_q = nn.Linear(state_dim, state_dim)
        self.consensus_memory_k = nn.Linear(state_dim, state_dim)
        self.consensus_memory_v = nn.Linear(state_dim, state_dim)
        self.consensus_memory_gate = nn.Linear(state_dim * 2, state_dim)
        self.consensus_ln = nn.LayerNorm(state_dim)

        # Консенсус: важность каждого агента (обучаемая)
        self.agent_importance = nn.Parameter(torch.zeros(n_agents))

        # Выход: проекция в state_dim → dot product с embedding (weight tying)
        # Экономит 3M параметров!
        self.dropout = nn.Dropout(dropout)
        self.pre_action = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Bias для выходного слоя
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def consensus_recall(self, query, memory_list):
        """Hopfield-подобный запрос к памяти консенсусов"""
        if len(memory_list) == 0:
            return query
        mem = torch.stack(memory_list[-self.memory_slots:], dim=1)  # (B, M, D)
        Q = self.consensus_memory_q(query).unsqueeze(1)
        K = self.consensus_memory_k(mem)
        V = self.consensus_memory_v(mem)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * 6.0 / math.sqrt(self.state_dim)
        attn = F.softmax(attn, dim=-1)
        recalled = torch.matmul(attn, V).squeeze(1)
        g = torch.sigmoid(self.consensus_memory_gate(torch.cat([query, recalled], dim=-1)))
        return self.consensus_ln(query + g * recalled)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embedding → Interface (Хофман: восприятие как интерфейс)
        pos = torch.arange(T, device=idx.device)
        raw_percepts = self.word_embedding(idx) + self.pos_embedding(pos)
        percepts = self.perception_interface(raw_percepts)  # "иконки" вместо "реальности"
        percepts = self.dropout(percepts)

        # Инициализация состояний агентов
        agent_states = torch.zeros(B, self.n_agents, self.state_dim, device=idx.device)
        agent_actions = torch.zeros(B, self.n_agents, self.state_dim, device=idx.device)

        # HDC память для каждого агента
        episodic_memories = torch.zeros(B, self.n_agents, self.hdc_dim, device=idx.device)
        # Crystallized memory — не обнуляется, но для batched training инициализируем
        crystallized_memories = torch.zeros(B, self.n_agents, self.hdc_dim, device=idx.device)

        logits_seq = []
        vocab = self.word_embedding.weight.shape[0]
        prev_logits = torch.zeros(B, vocab, device=idx.device)
        consensus_memory_list = []
        importance = F.softmax(self.agent_importance, dim=0)

        for t in range(T):
            current_percept = percepts[:, t, :]

            # === ПЕТЛЯ ХОФМАНА: feedback через weight tying ===
            # Softmax(logits) @ embedding.weight = "средний смысл" предыдущего предсказания
            feedback_probs = F.softmax(prev_logits, dim=-1)  # (B, vocab)
            feedback = feedback_probs @ self.word_embedding.weight  # (B, state_dim)
            fg = torch.sigmoid(self.feedback_gate(
                torch.cat([feedback, current_percept], dim=-1)
            ))
            modulated_percept = current_percept * (1 + fg)

            # === СЕНСОРНАЯ ИНЪЕКЦИЯ: только N_SENSORY агентов получают вход ===
            sensory_input = self.sensory_proj(modulated_percept)
            sensory_input = sensory_input.view(B, self.n_sensory, self.state_dim)
            sensory_input = sensory_input * torch.sigmoid(self.binding_keys)

            # Формируем входной сигнал для каждого агента
            agent_input = torch.zeros_like(agent_states)
            agent_input[:, :self.n_sensory, :] = sensory_input

            # === ДИНАМИЧЕСКИЙ ГРАФ: кто с кем общается ===
            # Оптимизация: пересчитываем граф каждые 4 шага (граф меняется медленно)
            if t % 4 == 0 or t == 0:
                adjacency = self.dynamic_graph.compute_graph(agent_actions)  # (B, N, N)
            # Входящие действия для каждого агента: взвешенная сумма действий соседей
            incoming_actions = torch.matmul(adjacency, agent_actions)  # (B, N, D)

            # === ВСЕ АГЕНТЫ ПАРАЛЛЕЛЬНО (батчированно) ===
            agent_actions, agent_states, episodic_memories, experiences = self.agents.forward_all(
                incoming_signal=agent_input,
                incoming_actions=incoming_actions,
                episodic_mem=episodic_memories,
                crystallized_mem=crystallized_memories,
                prev_states=agent_states,
                hdc_engine=self.hdc,
                step=t,
            )

            # Crystallized memory: медленное хеббовское обновление
            crystallized_memories = (
                0.995 * crystallized_memories.detach() +
                0.005 * experiences.detach()
            )

            # === COMPOSITION: мягкие коалиции (каждые 8 шагов) ===
            if t % 8 == 0 and t > 0:
                agent_states = self.agent_combination(agent_states, agent_actions)

            agent_states = self.dropout(agent_states)

            # === FITNESS-BASED SELECTION (Хофман: Fitness Beats Truth) ===
            # Агенты с высокой "энергией" (norm) вносят больше в консенсус
            # Это создаёт естественный отбор: полезные агенты усиливаются
            agent_energy = agent_states.norm(dim=-1, keepdim=True)  # (B, N, 1)
            fitness = torch.sigmoid(agent_energy - agent_energy.mean(dim=1, keepdim=True))
            agent_states_weighted = agent_states * fitness

            # === КОНСЕНСУС ===
            swarm_consensus = (agent_states_weighted * importance.view(1, -1, 1)).sum(dim=1)

            # Consensus memory recall
            swarm_consensus = self.consensus_recall(swarm_consensus, consensus_memory_list)
            consensus_memory_list.append(swarm_consensus.detach())

            # === ДЕЙСТВИЕ ===
            # Weight-tied output: project → dot product с embedding
            projected = self.pre_action(swarm_consensus)  # (B, state_dim)
            logits = F.linear(projected, self.word_embedding.weight, self.output_bias)  # (B, vocab)
            logits_seq.append(logits)
            prev_logits = logits.detach()

        logits = torch.stack(logits_seq, dim=1)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# ============================================================================
# ФАБРИКА МОДЕЛИ
# ============================================================================

def create_hoffman_swarm(vocab_size, config=None):
    """Создаёт модель с конфигурацией по умолчанию или пользовательской."""
    if config is None:
        config = {
            'n_agents': 32,
            'state_dim': 192,
            'hdc_dim': 512,       # Размерность HDC векторов (512 бит достаточно)
            'n_sensory': 8,
            'memory_slots': 64,
            'seq_len': 128,
            'dropout': 0.1,
        }

    model = HoffmanSwarmV2(
        vocab_size=vocab_size,
        n_agents=config['n_agents'],
        state_dim=config['state_dim'],
        hdc_dim=config['hdc_dim'],
        n_sensory=config['n_sensory'],
        memory_slots=config['memory_slots'],
        seq_len=config['seq_len'],
        dropout=config['dropout'],
    )
    return model, config


if __name__ == '__main__':
    # Тест: создать модель и прогнать один батч
    model, config = create_hoffman_swarm(vocab_size=8000)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Hoffman Swarm v2.0")
    print(f"  Agents: {config['n_agents']}")
    print(f"  State dim: {config['state_dim']}")
    print(f"  HDC dim: {config['hdc_dim']}")
    print(f"  Parameters: {n_params:,}")

    # Тестовый forward
    x = torch.randint(0, 8000, (2, 32))  # batch=2, seq=32
    y = torch.randint(0, 8000, (2, 32))  # targets
    logits, loss = model(x, y)
    print(f"  Forward OK: logits {logits.shape}, loss {loss.item():.4f}")
