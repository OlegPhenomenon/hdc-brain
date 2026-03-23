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


# ============================================================================
# BPE-LIKE SUBWORD TOKENIZER
# ============================================================================

class SubwordTokenizer:
    """Субсловный токенизатор (BPE-подобный).

    Вместо целых слов использует частотные подстроки (2-6 символов).
    Преимущества перед word-level:
    - Обрабатывает опечатки (разбивает на известные куски)
    - Может генерировать новые слова
    - Меньший vocab при том же покрытии
    - Триграммы/биграммы несут морфологическую информацию
    """
    def __init__(self, text, max_vocab=4000):
        # Базовый vocab: все ASCII символы
        chars = sorted(set(text))
        self.char_vocab = {c: i for i, c in enumerate(chars)}

        # Считаем частоту n-грамм (2-6 символов)
        ngram_counts = {}
        for n in range(2, 7):
            for i in range(len(text) - n):
                ng = text[i:i+n]
                if '\n' not in ng[1:]:  # Не ломаем строки
                    ngram_counts[ng] = ngram_counts.get(ng, 0) + 1

        # Берём самые частые n-граммы
        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: -x[1])

        # Строим vocab: символы + частые n-граммы
        vocab = list(chars)
        for ng, count in sorted_ngrams:
            if len(vocab) >= max_vocab:
                break
            if count >= 10:  # Минимальная частота
                vocab.append(ng)

        self.stoi = {s: i for i, s in enumerate(vocab)}
        self.itos = {i: s for i, s in enumerate(vocab)}
        self.vocab_size = len(vocab)
        # Сортируем токены по длине (длинные сначала для greedy matching)
        self.tokens_by_length = sorted(
            [(s, i) for s, i in self.stoi.items()],
            key=lambda x: -len(x[0])
        )

    def encode(self, text):
        """Greedy longest-match tokenization"""
        result = []
        i = 0
        while i < len(text):
            matched = False
            # Пробуем самый длинный токен
            for max_len in range(min(6, len(text) - i), 0, -1):
                substr = text[i:i+max_len]
                if substr in self.stoi:
                    result.append(self.stoi[substr])
                    i += max_len
                    matched = True
                    break
            if not matched:
                result.append(0)  # unknown char
                i += 1
        return result

    def decode(self, ids):
        return ''.join(self.itos.get(i, '?') for i in ids)


# Сохраняем обратную совместимость
HDCTokenizer = SubwordTokenizer


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

    def write(self, memory, item, position, gate=None):
        """Записать item в позицию position (bind + bundle).

        Если gate=None: фиксированный decay 0.95 (legacy).
        Если gate задан: агент сам решает силу записи (Хофман: сознательная память).
        gate: (BN, 1) — насколько сильно текущий опыт затирает старую память.
        """
        pos_code = self.encode_position(position)
        bound = self.bind(item, pos_code)
        if gate is not None:
            # Gated: агент решает что запомнить (0=забыть всё, 1=запомнить всё)
            return gate * memory + (1 - gate) * bound
        # Fallback: фиксированный decay
        return 0.95 * memory + 0.05 * bound

    def read(self, memory, query):
        """Прочитать из памяти по запросу (cosine similarity)"""
        # Нормализованный dot product
        sim = F.cosine_similarity(memory, query, dim=-1)
        return sim

    def to_hdc(self, float_vector):
        """Конвертация float вектора в bipolar HDC.
        Straight-Through Estimator: forward = hard sign, backward = tanh gradient.
        Это позволяет градиентам течь через HDC операции.
        """
        soft = torch.tanh(float_vector * 3.0)
        hard = torch.sign(float_vector)
        hard = torch.where(hard == 0, torch.ones_like(hard), hard)  # избегаем 0
        # STE: forward использует hard sign, backward — soft gradient от tanh
        return (hard - soft).detach() + soft

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

        # Общие ядра (shared across agents) с Pre-LN (стабилизация градиентов)
        self.perception_kernel = nn.Sequential(
            nn.LayerNorm(state_dim * 2),
            nn.Linear(state_dim * 2, state_dim),
            nn.GELU(),
        )
        self.decision_kernel = nn.Sequential(
            nn.LayerNorm(state_dim * 2),
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
        # Врождённый характер (обучается через backprop)
        self.agent_personality = nn.Parameter(torch.randn(n_agents, state_dim) * 0.1)

        # Приобретённый характер (формируется из опыта, НЕ через backprop)
        # EMA решений агента — какие решения он обычно принимает
        # Это "менталитет" — устойчивый паттерн поведения
        self.register_buffer('agent_character', torch.zeros(n_agents, state_dim))

        # Индивидуальные HDC ключи
        self.agent_hdc_key = nn.Parameter(torch.sign(torch.randn(n_agents, hdc_dim)))

        # === GATED MEMORY: агент решает что запомнить (Хофман: сознательная память) ===
        # Вместо фиксированного decay 0.95: gate = σ(W @ [state, experience])
        # Важное воспоминание → gate≈1 (сохранить), шум → gate≈0 (перезаписать)
        self.memory_gate = nn.Linear(state_dim + hdc_dim, 1)

        # === ROLE DROPOUT: заставляет агентов специализироваться ===
        # По Хофману, мета-агенты формируются из специализированных.
        # Случайное отключение 15% агентов → каждый учится быть незаменимым.
        self.role_dropout = nn.Dropout(p=0.15)

        # === ЖУРНАЛ МЫСЛЕЙ (для интроспекции) ===
        # Не участвует в обучении — только для наблюдения
        self.thought_log = []

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

        # Индивидуальность = врождённый характер + приобретённый менталитет
        personality = self.agent_personality  # (N, D) — врождённый
        character = self.agent_character      # (N, D) — приобретённый
        full_personality = personality + 0.3 * character  # Менталитет влияет на 30%
        modulated_signal = incoming_signal * (1 + 0.1 * full_personality.unsqueeze(0))

        # 1. PERCEIVE — все агенты параллельно
        # Reshape: (B, N, D) → (B*N, D) для прохода через shared linear
        BN = B * N
        perception_input = torch.cat([
            modulated_signal.view(BN, D),
            incoming_actions.view(BN, D)
        ], dim=-1)  # (BN, 2D)
        perception = self.perception_kernel(perception_input)  # (BN, D)

        # 2. REMEMBER (HDC) — STE позволяет градиентам течь через to_hdc_proj
        # bind и cosine_similarity дифференцируемы → grad flows: loss → memory_strength → sim → query → to_hdc_proj
        hdc_query = hdc_engine.to_hdc(self.to_hdc_proj(perception))  # STE: grad через tanh
        agent_keys = self.agent_hdc_key.unsqueeze(0).expand(B, -1, -1).reshape(BN, -1)
        personal_query = hdc_engine.bind(hdc_query, agent_keys)  # element-wise mul — diff
        ep_flat = episodic_mem.view(BN, -1).detach()  # HDC content не backprop'ится
        cr_flat = crystallized_mem.view(BN, -1).detach()
        episodic_sim = hdc_engine.read(ep_flat, personal_query)  # cosine_sim — diff через query
        crystal_sim = hdc_engine.read(cr_flat, personal_query)
        memory_strength = torch.sigmoid(episodic_sim + crystal_sim).unsqueeze(-1)
        # from_hdc_proj + memory_strength → gradient path к loss
        memory_recall = self.from_hdc_proj(ep_flat + cr_flat) * memory_strength

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

        # === ROLE DROPOUT: случайное отключение агентов при обучении ===
        # Заставляет агентов специализироваться, не полагаться на "лидера".
        # При eval все агенты активны (стандартное поведение Dropout).
        if self.training:
            # Dropout по агентам (dim=1), не по features
            role_mask = torch.ones(B, N, 1, device=actions.device)
            role_mask = self.role_dropout(role_mask)  # ~15% агентов обнулены
            actions = actions.view(B, N, D) * role_mask
            actions = actions.view(BN, D)

        # === ФОРМИРОВАНИЕ ХАРАКТЕРА ===
        with torch.no_grad():
            decision_pattern = decision.view(B, N, D).mean(dim=0)
            self.agent_character.mul_(0.99).add_(0.01 * decision_pattern)

        # === ЖУРНАЛ МЫСЛЕЙ (только при eval) ===
        if not self.training and len(self.thought_log) < 200:
            with torch.no_grad():
                energies = new_states.view(B, N, D).norm(dim=-1).mean(dim=0)
                gate_mean = gate.view(B, N, D).mean(dim=(0, 2))
                self.thought_log.append({
                    'step': step,
                    'agent_energies': energies.cpu().numpy(),
                    'gate_openness': gate_mean.cpu().numpy(),
                    'decision_norm': decision_pattern.norm(dim=-1).cpu().numpy(),
                })

        # 5. STORE с GATED MEMORY (Хофман: сознательное запоминание)
        # Memory gate вычисляется С градиентом — агент учится ЧТО запоминать
        # HDC запись остаётся без gradient (bipolar операции)
        mem_gate_input = torch.cat([new_states, hdc_query], dim=-1)  # (BN, D + hdc_dim)
        mem_gate = torch.sigmoid(self.memory_gate(mem_gate_input))  # (BN, 1)

        with torch.no_grad():
            experience_hdc = hdc_engine.to_hdc(self.to_hdc_proj(new_states.detach()))
            personal_exp = hdc_engine.bind(experience_hdc, agent_keys.detach())
            new_episodics = hdc_engine.write(ep_flat, personal_exp, step, gate=mem_gate.detach())

        # Reshape обратно: (BN, ...) → (B, N, ...)
        return (
            actions.view(B, N, D),
            new_states.view(B, N, D),
            new_episodics.view(B, N, -1),
            experience_hdc.view(B, N, -1),
            mem_gate.view(B, N, 1),  # для memory regularization loss
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
        # Обучаемая температура: высокая → dense (все общаются), низкая → sparse
        self.log_temperature = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0

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

        # Attention-подобные scores с обучаемой температурой
        temperature = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.scale * temperature)

        N = action_states.shape[1]
        # Sparse top-k только для больших графов (N > 48).
        # При N <= 48 overhead от topk + scatter > выигрыш от sparsity.
        if self.top_k < N and N > 48:
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
        # Температура для soft threshold (обучаемая)
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def forward(self, agent_states, agent_actions):
        """
        Мягкая коалиция: агенты с похожими действиями обмениваются состояниями.
        Используем sigmoid вместо hard threshold чтобы градиенты текли через coalition_proj.
        """
        B, N, D = agent_states.shape

        # Compute similarity между действиями агентов
        proj = self.coalition_proj(agent_actions)  # (B, N, D//4)
        proj_norm = F.normalize(proj, dim=-1)
        sim = torch.matmul(proj_norm, proj_norm.transpose(-2, -1))  # (B, N, N)

        # Soft threshold: sigmoid((sim - threshold) * temperature)
        # При temperature=10: sim=0.7 → 0.5, sim=0.8 → 0.73, sim=0.6 → 0.27
        # Градиенты текут через sigmoid → proj_norm → coalition_proj
        coalition_weights = torch.sigmoid((sim - self.threshold) * self.temperature)
        # Нормализуем: каждый агент усредняет состояния своей коалиции
        coalition_weights = coalition_weights / (coalition_weights.sum(dim=-1, keepdim=True) + 1e-8)

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

        # Consensus → agent modulation: recalled consensus влияет на agent_states
        # Это замыкает gradient path: consensus_memory_q/k/v → recalled → agent_states → loss
        self.consensus_to_agents = nn.Linear(state_dim, state_dim)

        # === RESIDUAL SHORTCUT: прямой провод от текущего токена к logits ===
        # Без этого ВСЯ информация о текущем токене должна пройти через agent_states.
        # Трансформеры имеют residual stream — прямой провод вход→выход.
        # Это даёт модели "bigram baseline" бесплатно.
        self.residual_proj = nn.Linear(state_dim, state_dim)

        # === GRU-ГЕЙТ для сенсорной инъекции ===
        # Вместо тупого += дать агентам решать что принять, а что забыть.
        # gate = sigmoid(W_z @ [agent_state, sensory_input])
        # agent_state = gate * agent_state + (1-gate) * sensory_input
        self.sensory_gate = nn.Linear(state_dim * 2, state_dim)

        # Выход: проекция в state_dim → dot product с embedding (weight tying)
        self.dropout = nn.Dropout(dropout)
        self.pre_action = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Bias для выходного слоя
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def consensus_recall(self, query, memory_list):
        """Hopfield-подобный запрос к памяти консенсусов (legacy, для совместимости)"""
        if len(memory_list) == 0:
            return query
        mem = torch.stack(memory_list[-self.memory_slots:], dim=1)
        Q = self.consensus_memory_q(query).unsqueeze(1)
        K = self.consensus_memory_k(mem)
        V = self.consensus_memory_v(mem)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * 6.0 / math.sqrt(self.state_dim)
        attn = F.softmax(attn, dim=-1)
        recalled = torch.matmul(attn, V).squeeze(1)
        g = torch.sigmoid(self.consensus_memory_gate(torch.cat([query, recalled], dim=-1)))
        return self.consensus_ln(query + g * recalled)

    def consensus_recall_ring(self, query, ring_buf, count):
        """Hopfield-подобный запрос к ring buffer консенсусов.

        Вместо Python list + torch.stack на каждом шаге,
        используем pre-allocated тензор ring_buf: (max_chunks, B, D).
        """
        if count == 0:
            return query
        n = min(count, self.memory_slots)
        # .clone() обязателен: ring_buf — один тензор, запись в любой индекс
        # инкрементирует version counter всего тензора. Без clone backward ломается.
        if count <= self.memory_slots:
            mem = ring_buf[:count].clone().permute(1, 0, 2)  # (B, count, D)
        else:
            mem = ring_buf[count - n:count].clone().permute(1, 0, 2)
        Q = self.consensus_memory_q(query).unsqueeze(1)
        K = self.consensus_memory_k(mem)
        V = self.consensus_memory_v(mem)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * 6.0 / math.sqrt(self.state_dim)
        attn = F.softmax(attn, dim=-1)
        recalled = torch.matmul(attn, V).squeeze(1)
        g = torch.sigmoid(self.consensus_memory_gate(torch.cat([query, recalled], dim=-1)))
        return self.consensus_ln(query + g * recalled)

    def _batched_sensory_injection(self, chunk_percepts, agent_states, prev_logits, importance):
        """Батчированная сенсорная инъекция всего chunk-а.

        Вместо per-token loop обрабатываем весь chunk параллельно.
        Feedback loop: используем top-k argmax вместо full softmax @ embedding.

        Args:
            chunk_percepts: (B, chunk_len, D)
            agent_states: (B, N, D) — текущие состояния агентов
            prev_logits: (B, vocab) — logits с прошлого шага (для feedback первого токена)
            importance: (N,) — pre-computed softmax(agent_importance)

        Returns:
            agent_states: (B, N, D) — обновлённые состояния
            chunk_logits: (B, chunk_len, vocab) — logits для каждого токена chunk-а
            last_logits: (B, vocab) — последние logits (для feedback следующего chunk-а)
            accumulated_input: (B, N, D) — среднее сенсорное воздействие по chunk-у
        """
        B, CL, D = chunk_percepts.shape
        N = self.n_agents
        binding_gate = torch.sigmoid(self.binding_keys)  # (n_sensory, D) — вычисляем 1 раз

        accumulated_sensory = torch.zeros(B, self.n_sensory, D, device=chunk_percepts.device)
        chunk_logits_list = []

        for t in range(CL):
            current_percept = chunk_percepts[:, t, :]  # (B, D)

            # === Feedback через top-k argmax (вместо full softmax @ embedding) ===
            # Top-1 argmax + embedding lookup: O(B*D) вместо O(B*vocab*D)
            top_idx = prev_logits.argmax(dim=-1)  # (B,)
            feedback = self.word_embedding(top_idx)  # (B, D)

            fg = torch.sigmoid(self.feedback_gate(
                torch.cat([feedback, current_percept], dim=-1)
            ))
            modulated_percept = current_percept * (1 + fg)

            # Сенсорная инъекция с GRU-гейтом: агенты решают что принять
            sensory_input = self.sensory_proj(modulated_percept)  # (B, n_sensory * D)
            sensory_input = sensory_input.view(B, self.n_sensory, D) * binding_gate

            # GRU-гейт: gate = σ(W @ [current_state, new_input])
            # Позволяет забывать/перевзвешивать вместо тупого +=
            current_sensory = agent_states[:, :self.n_sensory, :]  # (B, n_sensory, D)
            gate_input = torch.cat([
                current_sensory.reshape(B * self.n_sensory, D),
                sensory_input.reshape(B * self.n_sensory, D)
            ], dim=-1)
            gate = torch.sigmoid(self.sensory_gate(gate_input))  # (B*n_sensory, D)
            gate = gate.view(B, self.n_sensory, D)
            new_sensory = gate * current_sensory + (1 - gate) * sensory_input
            agent_states = torch.cat([new_sensory, agent_states[:, self.n_sensory:, :]], dim=1)
            accumulated_sensory = accumulated_sensory + sensory_input.detach()

            # Logits = swarm consensus + residual shortcut от текущего токена
            agent_energy = agent_states.norm(dim=-1, keepdim=True)
            fitness = torch.sigmoid(agent_energy - agent_energy.mean(dim=1, keepdim=True))
            swarm_consensus = (agent_states * fitness * importance.view(1, -1, 1)).sum(dim=1)
            projected = self.pre_action(swarm_consensus)

            # Residual shortcut: прямой провод от текущего percept к logits
            # Даёт модели "bigram baseline" без прохождения через agent bottleneck
            residual = self.residual_proj(current_percept)  # (B, D)
            combined = projected + residual
            logits = F.linear(combined, self.word_embedding.weight, self.output_bias)
            chunk_logits_list.append(logits)
            prev_logits = logits.detach()

        # Среднее сенсорное воздействие за chunk — дополняем нулями для non-sensory
        avg_sensory = accumulated_sensory / max(CL, 1)
        accumulated_input = torch.zeros(B, N, D, device=chunk_percepts.device)
        accumulated_input[:, :self.n_sensory, :] = avg_sensory

        chunk_logits = torch.stack(chunk_logits_list, dim=1)  # (B, CL, vocab)
        return agent_states, chunk_logits, prev_logits, accumulated_input

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embedding → Interface (Хофман: восприятие как интерфейс)
        pos = torch.arange(T, device=idx.device)
        raw_percepts = self.word_embedding(idx) + self.pos_embedding(pos)
        percepts = self.perception_interface(raw_percepts)
        percepts = self.dropout(percepts)

        # Инициализация состояний агентов
        agent_states = torch.zeros(B, self.n_agents, self.state_dim, device=idx.device)
        agent_actions = torch.zeros(B, self.n_agents, self.state_dim, device=idx.device)

        # HDC память для каждого агента
        episodic_memories = torch.zeros(B, self.n_agents, self.hdc_dim, device=idx.device)
        crystallized_memories = torch.zeros(B, self.n_agents, self.hdc_dim, device=idx.device)

        vocab = self.word_embedding.weight.shape[0]
        prev_logits = torch.zeros(B, vocab, device=idx.device)
        importance = F.softmax(self.agent_importance, dim=0)

        # Ring buffer для consensus memory (вместо Python list + stack)
        max_chunks = (T + 7) // 8
        consensus_buf = torch.zeros(max_chunks, B, self.state_dim, device=idx.device)
        consensus_count = 0

        # === CHUNK PROCESSING: 8 токенов за раз ===
        CHUNK = min(8, T)
        n_chunks = (T + CHUNK - 1) // CHUNK
        all_logits = []

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * CHUNK
            chunk_end = min(chunk_start + CHUNK, T)
            chunk_percepts = percepts[:, chunk_start:chunk_end, :]

            # === СЕНСОРНАЯ ИНЪЕКЦИЯ (батчированная) ===
            agent_states, chunk_logits, prev_logits, accumulated_input = \
                self._batched_sensory_injection(chunk_percepts, agent_states, prev_logits, importance)
            all_logits.append(chunk_logits)

            # === ОБСУЖДЕНИЕ после chunk-а ===
            adjacency = self.dynamic_graph.compute_graph(agent_actions)
            incoming_actions = torch.matmul(adjacency, agent_actions)

            # Передаём accumulated_input (среднее по chunk-у) вместо последнего токена
            agent_actions, agent_states, episodic_memories, experiences, mem_gates = self.agents.forward_all(
                incoming_signal=accumulated_input,
                incoming_actions=incoming_actions,
                episodic_mem=episodic_memories,
                crystallized_mem=crystallized_memories,
                prev_states=agent_states,
                hdc_engine=self.hdc,
                step=chunk_idx,
            )

            # Crystallized memory: HDC bundling
            with torch.no_grad():
                BN_hdc = crystallized_memories.shape[0] * crystallized_memories.shape[1]
                cm_flat = crystallized_memories.view(BN_hdc, -1)
                exp_flat = experiences.view(BN_hdc, -1)
                cm_updated = self.hdc.write(cm_flat, exp_flat, chunk_idx)
                crystallized_memories = cm_updated.view_as(crystallized_memories)

            # Composition: коалиции
            agent_states = self.agent_combination(agent_states, agent_actions)
            agent_states = self.dropout(agent_states)

            # Consensus memory (ring buffer) — recalled consensus модулирует agent_states
            swarm_consensus = (agent_states * importance.view(1, -1, 1)).sum(dim=1)
            recalled_consensus = self.consensus_recall_ring(swarm_consensus, consensus_buf, consensus_count)
            consensus_buf[consensus_count] = recalled_consensus.detach()
            consensus_count += 1

            # Recalled consensus → agent modulation (замыкает gradient path)
            # Каждый агент получает "память группы" через broadcast
            consensus_signal = self.consensus_to_agents(recalled_consensus)  # (B, D)
            agent_states = agent_states + 0.1 * consensus_signal.unsqueeze(1)

        logits = torch.cat(all_logits, dim=1)  # (B, T, vocab)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            # === AUXILIARY LOSSES (Хофман: давление на специализацию) ===

            # 1. Sparsity penalty: заставляет граф быть разреженным
            # Агенты должны выбирать КОГО слушать, а не болтать со всеми
            # L1 norm adjacency → давление к 0 (т.е. к sparse графу)
            sparsity_loss = adjacency.mean()  # adjacency уже нормирована, mean ≈ 1/N

            # 2. Orthogonality loss: "иконки" восприятия должны быть различимы
            # По Хофману: interface создаёт максимально контрастные представления
            # Штрафуем если разные токены дают похожие percept'ы
            if T > 1:
                percept_norm = F.normalize(percepts, dim=-1)  # (B, T, D)
                gram = torch.matmul(percept_norm, percept_norm.transpose(-2, -1))  # (B, T, T)
                # Убираем диагональ (similarity с собой = 1, это нормально)
                eye = torch.eye(T, device=gram.device).unsqueeze(0)
                off_diag = gram * (1 - eye)
                ortho_loss = (off_diag ** 2).mean()  # Штраф за корреляцию
            else:
                ortho_loss = torch.tensor(0.0, device=logits.device)

            # 3. Memory gate regularization: давление на разнообразие gate values
            # Без этого gate может застрять на 0.95 для всех → = фиксированный decay
            # Entropy-like: хотим чтобы gate был иногда 0 (забыть) и иногда 1 (запомнить)
            mem_gate_mean = mem_gates.mean()
            mem_gate_var = mem_gates.var()
            # Штраф если variance слишком мала (все gates одинаковые)
            mem_gate_loss = -0.1 * mem_gate_var + 0.01 * (mem_gate_mean - 0.5) ** 2

            # Итоговый loss: CE + auxiliary Hoffman pressures
            loss = ce_loss + 0.01 * sparsity_loss + 0.1 * ortho_loss + mem_gate_loss

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
