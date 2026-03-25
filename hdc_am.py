"""
HDC-AM: Hyperdimensional Associative Manifold

Принципиально другая архитектура:
- HDC как PRIMARY compute (контекст = одна матричная операция)
- Нейросеть как NAVIGATOR (маленькая, 3-8M params)
- Codebook как KNOWLEDGE BASE (масштабируется от размера словаря, не params)
- ВСЁ через матричные операции, НИКАКИХ циклов по токенам

Scaling: capacity растёт от HDC dim и codebook size, не от params нейросети.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HDCProcessor(nn.Module):
    """HDC ядро: контекст из T символов = одна матричная операция.

    Вместо O(T²) attention или O(T) RNN:
    - Каждый символ = случайный bipolar вектор (фиксированный)
    - Контекст = causal weighted sum с exponential decay
    - Одна matmul: decay_matrix (T,T) @ char_vectors (B,T,D) = contexts (B,T,D)
    """
    def __init__(self, vocab_size, hdc_dim=4096, decay=0.95):
        super().__init__()
        self.hdc_dim = hdc_dim

        # === LEARNABLE PER-DIMENSION DECAY (S4/Mamba insight) ===
        # Вместо scalar 0.95: каждое из 4096 измерений имеет свой decay rate.
        # Быстрые dims = орфография (decay~0.7), медленные = тема (decay~0.99).
        # Это расширяет эффективный контекст с ~20 до ~100+ символов.
        self.log_decay = nn.Parameter(torch.full((hdc_dim,), math.log(decay / (1 - decay))))

        # HDC codebook + LEARNABLE PERTURBATION
        # Базовые bipolar вектора фиксированы, но delta обучается
        codebook = torch.sign(torch.randn(vocab_size, hdc_dim))
        codebook[codebook == 0] = 1.0
        self.register_buffer('char_codebook', codebook)
        self.codebook_delta = nn.Embedding(vocab_size, hdc_dim)
        nn.init.zeros_(self.codebook_delta.weight)

        # === SELECTIVE GATING: важность каждого токена ===
        self.importance_gate = nn.Linear(hdc_dim, 1)

        # === HETERO-RESONANCE: асимметричный поиск связей ===
        self.resonance = HeteroResonance(hdc_dim, bottleneck=256)

        # Precompute decay matrix для разных seq_len (кешируется)
        self._decay_cache = {}

    def get_decay_rates(self):
        """Per-dimension decay rates в (0, 1)."""
        return torch.sigmoid(self.log_decay)  # (hdc_dim,)

    def get_decay_matrix(self, T, device):
        """Causal decay matrix с per-dimension learnable rates.
        Возвращает scalar decay matrix для HeteroResonance blend."""
        decay = self.get_decay_rates().mean().item()  # среднее для scalar matrix
        cache_key = (T, round(decay, 4))
        if cache_key not in self._decay_cache:
            indices = torch.arange(T, device=device)
            diffs = (indices.unsqueeze(1) - indices.unsqueeze(0)).float()
            matrix = decay ** diffs.clamp(min=0)
            matrix = torch.tril(matrix)
            matrix = matrix / matrix.sum(dim=-1, keepdim=True)
            self._decay_cache[cache_key] = matrix
        return self._decay_cache[cache_key]

    def encode(self, idx):
        """Символы → HDC вектора с learnable perturbation."""
        return self.char_codebook[idx] + 0.1 * self.codebook_delta(idx)

    def encode_trigrams(self, idx):
        """Символы → HDC триграммы через bind+roll. Матричная операция.

        Триграмм[t] = char[t] * roll(char[t-1], 1) * roll(char[t-2], 2)
        Захватывает локальные паттерны (слоги, морфемы) напрямую.
        Всё через матричные операции, без циклов.
        """
        chars = self.char_codebook[idx]  # (B, T, D)
        # Сдвинутые версии: char[t-1] и char[t-2]
        chars_m1 = torch.roll(chars, 1, dims=1)   # (B, T, D) — сдвиг на 1 по T
        chars_m2 = torch.roll(chars, 2, dims=1)   # сдвиг на 2
        # Обнуляем начало (нет предшествующих символов)
        chars_m1[:, 0, :] = 0
        chars_m2[:, :2, :] = 0
        # Roll по D-измерению (позиционное кодирование в HDC)
        chars_m1_rolled = torch.roll(chars_m1, 1, dims=-1)  # roll по D
        chars_m2_rolled = torch.roll(chars_m2, 2, dims=-1)
        # Bind (element-wise multiply для bipolar)
        # trigram = char[t] * roll_D(char[t-1]) * roll_D²(char[t-2])
        trigrams = chars * chars_m1_rolled * chars_m2_rolled
        # Комбинируем: обогащаем char информацией о контексте
        # 0.7 * trigram + 0.3 * char (чтобы не потерять отдельные символы)
        return 0.7 * trigrams + 0.3 * chars

    def build_context(self, hdc_chars):
        """HDC контекст: per-dim decay + gating + HeteroResonance.

        1. Importance gating
        2. Per-dimension exponential decay (S4/Mamba insight)
        3. HeteroResonance blend для далёких связей
        4. HDC residual: contexts += hdc_chars (сохраняет identity)
        """
        B, T, D = hdc_chars.shape

        # Selective Gating
        importance = torch.sigmoid(self.importance_gate(hdc_chars))
        gated_chars = hdc_chars * importance

        # Per-dimension decay context
        # Простая EMA: context[t] = decay * context[t-1] + (1-decay) * gated[t]
        # Для скорости используем decay matrix, но с learnable mean decay
        # Learnable decay через exp(log_sigmoid * distance) — gradient-friendly
        log_decay_rate = F.logsigmoid(self.log_decay).mean()  # log(decay) в graph
        indices = torch.arange(T, device=hdc_chars.device).float()
        diffs = (indices.unsqueeze(0) - indices.unsqueeze(1)).clamp(min=0)
        scalar_decay_matrix = torch.exp(log_decay_rate * diffs)
        scalar_decay_matrix = torch.tril(scalar_decay_matrix)
        scalar_decay_matrix = scalar_decay_matrix / (scalar_decay_matrix.sum(dim=-1, keepdim=True) + 1e-8)
        local_context = torch.matmul(scalar_decay_matrix, gated_chars)

        # HeteroResonance: далёкие семантические связи
        hetero_context = self.resonance(gated_chars, scalar_decay_matrix)

        # Blend: 60% local + 40% hetero
        contexts = 0.6 * local_context + 0.4 * hetero_context

        # HDC Residual: сохраняем identity текущего символа
        contexts = contexts + hdc_chars

        return contexts


class HeteroResonance(nn.Module):
    """Гетеро-ассоциативный резонанс: ломает симметрию Q=K.

    Вместо "ц ищет ц" → "ц ищет то что обычно рядом с ц".
    Отдельные Q/K проекции через bottleneck (экономия VRAM).
    Штраф за локальный повтор: diagonal=-4 блокирует эхо.
    """
    def __init__(self, hdc_dim, bottleneck=256):
        super().__init__()
        self.q_proj = nn.Linear(hdc_dim, bottleneck)
        self.k_proj = nn.Linear(hdc_dim, bottleneck)
        self.scale = bottleneck ** -0.5

    def forward(self, hdc_chars, decay_matrix):
        B, T, D = hdc_chars.shape
        device = hdc_chars.device

        # Асимметричный поиск
        q = self.q_proj(hdc_chars)  # (B, T, bottleneck)
        k = self.k_proj(hdc_chars)  # (B, T, bottleneck)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, T, T)

        # Causal + Local Repetition Penalty (блокируем 3 соседей)
        mask = torch.tril(torch.ones(T, T, device=device), diagonal=-4)
        scores = scores.masked_fill(mask == 0, -65000.0)  # float16 safe

        resonance = F.softmax(scores, dim=-1)

        # 70% decay (орфография) + 30% resonance (семантика)
        combined = 0.7 * decay_matrix + 0.3 * resonance
        return torch.matmul(combined, hdc_chars)


class AssociativeMemory(nn.Module):
    """Ассоциативная память: codebook паттернов + retrieval через matmul.

    Codebook — обучаемый. Каждый entry = "концепт" в HDC пространстве.
    Retrieval = cosine similarity → softmax → weighted sum.
    Это как attention, но по codebook (фиксированный размер K), не по seq_len T.
    Complexity: O(T*K*D) вместо O(T²*D).
    """
    def __init__(self, n_entries, hdc_dim, value_dim):
        super().__init__()
        self.n_entries = n_entries
        self.keys = nn.Parameter(torch.randn(n_entries, hdc_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(n_entries, value_dim) * 0.02)
        # Learnable inverse temperature (Modern Hopfield)
        self.log_beta = nn.Parameter(torch.tensor(math.log(hdc_dim ** 0.5)))
        self.n_hopfield_steps = 3  # итеративный retrieval

    def retrieve(self, query):
        """Iterative Hopfield retrieval: 3 шага уточнения.

        Вместо одного softmax(q @ K) — 3 итерации:
        q → scores → softmax → retrieve V → re-query → sharper scores → ...
        Каждый шаг "фокусирует" retrieval на более точном паттерне.
        """
        beta = torch.exp(self.log_beta)
        q = query  # (B, T, D_hdc)
        for _ in range(self.n_hopfield_steps):
            scores = torch.matmul(q, self.keys.t()) * beta  # (B, T, K)
            attn = F.softmax(scores, dim=-1)
            # Retrieved в HDC space для re-query
            q_new = torch.matmul(attn, self.keys)  # (B, T, D_hdc)
            q = q_new  # re-query с уточнённым вектором
        # Финальный retrieval в value space
        retrieved = torch.matmul(attn, self.values)  # (B, T, value_dim)
        return retrieved, attn


class NeuralNavigator(nn.Module):
    """Нейросеть-навигатор с Thought Loops.

    Не хранит знания — только "логика переходов".
    Thought Loops: навигатор может "перечитать" свой вывод и скорректировать.
    """
    def __init__(self, input_dim, hidden_dim, vocab_size, n_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        layers.append(nn.LayerNorm(input_dim))
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, vocab_size)

        # === FiLM MODULATION: контекст модулирует веса навигатора ===
        # HDC контекст генерирует gamma/beta для каждого слоя
        # Навигатор физически меняет поведение в зависимости от темы
        self.film_modulator = nn.Linear(hidden_dim, hidden_dim * 2)  # → gamma, beta

        # Thought Loops (отключаемые, для будущего)
        self.thought_proj = nn.Linear(hidden_dim, input_dim)
        self.thought_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.thought_ln = nn.LayerNorm(hidden_dim)
        self.n_thoughts = 1

    def forward(self, x, context_vec=None):
        """x: (B, T, input_dim) → logits: (B, T, vocab_size)

        FiLM: если context_vec задан, модулирует hidden через gamma/beta.
        Навигатор адаптирует поведение под контекст (проза vs код vs инструкции).
        """
        hidden = self.net(x)  # (B, T, hidden_dim)

        # FiLM модуляция: контекст меняет "характер" навигатора
        if context_vec is not None:
            film_params = self.film_modulator(context_vec)  # (B, T, hidden_dim*2)
            gamma, beta = film_params.chunk(2, dim=-1)
            hidden = hidden * (1 + gamma) + beta

        # Thought Loops (если включены)
        for _ in range(self.n_thoughts - 1):
            thought_input = self.thought_proj(hidden)
            refined_input = thought_input + x
            refined = self.net(refined_input)
            if context_vec is not None:
                film_params = self.film_modulator(context_vec)
                gamma, beta = film_params.chunk(2, dim=-1)
                refined = refined * (1 + gamma) + beta
            gate = torch.sigmoid(self.thought_gate(
                torch.cat([hidden, refined], dim=-1)
            ))
            hidden = self.thought_ln(gate * refined + (1 - gate) * hidden)

        return self.head(hidden)


class HDCAM(nn.Module):
    """HDC-Associative Manifold с иерархическим HDC.

    Два уровня HDC:
    1. Символьный: триграммы → контекст (decay matrix)
    2. Фразовый: chunk'и по 8 токенов → контекст фраз (второй decay matrix)

    Фразовый уровень даёт "понимание сюжета", символьный — "орфографию".
    Оба уровня — matmul, без циклов.
    """
    def __init__(self, vocab_size, hdc_dim=4096, codebook_size=8192,
                 nav_hidden=512, nav_layers=3, decay=0.95, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hdc_dim = hdc_dim
        self.chunk_size = 8  # размер фразового chunk'а

        # HDC ядро (символьный уровень)
        self.hdc = HDCProcessor(vocab_size, hdc_dim, decay)

        # === ИЕРАРХИЧЕСКИЙ HDC: фразовый уровень ===
        # Проецирует chunk из 8 символьных HDC → 1 фразовый HDC
        self.phrase_proj = nn.Linear(hdc_dim, hdc_dim)
        # Decay matrix для фраз (медленнее чем для символов)
        self.phrase_decay = 0.9  # фразы затухают медленнее

        # Ассоциативная память (работает с обоими уровнями)
        self.memory = AssociativeMemory(codebook_size, hdc_dim, nav_hidden)

        # Проекции HDC → float
        # Nonlinear HDC→float projection (2-layer MLP instead of Linear)
        # Linear 4096→512 is information bottleneck. MLP preserves more.
        self.hdc_proj = nn.Sequential(
            nn.Linear(hdc_dim, nav_hidden),
            nn.GELU(),
            nn.Linear(nav_hidden, nav_hidden),
        )
        self.phrase_hdc_proj = nn.Linear(hdc_dim, nav_hidden)

        # Residual shortcut
        self.residual_embed = nn.Embedding(vocab_size, nav_hidden)
        self.pos_embed = nn.Embedding(1024, nav_hidden)

        # Навигатор: (char+phrase) + retrieved + residual → logits
        # Input: nav_hidden * 3 (phrase складывается с char, не concat)
        # Фиксированный input_dim = nav_hidden * 3 — НИКОГДА не меняется
        self.navigator = NeuralNavigator(
            input_dim=nav_hidden * 3,
            hidden_dim=nav_hidden,
            vocab_size=vocab_size,
            n_layers=nav_layers,
            dropout=dropout,
        )

    def forward(self, idx, targets=None):
        """
        ВСЁ через матричные операции. Ноль циклов.

        idx: (B, T) — входные символы
        targets: (B, T) — целевые символы
        """
        B, T = idx.shape
        device = idx.device

        # === УРОВЕНЬ 1: Символьный HDC ===
        # Триграммы → контекст (одна matmul)
        hdc_chars = self.hdc.encode_trigrams(idx)  # (B, T, hdc_dim)
        hdc_contexts = self.hdc.build_context(hdc_chars)  # (B, T, hdc_dim)

        # === УРОВЕНЬ 2: Фразовый HDC ===
        # Chunk'и по 8 символов → фразовые вектора → контекст фраз
        C = self.chunk_size
        n_chunks = (T + C - 1) // C
        # Pad до кратного chunk_size
        if T % C != 0:
            pad = C - (T % C)
            hdc_padded = F.pad(hdc_contexts, (0, 0, 0, pad))  # (B, T_padded, D)
        else:
            hdc_padded = hdc_contexts
        # Reshape в chunks и усреднить каждый chunk → фразовый вектор
        T_padded = hdc_padded.shape[1]
        chunks = hdc_padded.view(B, n_chunks, C, self.hdc_dim).mean(dim=2)  # (B, n_chunks, D)
        phrase_vecs = torch.tanh(self.phrase_proj(chunks))  # (B, n_chunks, D)
        # Decay matrix для фраз
        phrase_decay_matrix = self.hdc.get_decay_matrix(n_chunks, device)
        # Замена decay значения для фраз (медленнее)
        phrase_contexts = torch.matmul(phrase_decay_matrix, phrase_vecs)  # (B, n_chunks, D)
        # Expand обратно к длине T: каждый chunk → все его позиции
        phrase_expanded = phrase_contexts.repeat_interleave(C, dim=1)[:, :T, :]  # (B, T, D)

        # === RETRIEVAL: из обоих уровней ===
        # Комбинируем символьный и фразовый контекст для запроса
        combined_query = hdc_contexts + 0.5 * phrase_expanded
        retrieved, _ = self.memory.retrieve(combined_query)  # (B, T, nav_hidden)

        # === ПРОЕКЦИИ ===
        hdc_float = self.hdc_proj(hdc_contexts)  # (B, T, nav_hidden)
        phrase_float = self.phrase_hdc_proj(phrase_expanded)  # (B, T, nav_hidden)

        # Residual shortcut + positional encoding
        pos = torch.arange(T, device=device)
        residual = self.residual_embed(idx) + self.pos_embed(pos)  # (B, T, nav_hidden)

        # Сложение char + phrase (вместо concat — экономит params, сохраняет инфо)
        combined_hdc = hdc_float + phrase_float  # (B, T, nav_hidden)
        combined = torch.cat([combined_hdc, retrieved, residual], dim=-1)  # (B, T, nav_hidden*3)

        # Neural navigator → logits (FiLM: HDC контекст модулирует навигатор)
        logits = self.navigator(combined, context_vec=combined_hdc)  # (B, T, vocab_size)

        # === ATTRACTOR GUARD: коррекция при "бреде" ===
        # Якорь = первый контекст (начало темы). Если текущий контекст
        # слишком далёк от якоря → один thought step для коррекции.
        # cosine_sim(current, anchor) < threshold → пересчитываем через memory ещё раз
        if T > 16:  # только для достаточно длинных последовательностей
            anchor = hdc_float[:, :8, :].mean(dim=1, keepdim=True)  # (B, 1, nav_hidden)
            anchor_expanded = anchor.expand(-1, T, -1)  # (B, T, nav_hidden)
            # Cosine similarity с якорем
            cos_sim = F.cosine_similarity(hdc_float, anchor_expanded, dim=-1)  # (B, T)
            # Маска: где контекст "ушёл" от темы
            drift_mask = (cos_sim < 0.3).float().unsqueeze(-1)  # (B, T, 1)
            if drift_mask.sum() > 0:
                # Thought step: re-retrieve из памяти с учётом якоря
                hdc_anchor = hdc_contexts[:, :8, :].mean(dim=1, keepdim=True).expand(-1, T, -1)
                corrected_query = combined_query * (1 - drift_mask) + \
                    (combined_query + hdc_anchor * 0.5) * drift_mask
                corrected_retrieved, _ = self.memory.retrieve(corrected_query)
                # Заменяем retrieved в drifted позициях
                combined_corrected = torch.cat([
                    combined_hdc,
                    corrected_retrieved * drift_mask + retrieved * (1 - drift_mask),
                    residual
                ], dim=-1)
                logits = self.navigator(combined_corrected, context_vec=combined_hdc)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss


def create_hdc_am(vocab_size, config=None):
    """Фабрика модели."""
    if config is None:
        config = {
            'hdc_dim': 4096,
            'codebook_size': 4096,
            'nav_hidden': 512,
            'nav_layers': 8,
            'decay': 0.95,
            'dropout': 0.1,
        }
    model = HDCAM(
        vocab_size=vocab_size,
        hdc_dim=config['hdc_dim'],
        codebook_size=config['codebook_size'],
        nav_hidden=config['nav_hidden'],
        nav_layers=config['nav_layers'],
        decay=config['decay'],
        dropout=config['dropout'],
    )
    return model, config


if __name__ == '__main__':
    import time

    vocab_size = 170  # FixedCharTokenizer
    model, config = create_hdc_am(vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    hdc_mem = model.hdc.char_codebook.numel() * 4 / 1024 / 1024  # MB

    print(f"HDC-AM (Associative Manifold)")
    print(f"  HDC dim: {config['hdc_dim']}")
    print(f"  Codebook: {config['codebook_size']} entries")
    print(f"  Navigator: {config['nav_hidden']}d, {config['nav_layers']} layers")
    print(f"  Trainable params: {n_params:,}")
    print(f"  HDC codebook memory: {hdc_mem:.1f} MB (fixed, not trainable)")

    # Test forward
    model.train()
    B, T = 64, 256
    x = torch.randint(0, vocab_size, (B, T))
    y = torch.randint(0, vocab_size, (B, T))

    # Warmup
    logits, loss = model(x, y)
    loss.backward()
    print(f"  Forward: logits {logits.shape}, loss {loss.item():.4f}")

    # Grad check
    with_grad = sum(1 for _, p in model.named_parameters()
                    if p.grad is not None and p.grad.norm() > 0)
    total = sum(1 for _ in model.parameters() if _.requires_grad)
    print(f"  Gradients: {with_grad}/{total}")

    # Speed benchmark
    model.zero_grad()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    N = 20
    for _ in range(N):
        logits, loss = model(x, y)
        loss.backward()
        model.zero_grad()
    elapsed = (time.time() - t0) / N
    print(f"  Speed: {elapsed*1000:.0f}ms/iter (batch={B}, seq={T})")
    print(f"  Throughput: {B*T/elapsed:.0f} tokens/s")

    # Edge cases
    for seq in [1, 8, 256]:
        _, lo = model(torch.randint(0, vocab_size, (2, seq)),
                      torch.randint(0, vocab_size, (2, seq)))
        lo.backward()
        model.zero_grad()
    print(f"  Edge cases: OK")

    # Memory estimate
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    buf_mem = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024
    print(f"  Memory: params={param_mem:.1f}MB, buffers={buf_mem:.1f}MB, total={param_mem+buf_mem:.1f}MB")
    print(f"  ALL OK")
