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
        self.decay = decay

        # Фиксированный HDC codebook: каждый символ = случайный bipolar вектор
        # НЕ обучаемый — это "атомы" пространства
        codebook = torch.sign(torch.randn(vocab_size, hdc_dim))
        codebook[codebook == 0] = 1.0
        self.register_buffer('char_codebook', codebook)

        # Precompute decay matrix для разных seq_len (кешируется)
        self._decay_cache = {}

    def get_decay_matrix(self, T, device):
        """Causal decay matrix: (T, T) lower-triangular с экспоненциальным затуханием."""
        if T not in self._decay_cache or self._decay_cache[T].device != device:
            indices = torch.arange(T, device=device)
            # decay_matrix[i,j] = decay^(i-j) if j <= i, else 0
            diffs = (indices.unsqueeze(1) - indices.unsqueeze(0)).float()
            matrix = self.decay ** diffs.clamp(min=0)
            matrix = torch.tril(matrix)  # causal: будущее не видно
            # Нормализуем строки чтобы сумма ≈ 1
            matrix = matrix / matrix.sum(dim=-1, keepdim=True)
            self._decay_cache[T] = matrix
        return self._decay_cache[T]

    def encode(self, idx):
        """Символы → HDC bipolar вектора. Lookup, мгновенно."""
        return self.char_codebook[idx]  # (B, T, D_hdc)

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
        """Весь контекст = ОДНА матричная операция.

        context[t] = Σ(decay^(t-i) * hdc_chars[i]) для i=0..t
        = causal_decay_matrix @ hdc_chars

        Args: hdc_chars (B, T, D_hdc) — bipolar вектора символов
        Returns: contexts (B, T, D_hdc) — HDC контекст для каждой позиции
        """
        B, T, D = hdc_chars.shape
        decay_matrix = self.get_decay_matrix(T, hdc_chars.device)
        # ОДНА матричная операция: (T,T) @ (B,T,D) → (B,T,D)
        contexts = torch.matmul(decay_matrix, hdc_chars)
        return contexts


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
        # Keys: HDC-пространство (bipolar-like, обучаемые)
        self.keys = nn.Parameter(torch.randn(n_entries, hdc_dim) * 0.02)
        # Values: float пространство для нейросети
        self.values = nn.Parameter(torch.randn(n_entries, value_dim) * 0.02)
        self.scale = hdc_dim ** -0.5

    def retrieve(self, query):
        """Retrieval = один matmul.

        Args: query (B, T, D_hdc) — HDC контексты
        Returns: retrieved (B, T, value_dim) — извлечённые паттерны
                 attn_weights (B, T, K) — что именно извлекли
        """
        # Cosine-like similarity: query @ keys.T
        # (B, T, D_hdc) @ (D_hdc, K) → (B, T, K)
        scores = torch.matmul(query, self.keys.t()) * self.scale
        attn = F.softmax(scores, dim=-1)
        # Weighted sum of values: (B, T, K) @ (K, value_dim) → (B, T, value_dim)
        retrieved = torch.matmul(attn, self.values)
        return retrieved, attn


class NeuralNavigator(nn.Module):
    """Маленькая нейросеть-навигатор: HDC context + retrieved → logits.

    Не хранит знания — только "логика переходов".
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

    def forward(self, x):
        """x: (B, T, input_dim) → logits: (B, T, vocab_size)"""
        return self.head(self.net(x))


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
        self.hdc_proj = nn.Linear(hdc_dim, nav_hidden)
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

        # Neural navigator → logits
        logits = self.navigator(combined)  # (B, T, vocab_size)

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
                logits = self.navigator(combined_corrected)

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
            'nav_layers': 4,
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
