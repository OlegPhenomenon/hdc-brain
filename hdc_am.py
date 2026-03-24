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
    """HDC-Associative Manifold: полная модель.

    Pipeline (ВСЁ через matmul, никаких циклов):
    1. idx → HDC bipolar vectors (embedding lookup)
    2. HDC vectors → contexts (one matmul: decay_matrix @ vectors)
    3. contexts → retrieve from codebook (one matmul: context @ keys.T)
    4. context + retrieved + residual → neural navigator → logits
    """
    def __init__(self, vocab_size, hdc_dim=4096, codebook_size=8192,
                 nav_hidden=512, nav_layers=3, decay=0.95, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hdc_dim = hdc_dim

        # HDC ядро
        self.hdc = HDCProcessor(vocab_size, hdc_dim, decay)

        # Ассоциативная память
        self.memory = AssociativeMemory(codebook_size, hdc_dim, nav_hidden)

        # Проекция HDC → float для нейросети
        self.hdc_proj = nn.Linear(hdc_dim, nav_hidden)

        # Residual shortcut (доказано что работает)
        self.residual_embed = nn.Embedding(vocab_size, nav_hidden)
        self.pos_embed = nn.Embedding(1024, nav_hidden)  # позиционное кодирование

        # Навигатор: context_proj + retrieved + residual → logits
        # Input: nav_hidden * 3 (hdc_proj + retrieved + residual)
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

        # 1. HDC encoding: символы → bipolar вектора (lookup, O(1))
        # HDC триграммы: захватывают слоги/морфемы напрямую
        hdc_chars = self.hdc.encode_trigrams(idx)  # (B, T, hdc_dim)

        # 2. HDC context: ОДНА матричная операция
        hdc_contexts = self.hdc.build_context(hdc_chars)  # (B, T, hdc_dim)

        # 3. Associative retrieval: ОДНА матричная операция
        retrieved, attn_weights = self.memory.retrieve(hdc_contexts)  # (B, T, nav_hidden)

        # 4. Проекция HDC → float
        hdc_float = self.hdc_proj(hdc_contexts)  # (B, T, nav_hidden)

        # 5. Residual shortcut + positional encoding
        pos = torch.arange(T, device=device)
        residual = self.residual_embed(idx) + self.pos_embed(pos)  # (B, T, nav_hidden)

        # 6. Конкатенация: HDC context + retrieved pattern + residual
        combined = torch.cat([hdc_float, retrieved, residual], dim=-1)  # (B, T, nav_hidden*3)

        # 7. Neural navigator → logits (матричные операции)
        logits = self.navigator(combined)  # (B, T, vocab_size)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss


def create_hdc_am(vocab_size, config=None):
    """Фабрика модели."""
    if config is None:
        config = {
            'hdc_dim': 3072,
            'codebook_size': 6144,
            'nav_hidden': 384,
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
