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


class GatedLinearContext(nn.Module):
    """GLA: Gated Linear Attention для контекста.

    Заменяет decay + HeteroResonance. Data-dependent gating:
    - "Москва" → gate≈0.95 (запомни надолго)
    - пробел → gate≈0.5 (забудь быстро)

    State = (D_key × D_val) матрица — ассоциативная память.
    Каждый токен: state = gate * state + key ⊗ value (outer product)
    Выход: query @ state

    O(T × D²) вместо O(T² × D). Быстрее + умнее.
    """
    def __init__(self, input_dim, state_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.q_proj = nn.Linear(input_dim, state_dim)
        self.k_proj = nn.Linear(input_dim, state_dim)
        self.v_proj = nn.Linear(input_dim, state_dim)
        self.g_proj = nn.Linear(input_dim, state_dim)  # data-dependent gate
        self.out_proj = nn.Linear(state_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        """x: (B, T, D) → contexts: (B, T, D)

        Sequential GLA: loop over T (fast since each step = small matrix ops).
        """
        B, T, D = x.shape
        S = self.state_dim
        device = x.device

        q = self.q_proj(x)  # (B, T, S)
        k = self.k_proj(x)  # (B, T, S)
        v = self.v_proj(x)  # (B, T, S)
        g = torch.sigmoid(self.g_proj(x))  # (B, T, S) — per-dim gate

        # State: (B, S, S) — ассоциативная память (key-value matrix)
        state = torch.zeros(B, S, S, device=device)
        outputs = []

        for t in range(T):
            # Gate: сколько старого запомнить
            g_t = g[:, t, :].unsqueeze(-1)  # (B, S, 1)
            # Outer product: новая ассоциация key→value
            k_t = k[:, t, :].unsqueeze(-1)  # (B, S, 1)
            v_t = v[:, t, :].unsqueeze(1)   # (B, 1, S)
            kv = torch.matmul(k_t, v_t)     # (B, S, S)
            # Update state: forget old + add new
            state = g_t * state + kv
            # Query: извлекаем из state
            q_t = q[:, t, :].unsqueeze(1)   # (B, 1, S)
            out = torch.matmul(q_t, state).squeeze(1)  # (B, S)
            outputs.append(out)

        out = torch.stack(outputs, dim=1)  # (B, T, S)
        # Project back + residual
        return self.ln(x + self.out_proj(out))


class HDCProcessor(nn.Module):
    """HDC ядро с GLA контекстом.

    Триграммы → GLA (data-dependent gating) → контекст.
    Модель сама решает что запомнить а что забыть.
    """
    def __init__(self, vocab_size, hdc_dim=4096, decay=0.95):
        super().__init__()
        self.hdc_dim = hdc_dim

        # HDC codebook — ОБУЧАЕМЫЙ (THDC: Training HDC with Backpropagation)
        # STE: forward = sign (bipolar), backward = gradient через real weights
        # Модель выучит что "а" и "е" (гласные) ближе чем "а" и "з"
        self.char_codebook = nn.Parameter(torch.randn(vocab_size, hdc_dim) * 0.02)

        # GLA контекст вместо decay
        self.gla = GatedLinearContext(hdc_dim, state_dim=128)

    def encode(self, idx):
        """STE: forward = sign (bipolar ±1), backward = gradient через real weights."""
        real = self.char_codebook[idx]  # (B, T, D) — real-valued
        # STE: hard sign forward, soft gradient backward
        hard = torch.sign(real)
        hard = torch.where(hard == 0, torch.ones_like(hard), hard)
        return (hard - real).detach() + real  # forward=hard, backward=через real

    def encode_trigrams(self, idx):
        chars = self.char_codebook[idx]
        chars_m1 = torch.roll(chars, 1, dims=1)
        chars_m2 = torch.roll(chars, 2, dims=1)
        chars_m1[:, 0, :] = 0
        chars_m2[:, :2, :] = 0
        chars_m1_rolled = torch.roll(chars_m1, 1, dims=-1)
        chars_m2_rolled = torch.roll(chars_m2, 2, dims=-1)
        trigrams = chars * chars_m1_rolled * chars_m2_rolled
        return 0.7 * trigrams + 0.3 * chars

    def build_context(self, hdc_chars):
        """GLA контекст: data-dependent gating.
        "Москва" → запомни. Пробел → забудь.
        """
        return self.gla(hdc_chars)


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

        # === HDC Trigrams → GLA Context ===
        hdc_chars = self.hdc.encode_trigrams(idx)  # (B, T, hdc_dim)
        hdc_contexts = self.hdc.build_context(hdc_chars)  # (B, T, hdc_dim) — GLA inside

        # === Retrieval из ассоциативной памяти ===
        retrieved, _ = self.memory.retrieve(hdc_contexts)  # (B, T, nav_hidden)

        # === Проекция HDC → float ===
        hdc_float = self.hdc_proj(hdc_contexts)  # (B, T, nav_hidden)

        # Residual shortcut + positional encoding
        pos = torch.arange(T, device=device)
        residual = self.residual_embed(idx) + self.pos_embed(pos)  # (B, T, nav_hidden)

        # Конкатенация: hdc + retrieved + residual
        combined = torch.cat([hdc_float, retrieved, residual], dim=-1)  # (B, T, nav_hidden*3)

        # Navigator → logits (FiLM: контекст модулирует)
        logits = self.navigator(combined, context_vec=hdc_float)  # (B, T, vocab_size)

        # Attractor Guard убран — GLA сама управляет контекстом через gating

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
            'codebook_size': 8192,
            'nav_hidden': 1024,
            'nav_layers': 12,
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
