"""
HDC-Brain v14.2: Optimized HDC language model with Logic Layer.

Принципы v14 + оптимизации + логика:
  STE Bipolar Codebook → Cyclic Permutation
  → HDCBlock × N (Memory + Attention + LogicLayer + Controller)
  → ThoughtLoop → Output

Оптимизации:
  - D=2048 (компромисс скорость/качество)
  - Линейная EMA-память O(T) вместо O(T²)
  - KV-cache для инференса
  - Бинарный режим инференса (XNOR)
  - torch.compile совместимость (развёрнутый ThoughtLoop)

LogicLayer:
  - Модель учит СТРУКТУРЫ предложений (паттерны ролей)
  - Структуры хранятся отдельно от слов
  - При предсказании: сначала структура → потом слово
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Утилиты
# ============================================================

def _ste_sign(w):
    """STE бинаризация: forward = sign, backward = straight-through."""
    alpha = torch.mean(torch.abs(w), dim=-1, keepdim=True)
    hard = alpha * torch.sign(w)
    hard = torch.where(hard == 0, alpha, hard)
    return (hard - w).detach() + w


def _ste_sign_vec(w):
    """STE для одномерного вектора (binding vectors)."""
    alpha = torch.mean(torch.abs(w))
    hard = alpha * torch.sign(w)
    hard = torch.where(hard == 0, alpha * torch.ones_like(hard), hard)
    return (hard - w).detach() + w


# ============================================================
# Блоки
# ============================================================

class HDCMemoryEMA(nn.Module):
    """HDC память через экспоненциальное скользящее среднее.
    Полностью векторизованная через grouped conv1d. O(T).
    """
    def __init__(self, hdc_dim):
        super().__init__()
        self.decay_logit = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        B, T, D = x.shape
        decay = torch.sigmoid(self.decay_logit)
        powers = decay ** torch.arange(T, device=x.device, dtype=x.dtype)
        kernel = powers.flip(0).view(1, 1, T).expand(D, 1, T)
        x_t = x.transpose(1, 2)
        x_padded = F.pad(x_t, (T - 1, 0))
        context = F.conv1d(x_padded, kernel, groups=D)
        return context.transpose(1, 2)


class HDCAttention(nn.Module):
    """Boolean causal attention через binding.
    3×D параметров. Поддерживает KV-cache.
    """
    def __init__(self, hdc_dim):
        super().__init__()
        self.hdc_dim = hdc_dim
        self.bv_q = nn.Parameter(torch.randn(hdc_dim) * 0.5)
        self.bv_k = nn.Parameter(torch.randn(hdc_dim) * 0.5)
        self.bv_v = nn.Parameter(torch.randn(hdc_dim) * 0.5)
        self.scale = hdc_dim ** -0.5

    def forward(self, x, kv_cache=None):
        B, T, D = x.shape
        bv_q = _ste_sign_vec(self.bv_q)
        bv_k = _ste_sign_vec(self.bv_k)
        bv_v = _ste_sign_vec(self.bv_v)

        Q = x * bv_q
        K_new = x * bv_k
        V_new = x * bv_v

        if kv_cache is not None:
            K_old, V_old = kv_cache
            K = torch.cat([K_old, K_new], dim=1)
            V = torch.cat([V_old, V_new], dim=1)
        else:
            K, V = K_new, V_new

        T_full = K.shape[1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Каузальная маска: Q позиции видят только K позиции <= себя
        q_pos = torch.arange(T_full - T, T_full, device=x.device)
        k_pos = torch.arange(T_full, device=x.device)
        causal = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0)  # (T, T_full)
        scores = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))
        attn = torch.sigmoid(scores * 4.0)
        attn = attn.masked_fill(~causal.unsqueeze(0), 0.0)

        out = torch.matmul(attn, V)
        new_cache = (K, V)
        return out, new_cache


class LogicLayer(nn.Module):
    """Слой логики — учит структуры предложений.

    Идея: модель учит N "правил" (структурных паттернов).
    Каждое правило — биполярный вектор в HDC пространстве.

    Как это работает:
    1. Входной токен привязывается (bind) к каждому правилу
    2. Результат = "что этот токен означает в контексте правила"
    3. Модель учит какие правила активировать для какого контекста
    4. Это даёт структурное понимание: не "какое слово следующее",
       а "какая РОЛЬ следующая, и какое слово её заполняет"

    Параметры: ~n_rules × D + D = минимум.
    """
    def __init__(self, hdc_dim, n_rules=32):
        super().__init__()
        self.hdc_dim = hdc_dim
        self.n_rules = n_rules

        # Структурные правила — каждое правило = биполярный вектор
        self.rules = nn.Parameter(torch.randn(n_rules, hdc_dim) * 0.5)

        # Binding vector для запроса (вместо nn.Linear — HDC-стиль)
        self.query_bind = nn.Parameter(torch.randn(hdc_dim) * 0.5)

        # Gate: модель решает сколько логики добавить
        self.logic_gate = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1)≈0.27, осторожный старт

    def forward(self, x):
        B, T, D = x.shape

        # Бинаризуем правила через STE
        rules = _ste_sign(self.rules)  # (n_rules, D)

        # Запрос через binding (HDC-стиль, не матричное умножение)
        query = x * _ste_sign_vec(self.query_bind)  # (B, T, D)

        # Какие правила активны для каждого токена
        similarity = torch.matmul(query, rules.T) * (D ** -0.5)  # (B, T, n_rules)
        rule_weights = torch.sigmoid(similarity * 4.0)  # (B, T, n_rules)

        # Абдукция: bind каждого токена с каждым правилом
        # x ⊙ rule = "что x означает в контексте rule"
        # (B, T, 1, D) * (n_rules, D) → (B, T, n_rules, D)
        bindings = x.unsqueeze(2) * rules.unsqueeze(0).unsqueeze(0)

        # Взвешенная сумма по правилам → (B, T, D)
        logic_output = (rule_weights.unsqueeze(-1) * bindings).sum(dim=2)

        gate = torch.sigmoid(self.logic_gate)
        return x + gate * (logic_output - x)


class ControllerBlock(nn.Module):
    """Residual: LN → down → GELU → up."""
    def __init__(self, hdc_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hdc_dim)
        self.down = nn.Linear(hdc_dim, inner_dim)
        self.up = nn.Linear(inner_dim, hdc_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = F.gelu(self.down(h))
        h = self.dropout(self.up(h))
        return x + h


class HDCBlock(nn.Module):
    """Memory + Attention + Logic + Controller."""
    def __init__(self, hdc_dim, controller_dim, n_rules=32, dropout=0.1):
        super().__init__()
        self.memory = HDCMemoryEMA(hdc_dim)
        self.attention = HDCAttention(hdc_dim)
        self.logic = LogicLayer(hdc_dim, n_rules)
        self.controller = ControllerBlock(hdc_dim, controller_dim, dropout)
        self.ln_mem = nn.LayerNorm(hdc_dim)
        self.ln_attn = nn.LayerNorm(hdc_dim)
        self.ln_logic = nn.LayerNorm(hdc_dim)

    def forward(self, x, kv_cache=None):
        mem = self.memory(x)
        x = self.ln_mem(x + mem)
        attn, new_cache = self.attention(x, kv_cache)
        x = self.ln_attn(x + attn)
        x = self.ln_logic(self.logic(x))
        x = self.controller(x)
        return x, new_cache


class ThoughtLoop(nn.Module):
    """Thought Loop с развёрнутой структурой для torch.compile."""
    def __init__(self, hdc_dim, max_thoughts=3):
        super().__init__()
        self.max_thoughts = max_thoughts
        self.thought_gates = nn.Parameter(torch.zeros(max_thoughts))
        self.thought_pos = nn.Parameter(torch.randn(max_thoughts, hdc_dim) * 0.01)
        self.ln = nn.LayerNorm(hdc_dim)

    def _run_blocks(self, h, blocks):
        for block in blocks:
            h, _ = block(h)
        return h

    def forward(self, h, blocks, n_thoughts=None):
        if n_thoughts is None:
            n_thoughts = self.max_thoughts
        n_thoughts = min(n_thoughts, self.max_thoughts)

        h = self._run_blocks(h, blocks)

        if n_thoughts <= 1:
            return h

        for t in range(1, n_thoughts):
            gate = torch.sigmoid(self.thought_gates[t])
            thought_input = self.ln(h) + self.thought_pos[t]
            thought = self._run_blocks(thought_input, blocks)
            h = h + gate * (thought - h)

        return h


# ============================================================
# Основная модель
# ============================================================

class HDCBrainV14_2(nn.Module):
    """HDC-Brain v14.2: Optimized + Logic Layer.

    Encoder:  BPE Token → STE Bipolar Codebook → Cyclic Permutation
    Process:  ThoughtLoop(N × HDCBlock(Memory + Attention + Logic + Controller))
    Decoder:  Output @ Codebook^T → logits (weight-tied)

    Новое:
    - LogicLayer в каждом блоке (структурные правила)
    - KV-cache для быстрого инференса
    - Бинарный режим для edge-устройств
    """
    def __init__(self, vocab_size, hdc_dim=2048, max_seq_len=128,
                 n_blocks=4, controller_dim=512, n_rules=32,
                 dropout=0.1, max_thoughts=3):
        super().__init__()
        self.hdc_dim = hdc_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_blocks = n_blocks

        # STE Bipolar Codebook
        self.codebook = nn.Parameter(torch.randn(vocab_size, hdc_dim) * 0.02)

        # HDC Blocks с LogicLayer
        self.blocks = nn.ModuleList([
            HDCBlock(hdc_dim, controller_dim, n_rules, dropout)
            for _ in range(n_blocks)
        ])

        # Thought Loop
        self.thought_loop = ThoughtLoop(hdc_dim, max_thoughts)

        # Output
        self.output_ln = nn.LayerNorm(hdc_dim)
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        # Бинарный кодбук для инференса (заполняется при вызове to_binary())
        self._binary_codebook = None

    def _ste_encode(self, idx):
        real = self.codebook[idx]
        return _ste_sign(real)

    def _cyclic_position(self, x):
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        indices = (torch.arange(D, device=x.device).unsqueeze(0) - positions.unsqueeze(1)) % D
        return x.gather(2, indices.unsqueeze(0).expand(B, -1, -1))

    def forward(self, idx, targets=None, n_thoughts=None):
        B, T = idx.shape
        tokens = self._ste_encode(idx)
        tokens = self._cyclic_position(tokens)
        h = self.thought_loop(tokens, self.blocks, n_thoughts)
        h = self.output_ln(h)
        logits = F.linear(h, self.codebook) * self.output_scale

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    # --------------------------------------------------------
    # KV-cache инференс
    # --------------------------------------------------------
    @torch.no_grad()
    def generate(self, start_ids, max_len=100, temperature=0.8, top_k=40,
                 n_thoughts=None, use_cache=True):
        """Авторегрессивная генерация с KV-cache."""
        idx = start_ids.clone()

        if not use_cache:
            return self._generate_no_cache(idx, max_len, temperature, top_k, n_thoughts)

        # Prefill: обработать весь контекст
        context = idx[:, -self.max_seq_len:]
        caches = [None] * self.n_blocks

        tokens = self._ste_encode(context)
        tokens = self._cyclic_position(tokens)

        # Первый thought — с кэшированием
        h = tokens
        for i, block in enumerate(self.blocks):
            h, caches[i] = block(h)

        h_out = self.output_ln(h)
        logits = F.linear(h_out, self.codebook) * self.output_scale
        next_id = self._sample(logits[:, -1, :], temperature, top_k)
        idx = torch.cat([idx, next_id], dim=1)

        # Decode: по одному токену с кэшем
        for _ in range(max_len - 1):
            new_tok = self._ste_encode(next_id)  # (B, 1, D)
            # Позиция = текущая длина - 1
            pos = idx.shape[1] - 1
            pos_t = torch.tensor([pos], device=idx.device)
            d_idx = (torch.arange(self.hdc_dim, device=idx.device) - pos_t) % self.hdc_dim
            new_tok = new_tok.gather(2, d_idx.unsqueeze(0).unsqueeze(0).expand(new_tok.shape[0], 1, -1))

            h = new_tok
            for i, block in enumerate(self.blocks):
                h, caches[i] = block(h, kv_cache=caches[i])

            h_out = self.output_ln(h)
            logits = F.linear(h_out, self.codebook) * self.output_scale
            next_id = self._sample(logits[:, -1, :], temperature, top_k)
            idx = torch.cat([idx, next_id], dim=1)

        return idx

    @torch.no_grad()
    def _generate_no_cache(self, idx, max_len, temperature, top_k, n_thoughts):
        for _ in range(max_len):
            context = idx[:, -self.max_seq_len:]
            logits, _ = self(context, n_thoughts=n_thoughts)
            next_id = self._sample(logits[:, -1, :], temperature, top_k)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def _sample(self, logits, temperature, top_k):
        logits = logits / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

    # --------------------------------------------------------
    # Бинарный инференс
    # --------------------------------------------------------
    @torch.no_grad()
    def to_binary(self):
        """Конвертировать кодбук в бинарный для быстрого инференса.
        Кодбук: float → {-1, +1} stored as bool (True = +1).
        Similarity через XNOR + popcount вместо float dot product.
        """
        binary = (torch.sign(self.codebook.data) > 0)  # bool tensor
        self._binary_codebook = binary  # (V, D) bool
        return self

    @torch.no_grad()
    def binary_logits(self, h):
        """Быстрые logits через XNOR + popcount.
        h: (B, T, D) float → бинаризуем → xnor с кодбуком → count.
        """
        if self._binary_codebook is None:
            self.to_binary()

        h_bin = (h > 0)  # (B, T, D) bool
        cb = self._binary_codebook  # (V, D) bool

        # XNOR = NOT XOR. Agreement count = D - hamming_distance
        # (B, T, D) xnor (V, D) → popcount → (B, T, V)
        # Через матмул: agreement = h_bin @ cb.T + (1-h_bin) @ (1-cb).T
        #             = h_bin @ cb.T + (1-h_bin) @ (1-cb.T)
        # Проще: 2 * (h_float @ cb_float) + D  (где float = 2*bool - 1)
        h_f = h_bin.float() * 2 - 1  # {-1, +1}
        cb_f = cb.float() * 2 - 1     # {-1, +1}
        agreement = torch.matmul(h_f, cb_f.T)  # (B, T, V)
        return agreement  # пропорционально cosine similarity

    # --------------------------------------------------------
    # Статистика
    # --------------------------------------------------------
    def param_breakdown(self):
        n_total = sum(p.numel() for p in self.parameters())
        n_cb = self.codebook.numel()
        n_blocks = sum(p.numel() for b in self.blocks for p in b.parameters())
        n_logic = sum(
            p.numel() for b in self.blocks for n, p in b.logic.named_parameters()
        )
        n_thought = sum(p.numel() for p in self.thought_loop.parameters())
        return {
            "total": n_total,
            "codebook": n_cb,
            "blocks": n_blocks,
            "logic_layers": n_logic,
            "thought_loop": n_thought,
        }


# ============================================================
# Конфигурации
# ============================================================

CONFIGS = {
    "tiny": {
        "hdc_dim": 512,
        "max_seq_len": 128,
        "n_blocks": 3,
        "controller_dim": 128,
        "n_rules": 16,
        "max_thoughts": 2,
        "dropout": 0.1,
    },
    "small": {
        "hdc_dim": 1024,
        "max_seq_len": 128,
        "n_blocks": 4,
        "controller_dim": 256,
        "n_rules": 32,
        "max_thoughts": 3,
        "dropout": 0.1,
    },
    "medium": {
        "hdc_dim": 2048,
        "max_seq_len": 128,
        "n_blocks": 4,
        "controller_dim": 512,
        "n_rules": 32,
        "max_thoughts": 3,
        "dropout": 0.1,
    },
}


def create_model(vocab_size, config_name="small"):
    cfg = CONFIGS[config_name].copy()
    model = HDCBrainV14_2(vocab_size=vocab_size, **cfg)
    return model, cfg


if __name__ == "__main__":
    vocab_size = 16000
    print("HDC-Brain v14.2 — model sizes\n")
    for name, cfg in CONFIGS.items():
        model = HDCBrainV14_2(vocab_size=vocab_size, **cfg)
        bd = model.param_breakdown()
        print(f"{name:>8}: {bd['total']:>12,} params")
        print(f"          codebook:  {bd['codebook']:>10,} ({bd['codebook']/bd['total']*100:.0f}%)")
        print(f"          blocks:    {bd['blocks']:>10,} ({bd['blocks']/bd['total']*100:.0f}%)")
        print(f"          logic:     {bd['logic_layers']:>10,} ({bd['logic_layers']/bd['total']*100:.0f}%)")
        print(f"          thoughts:  {bd['thought_loop']:>10,}")
        print()

    # Quick forward test
    print("Forward test (small config)...")
    model = HDCBrainV14_2(vocab_size=vocab_size, **CONFIGS["small"])
    x = torch.randint(0, vocab_size, (2, 64))
    y = torch.randint(0, vocab_size, (2, 64))
    logits, loss = model(x, y, n_thoughts=1)
    print(f"  logits: {logits.shape}, loss: {loss.item():.3f}")

    # Binary logits test
    model.to_binary()
    h = torch.randn(2, 64, CONFIGS["small"]["hdc_dim"])
    bl = model.binary_logits(h)
    print(f"  binary_logits: {bl.shape}")
    print("\nOK")
