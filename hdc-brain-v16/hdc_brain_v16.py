"""
HDC-Brain v16: Binary HDC + Logical Reasoning Layer

Эволюция: все бинарные компоненты из v15 + НОВОЕ:

1. HDC Fact Memory — явное хранение фактов как bind-пар
   Факт = bind(content, role) — ассоциативная связь
   Память = causal bundle (cumsum + sign = majority vote)
   Запрос = unbind(memory, query) — извлечение по ключу

2. HDC Logic Layer — правила вывода через bind/unbind
   Modus ponens: если memory содержит bind(A, R),
   то unbind(memory, R) ≈ A → вывод
   Это обобщённый логический вывод без обучения на датасетах логики

3. Structured Thought Loops — каждый thought имеет цель:
   - Thought 0: Comprehension (блоки, как раньше)
   - Thought 1+: Retrieval + Reasoning (факты + логика)

4. HDC Primitives с логической интерпретацией:
   bind(A, B)    = AND / ассоциация (A ⊙ B)
   bundle(A, B)  = OR / множество (majority vote)
   permute(A)    = порядок / роль
   negate(A)     = NOT (-A в bipolar)
   unbind(AB, A) = B (извлечение: bind — self-inverse)

Трансформер учит логику из данных. HDC имеет логику в операциях.
Forward = бинарный, Backward = float shadow через STE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# HDC Primitives — логические операции в гиперпространстве
# ============================================================

def ste_sign(x):
    """Sign with straight-through estimator. Forward: sign(), Backward: identity."""
    return (torch.sign(x) - x).detach() + x


def ste_sign_scaled(x):
    """Scaled sign: alpha * sign(x), gradient flows through x."""
    alpha = torch.mean(torch.abs(x), dim=-1, keepdim=True).clamp(min=1e-6)
    hard = alpha * torch.sign(x)
    hard = torch.where(hard == 0, alpha, hard)
    return (hard - x).detach() + x


def hdc_bind(a, b):
    """HDC Bind (⊙): ассоциация / AND.
    bind("кошка", "мяукает") = факт "кошка мяукает"
    Self-inverse: unbind(bind(A,B), A) = B
    """
    return ste_sign(a) * ste_sign(b)


def hdc_unbind(bound, key):
    """HDC Unbind: извлечение из связи. unbind = bind (self-inverse в bipolar).
    unbind(bind("кошка","мяукает"), "кошка") ≈ "мяукает"
    """
    return hdc_bind(bound, key)


def hdc_bundle(*vectors):
    """HDC Bundle: множество / OR. Majority vote через sign(sum).
    bundle("кошка", "собака") ≈ "животные"
    """
    stacked = torch.stack([ste_sign(v) for v in vectors], dim=0)
    return ste_sign(stacked.sum(dim=0))


def hdc_permute(x, shift=1):
    """HDC Permute: кодирование позиции / роли.
    permute¹(x) = субъект, permute²(x) = объект
    """
    return torch.roll(ste_sign(x), shift, dims=-1)


def hdc_negate(x):
    """HDC Negate: логическое отрицание. NOT A = -A в bipolar."""
    return -ste_sign(x)


def hdc_similarity(a, b):
    """Cosine-like similarity в HDC: доля совпадающих бит.
    Высокая → истина, низкая → ложь.
    """
    a_sign = ste_sign(a)
    b_sign = ste_sign(b)
    D = a.shape[-1]
    return torch.sum(a_sign * b_sign, dim=-1) / D


def binary_linear(x, weight):
    """Binary linear: sign(x) @ sign(weight.T), scaled."""
    x_bin = ste_sign(x)
    w_bin = ste_sign(weight)
    return F.linear(x_bin, w_bin)


# ============================================================
# Binary Components (proven in v15)
# ============================================================

class BinaryLinear(nn.Module):
    """Linear layer with binary weights (STE)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, x):
        return binary_linear(x, self.weight)


class HDCTrigramContext(nn.Module):
    """Trigram bind context from v15. Zero params, O(T*D).

    bigram  = token[t] ⊙ permute¹(token[t-1])
    trigram = bigram   ⊙ permute²(token[t-2])
    out[t]  = bundle(token[t], bigram, trigram)

    Сохраняет порядок слов через HDC permute.
    """
    def __init__(self, hdc_dim):
        super().__init__()
        self.hdc_dim = hdc_dim

    def forward(self, x):
        B, T, D = x.shape
        signs = ste_sign(x)

        out = torch.zeros_like(x)
        out[:, 0, :] = x[:, 0, :]

        if T > 1:
            shifted1 = torch.roll(signs[:, :-1, :], 1, dims=-1)
            bigrams = signs[:, 1:, :] * shifted1
            out[:, 1, :] = (2 * x[:, 1, :] + bigrams[:, 0, :]) / 3

        if T > 2:
            shifted2 = torch.roll(signs[:, :-2, :], 2, dims=-1)
            trigrams = bigrams[:, 1:, :] * shifted2
            out[:, 2:, :] = (x[:, 2:, :] + bigrams[:, 1:, :] + trigrams) / 3

        return out


class HDCBinaryAttention(nn.Module):
    """HDC Attention: binding Q/K/V + Hamming-like similarity.
    3*D params (not 3*D*D like standard attention).
    """
    def __init__(self, hdc_dim):
        super().__init__()
        self.hdc_dim = hdc_dim
        self.bv_q = nn.Parameter(torch.randn(hdc_dim) * 0.02)
        self.bv_k = nn.Parameter(torch.randn(hdc_dim) * 0.02)
        self.bv_v = nn.Parameter(torch.randn(hdc_dim) * 0.02)
        self.scale = hdc_dim ** -0.5

    def forward(self, x):
        B, T, D = x.shape
        bq = ste_sign(self.bv_q)
        bk = ste_sign(self.bv_k)
        bv = ste_sign(self.bv_v)

        Q = ste_sign(x * bq)
        K = ste_sign(x * bk)
        V = x * bv

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float('-inf'))
        attn = torch.sigmoid(scores * 4.0)
        attn = attn.masked_fill(~causal, 0.0)

        return torch.matmul(attn, V)


class BinaryController(nn.Module):
    """Controller with binary linear layers (STE)."""
    def __init__(self, hdc_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hdc_dim)
        self.down = BinaryLinear(hdc_dim, inner_dim)
        self.up = BinaryLinear(inner_dim, hdc_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = F.gelu(self.down(h))
        h = self.dropout(self.up(h))
        return x + h


class HDCBlockV16(nn.Module):
    """Full binary HDC block: Trigram + Binary Attention + Binary Controller."""
    def __init__(self, hdc_dim, controller_dim, dropout=0.1):
        super().__init__()
        self.trigram = HDCTrigramContext(hdc_dim)
        self.attention = HDCBinaryAttention(hdc_dim)
        self.controller = BinaryController(hdc_dim, controller_dim, dropout)
        self.ln_ctx = nn.LayerNorm(hdc_dim)
        self.ln_attn = nn.LayerNorm(hdc_dim)

    def forward(self, x):
        ctx = self.trigram(x)
        x = self.ln_ctx(x + ctx)
        attn = self.attention(x)
        x = self.ln_attn(x + attn)
        x = self.controller(x)
        return x


# ============================================================
# NEW: Logical Reasoning Components
# ============================================================

class HDCFactMemory(nn.Module):
    """Явная HDC память фактов.

    Запись: fact = bind(content, role_write)
    Память: causal bundle = cumsum(facts) → sign = majority vote
    Чтение: answer = unbind(memory, bind(query, role_read))

    Почему это работает:
    - bind создаёт уникальную ассоциацию (как XOR в бинарном)
    - bundle (cumsum+sign) хранит множество фактов (как OR)
    - unbind извлекает по ключу (bind — self-inverse)
    - Всё дифференцируемо через STE

    Логический смысл:
    - write = "запомнить факт"
    - read = "вспомнить по ассоциации"
    - Это аналог рабочей памяти / базы знаний

    Params: 2*D (roles) + 2*(D+1) (gates) ≈ 12K при D=4096
    """
    def __init__(self, hdc_dim):
        super().__init__()
        self.hdc_dim = hdc_dim
        # Learned role vectors: определяют "как записывать" и "как читать"
        self.role_write = nn.Parameter(torch.randn(hdc_dim) * 0.02)
        self.role_read = nn.Parameter(torch.randn(hdc_dim) * 0.02)
        # Gates: модель учится что записывать и насколько доверять прочитанному
        self.write_gate = nn.Linear(hdc_dim, 1)
        self.read_gate = nn.Linear(hdc_dim, 1)

    def write(self, x):
        """Записать факты в память. Returns memory state (B, T, D).

        Каждая позиция t записывает факт = bind(content[t], role_write).
        Память[t] = bundle(fact[0], fact[1], ..., fact[t]) = causal cumsum.
        """
        signs = ste_sign(x)
        role = ste_sign(self.role_write)

        # Факт = bind(content, role)
        facts = signs * role  # (B, T, D)

        # Gated write: модель решает, что важно запомнить
        w_gate = torch.sigmoid(self.write_gate(x))  # (B, T, 1)
        gated_facts = facts * w_gate

        # Causal bundle: memory[t] = sum(facts[0:t+1])
        memory = torch.cumsum(gated_facts, dim=1)  # (B, T, D)

        return memory

    def read(self, x, memory):
        """Прочитать из памяти. Returns enriched x.

        Запрос = bind(content, role_read)
        Ответ  = unbind(memory, query) = memory * query
        (bind — self-inverse в bipolar, поэтому unbind = bind)
        """
        signs = ste_sign(x)
        role = ste_sign(self.role_read)

        # Запрос
        query = signs * role  # (B, T, D)

        # Бинаризуем память (majority vote) перед чтением
        memory_sign = ste_sign(memory)

        # Unbind: извлекаем ассоциированную информацию
        retrieved = memory_sign * query  # (B, T, D)

        # Gate: насколько доверять прочитанному
        r_gate = torch.sigmoid(self.read_gate(x))  # (B, T, 1)

        return x + r_gate * retrieved


class HDCLogicLayer(nn.Module):
    """Логический вывод через HDC bind/unbind.

    Идея: у нас есть N "правил" (learned vectors R_i).
    Каждое правило кодирует тип отношения:
    - R_0 может выучить "is-a" (Сократ is-a человек)
    - R_1 может выучить "implies" (человек implies смертен)
    - R_2 может выучить "has-property"
    - R_3 может выучить "contradicts" (отрицание)

    Вывод: unbind(memory, R_i) извлекает заключение по правилу i.
    Это обобщённый modus ponens:
      Если память содержит bind(A, R_i),
      то unbind(memory, R_i) ≈ A.

    Модель сама учит, какие правила полезны.

    Params: n_rules*D (rules) + D*n_rules (gate) ≈ 50K при D=4096, n=4
    """
    def __init__(self, hdc_dim, n_rules=4):
        super().__init__()
        self.n_rules = n_rules
        self.hdc_dim = hdc_dim
        # Правила вывода: каждое правило — learned hypervector
        self.rules = nn.Parameter(torch.randn(n_rules, hdc_dim) * 0.02)
        # Gate: какие правила применять в каждой позиции
        self.inference_gate = nn.Linear(hdc_dim, n_rules)

    def forward(self, x, memory):
        """Применить правила к памяти, получить заключения.

        Для каждого правила R_i:
          conclusion_i = unbind(memory, R_i) = memory ⊙ R_i
        Финальный вывод = weighted sum по всем правилам.

        Args:
            x: hidden state (B, T, D) — для вычисления gate
            memory: fact memory (B, T, D) — из HDCFactMemory.write()
        Returns:
            conclusions: (B, T, D) — логические заключения
        """
        rule_signs = ste_sign(self.rules)  # (n_rules, D)
        memory_sign = ste_sign(memory)  # (B, T, D)

        # Какие правила активны для каждой позиции
        gates = torch.sigmoid(self.inference_gate(x))  # (B, T, n_rules)

        # Применяем каждое правило: unbind(memory, rule)
        # Делаем в цикле чтобы не раздувать память (O(BTD) вместо O(BTRD))
        result = torch.zeros_like(x)
        for i in range(self.n_rules):
            # Modus ponens: conclusion = unbind(memory, rule_i)
            conclusion = memory_sign * rule_signs[i]  # (B, T, D)
            result = result + gates[:, :, i:i+1] * conclusion

        return result


# ============================================================
# Main Model
# ============================================================

class HDCBrainV16(nn.Module):
    """HDC-Brain v16: Full Binary + Logical Reasoning.

    Архитектура:
    1. Binary Codebook (STE) → кодирование токенов
    2. Cyclic Position → позиционное кодирование
    3. 6x HDCBlock (Trigram + Attention + Controller) → понимание
    4. HDCFactMemory → запись/чтение фактов
    5. Thought Loops с HDCLogicLayer → рассуждение
    6. Binary Output → предсказание

    Отличия от трансформера:
    ┌───────────────┬────────────────────┬──────────────────────────┐
    │               │    Трансформер     │   HDC-Brain v16          │
    ├───────────────┼────────────────────┼──────────────────────────┤
    │ Логика        │ выучена из данных  │ встроена в операции       │
    │ Факты         │ размазаны по весам │ явно в bind-парах        │
    │ Рассуждение   │ CoT из датасета    │ Thought + Logic Layer    │
    │ Проверяемость │ чёрный ящик        │ unbind = интерпретация   │
    │ Галлюцинации  │ постоянные         │ сверка с фактами         │
    └───────────────┴────────────────────┴──────────────────────────┘
    """
    def __init__(self, vocab_size, hdc_dim=4096, max_seq_len=512,
                 n_blocks=6, controller_dim=768, dropout=0.1,
                 max_thoughts=4, use_checkpoint=False, n_rules=4):
        super().__init__()
        self.hdc_dim = hdc_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint
        self.max_thoughts = max_thoughts

        # === Binary Codebook (STE) ===
        self.codebook = nn.Parameter(torch.randn(vocab_size, hdc_dim) * 0.02)

        # === HDC Blocks (proven components) ===
        self.blocks = nn.ModuleList([
            HDCBlockV16(hdc_dim, controller_dim, dropout)
            for _ in range(n_blocks)
        ])

        # === NEW: Fact Memory ===
        self.fact_memory = HDCFactMemory(hdc_dim)

        # === NEW: Logic Layer ===
        self.logic_layer = HDCLogicLayer(hdc_dim, n_rules=n_rules)

        # === Thought Loop ===
        self.thought_gates = nn.Parameter(torch.zeros(max_thoughts))
        self.thought_pos = nn.Parameter(torch.randn(max_thoughts, hdc_dim) * 0.01)
        self.thought_ln = nn.LayerNorm(hdc_dim)

        # === Output ===
        self.output_ln = nn.LayerNorm(hdc_dim)
        self.output_scale = nn.Parameter(torch.tensor(hdc_dim ** -0.5))

    def _cyclic_position(self, x):
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        indices = (torch.arange(D, device=x.device).unsqueeze(0) - positions.unsqueeze(1)) % D
        return x.gather(2, indices.unsqueeze(0).expand(B, -1, -1))

    def forward(self, idx, targets=None, n_thoughts=None):
        B, T = idx.shape

        # === 1. Encode: Binary Codebook + Cyclic Position ===
        tokens = ste_sign_scaled(self.codebook[idx])
        tokens = self._cyclic_position(tokens)

        # === 2. Comprehension: process through blocks ===
        h = tokens
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)

        # === 3. Fact Memory: write observations, read associations ===
        memory = self.fact_memory.write(h)
        h = self.fact_memory.read(h, memory)

        # === 4. Thought Loops with Logic ===
        if n_thoughts is None:
            n_thoughts = self.max_thoughts if self.training else 2
        n_thoughts = min(n_thoughts, self.max_thoughts)

        if n_thoughts > 1:
            for t in range(1, n_thoughts):
                gate = torch.sigmoid(self.thought_gates[t])
                thought_input = self.thought_ln(h) + self.thought_pos[t]

                # Re-process through blocks
                thought = thought_input
                for block in self.blocks:
                    if self.use_checkpoint and self.training:
                        thought = torch.utils.checkpoint.checkpoint(block, thought, use_reentrant=False)
                    else:
                        thought = block(thought)

                # Apply logical inference using fact memory
                logic_conclusions = self.logic_layer(thought, memory)
                thought = thought + logic_conclusions

                # Enrich memory with new conclusions (thought writes new facts)
                new_facts = self.fact_memory.write(thought)
                memory = memory + new_facts  # bundle: accumulate knowledge

                # Gated update
                h = h + gate * (thought - h)

        # === 5. Output: binary similarity with codebook ===
        h = self.output_ln(h)
        h_bin = ste_sign(h)
        cb_bin = ste_sign(self.codebook)
        logits = F.linear(h_bin, cb_bin) * self.output_scale

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, start_ids, max_len=200, temperature=0.8, top_k=40,
                 n_thoughts=None):
        """Autoregressive generation."""
        idx = start_ids.clone()
        for _ in range(max_len):
            context = idx[:, -self.max_seq_len:]
            logits, _ = self(context, n_thoughts=n_thoughts)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    @torch.no_grad()
    def inspect_memory(self, idx, position=-1):
        """Inspect fact memory state at a given position.

        Возвращает top-K ближайших токенов к содержимому памяти.
        Полезно для интерпретации: "какие факты модель запомнила?"
        """
        B, T = idx.shape
        tokens = ste_sign_scaled(self.codebook[idx])
        tokens = self._cyclic_position(tokens)

        h = tokens
        for block in self.blocks:
            h = block(h)

        memory = self.fact_memory.write(h)
        memory_at_pos = memory[:, position, :]  # (B, D)

        # Какие токены похожи на содержимое памяти?
        cb_sign = torch.sign(self.codebook)
        mem_sign = torch.sign(memory_at_pos)
        similarity = torch.matmul(mem_sign, cb_sign.T)  # (B, vocab)

        return similarity

    @torch.no_grad()
    def inspect_rules(self):
        """Inspect learned logic rules.

        Для каждого правила показывает, какие пары токенов
        оно связывает сильнее всего.
        """
        rule_signs = torch.sign(self.rules)  # (n_rules, D)
        cb_sign = torch.sign(self.codebook)  # (vocab, D)

        # Для каждого правила: какие токены наиболее "похожи" на правило
        rule_token_sim = torch.matmul(rule_signs, cb_sign.T)  # (n_rules, vocab)

        return rule_token_sim

    def binary_param_count(self):
        """Count params that will be binary at inference."""
        binary = 0
        float_only = 0
        for name, p in self.named_parameters():
            if 'ln' in name or 'thought' in name or 'scale' in name or 'gate' in name:
                float_only += p.numel()
            else:
                binary += p.numel()
        return binary, float_only


def create_hdc_brain_v16(vocab_size, config=None):
    """Factory."""
    if config is None:
        config = {
            'hdc_dim': 4096,
            'max_seq_len': 512,
            'n_blocks': 6,
            'controller_dim': 768,
            'dropout': 0.1,
            'max_thoughts': 4,
            'use_checkpoint': True,
            'n_rules': 4,
        }
    model = HDCBrainV16(
        vocab_size=vocab_size,
        hdc_dim=config['hdc_dim'],
        max_seq_len=config['max_seq_len'],
        n_blocks=config['n_blocks'],
        controller_dim=config['controller_dim'],
        dropout=config['dropout'],
        max_thoughts=config.get('max_thoughts', 4),
        use_checkpoint=config.get('use_checkpoint', False),
        n_rules=config.get('n_rules', 4),
    )
    return model, config


# ============================================================
# Test & Benchmark
# ============================================================

if __name__ == '__main__':
    import time

    vocab_size = 16000
    config = {
        'hdc_dim': 4096,
        'max_seq_len': 512,
        'n_blocks': 6,
        'controller_dim': 768,
        'dropout': 0.1,
        'max_thoughts': 4,
        'use_checkpoint': False,
        'n_rules': 4,
    }
    model, config = create_hdc_brain_v16(vocab_size, config)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"HDC-Brain v16: Binary + Logic")
    print(f"  HDC dim: {config['hdc_dim']}")
    print(f"  Blocks: {config['n_blocks']} (Trigram + Attention + Controller)")
    print(f"  Controller: {config['controller_dim']}d inner")
    print(f"  Thought Loops: {config['max_thoughts']} max")
    print(f"  Logic Rules: {config['n_rules']}")
    print(f"  Vocab: {vocab_size} (BPE)")
    print(f"  Params: {n_params:,}")

    # Params breakdown
    codebook_params = model.codebook.numel()
    block_params = sum(p.numel() for block in model.blocks for p in block.parameters())
    memory_params = sum(p.numel() for p in model.fact_memory.parameters())
    logic_params = sum(p.numel() for p in model.logic_layer.parameters())
    thought_params = (model.thought_gates.numel() + model.thought_pos.numel() +
                      sum(p.numel() for p in model.thought_ln.parameters()))
    other_params = n_params - codebook_params - block_params - memory_params - logic_params - thought_params

    print(f"\n  Params breakdown:")
    print(f"    Codebook:     {codebook_params:>10,} ({codebook_params/n_params*100:.1f}%)")
    print(f"    Blocks:       {block_params:>10,} ({block_params/n_params*100:.1f}%)")
    print(f"    Fact Memory:  {memory_params:>10,} ({memory_params/n_params*100:.1f}%) [NEW]")
    print(f"    Logic Layer:  {logic_params:>10,} ({logic_params/n_params*100:.1f}%) [NEW]")
    print(f"    Thought Loop: {thought_params:>10,} ({thought_params/n_params*100:.1f}%)")
    print(f"    Other:        {other_params:>10,} ({other_params/n_params*100:.1f}%)")
    print(f"    TOTAL:        {n_params:>10,}")

    binary, float_only = model.binary_param_count()
    print(f"\n  Binary at inference: {binary:,} ({binary/n_params*100:.1f}%)")
    print(f"  Float only:         {float_only:,} ({float_only/n_params*100:.1f}%)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()

    B, T = 8, 256
    x = torch.randint(0, vocab_size, (B, T), device=device)
    y = torch.randint(0, vocab_size, (B, T), device=device)

    # Forward + backward (2 thoughts = comprehension + reasoning)
    logits, loss = model(x, y, n_thoughts=2)
    loss.backward()
    print(f"\n  Forward (2 thoughts): logits {logits.shape}, loss {loss.item():.4f}")

    # Gradient check
    with_grad = sum(1 for _, p in model.named_parameters()
                    if p.grad is not None and p.grad.norm() > 0)
    total = sum(1 for _ in model.parameters() if _.requires_grad)
    print(f"  Gradients: {with_grad}/{total}")

    # Check new components have gradients
    for name in ['fact_memory.role_write', 'fact_memory.role_read',
                 'logic_layer.rules', 'logic_layer.inference_gate.weight']:
        parts = name.split('.')
        param = model
        for part in parts:
            param = getattr(param, part)
        if hasattr(param, 'grad') and param.grad is not None:
            print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
        else:
            print(f"  {name}: NO GRADIENT!")

    # Speed: 1 thought vs 2 vs 4
    model.zero_grad()
    for n_t in [1, 2, 4]:
        t0 = time.time()
        N = 5
        for _ in range(N):
            logits, loss = model(x, y, n_thoughts=n_t)
            loss.backward()
            model.zero_grad()
        elapsed = (time.time() - t0) / N
        print(f"  Speed ({n_t} thoughts): {elapsed*1000:.0f}ms/iter")

    # HDC Logic demo: bind/unbind
    print(f"\n  HDC Logic Primitives Demo:")
    a = torch.randn(4096, device=device)
    b = torch.randn(4096, device=device)

    # bind + unbind = recovery
    ab = hdc_bind(a, b)
    recovered_b = hdc_unbind(ab, a)
    sim = hdc_similarity(recovered_b.unsqueeze(0), ste_sign(b).unsqueeze(0))
    print(f"    bind(A,B) then unbind(AB,A) → similarity to B: {sim.item():.4f} (1.0 = perfect)")

    # Negation
    neg_a = hdc_negate(a)
    sim_neg = hdc_similarity(ste_sign(a).unsqueeze(0), neg_a.unsqueeze(0))
    print(f"    similarity(A, NOT A): {sim_neg.item():.4f} (-1.0 = perfect negation)")

    # Bundle
    c = torch.randn(4096, device=device)
    abc = hdc_bundle(a, b, c)
    sim_a = hdc_similarity(abc.unsqueeze(0), ste_sign(a).unsqueeze(0))
    sim_b = hdc_similarity(abc.unsqueeze(0), ste_sign(b).unsqueeze(0))
    print(f"    bundle(A,B,C) → similarity to A: {sim_a.item():.4f}, to B: {sim_b.item():.4f}")

    print(f"\n  ALL OK")
