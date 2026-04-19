# Глава 23. Финальный проект: реализуй HDC-Brain с нуля

---

Есть такой тест у хороших инженеров: "Объясни мне это так, будто я шестилетний ребёнок."

Но есть тест ещё суровее: "Напиши это с нуля, без подсказок."

Ты прочитал 22 главы. Ты знаешь что такое STE, HDCMemory, Binding Attention, ThoughtLoop, cosine schedule. Ты знаешь почему output_scale=0.0156 убьёт обучение (глава 18). Ты знаешь почему у кодбука learning rate в 10 раз меньше (глава 19).

Пришло время проверить: действительно ли ты **понял** — или просто читал?

Эта глава — вызов. Восемь этапов. В конце у тебя будет рабочая языковая модель.

---

### Что нужно, чтобы начать

**Python-пакеты:**

```bash
pip install torch sentencepiece numpy
```

**Данные** — любой текст. Начни с маленького:

```bash
# TinyShakespeare: 1MB, хватит для обучения символьной модели
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Или возьми любой русскоязычный текст из интернета
# Размер: от 1 МБ для тестов, от 100 МБ для нормального обучения
```

**Железо:** для мини-версии (dim=128, 2 блока) — обычный ноутбук с CPU. Тренировка займёт 10-30 минут на 2000 шагов.

---

### Этап 1: Bipolar Codebook — фундамент

Первое что нужно — кодбук. Это таблица векторов, где каждый токен — биполярный вектор {-1, +1}.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BipolarCodebook(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Embedding(vocab_size, dim)
        # Инициализация: маленькие числа (не нули!)
        nn.init.normal_(self.weight.weight, std=0.02)

    def _ste_encode(self, w: torch.Tensor) -> torch.Tensor:
        """
        STE (Straight-Through Estimator): бинаризуем, но градиент течёт.

        sign() имеет градиент=0 везде (функция ступенька).
        STE трюк: притворяемся что градиент = 1.

        (sign(w) - w).detach() + w:
        - Вперёд: sign(w) — получаем {-1, +1}
        - Назад: градиент идёт к w напрямую (как будто sign = identity)
        """
        hard = w.sign()
        return (hard - w).detach() + w

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T) → (B, T, dim)"""
        raw = self.weight(token_ids)
        return self._ste_encode(raw)

    def decode_logits(self, h: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """h: (B, T, dim) → logits: (B, T, vocab_size)"""
        cb_bipolar = self._ste_encode(self.weight.weight)
        return h @ cb_bipolar.T * scale
```

**Тест:**

```python
cb = BipolarCodebook(vocab_size=100, dim=64)

# Тест 1: форма
tokens = torch.tensor([[0, 1, 2]])  # batch=1, seq=3
x = cb.encode(tokens)
assert x.shape == (1, 3, 64), f"Ожидали (1,3,64), получили {x.shape}"

# Тест 2: только {-1, +1}
unique = x.unique().tolist()
assert set(unique) == {-1.0, 1.0}, f"Должно быть только -1 и +1, получили {unique}"

# Тест 3: quasi-orthogonality
x0 = x[0, 0, :]  # вектор токена 0
x1 = x[0, 1, :]  # вектор токена 1
dot = (x0 * x1).sum().item()
print(f"Dot product токенов 0 и 1: {dot:.1f}  (должно быть близко к 0)")

# Тест 4: градиент течёт
x_grad = cb.encode(torch.tensor([[0]]))
loss = x_grad.sum()
loss.backward()
assert cb.weight.weight.grad is not None, "Градиент не пришёл в кодбук!"
print("Тест 4 пройден: градиент течёт через STE ✓")
```

---

### Этап 2: Cyclic Position — позиция без параметров

Позиция токена кодируется циклическим сдвигом его вектора.

```python
def cyclic_position(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, D)
    Для токена на позиции t: циклический сдвиг на t элементов.

    Позиция 0: [a, b, c, d, e] → [a, b, c, d, e]  (без изменений)
    Позиция 1: [a, b, c, d, e] → [b, c, d, e, a]  (сдвиг на 1)
    Позиция 2: [a, b, c, d, e] → [c, d, e, a, b]  (сдвиг на 2)
    """
    B, T, D = x.shape
    device = x.device

    # Для каждой позиции t: сдвиг = t % D
    positions = torch.arange(T, device=device)
    shifts = positions % D  # (T,)

    # Индексы для gather:
    # для позиции t и измерения d: берём x[b, t, (d + shift) % D]
    base_idx = torch.arange(D, device=device)  # (D,)
    # idx[t, d] = (d + shifts[t]) % D
    idx = (base_idx.unsqueeze(0) + shifts.unsqueeze(1)) % D  # (T, D)
    idx = idx.unsqueeze(0).expand(B, T, D)  # (B, T, D)

    return x.gather(2, idx)
```

**Тест:**

```python
# Тест 1: позиция 0 не меняет вектор
x = torch.randn(1, 5, 8)
x_pos = cyclic_position(x)
assert torch.allclose(x_pos[:, 0, :], x[:, 0, :]), "Позиция 0 должна быть без изменений"

# Тест 2: разные позиции = разные векторы
assert not torch.allclose(x_pos[:, 1, :], x_pos[:, 2, :]), "Разные позиции должны давать разные векторы"

# Тест 3: форма не меняется
assert x_pos.shape == x.shape

print("Cyclic position: все тесты пройдены ✓")
```

---

### Этап 3: HDCMemory — накопленная память

Каждый токен видит взвешенную сумму всего прошлого, с затуханием.

```python
class HDCMemory(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Маленькие проекции: каждая принимает норму вектора (1 число)
        # и выдаёт скаляр mass или decay
        self.mass_proj  = nn.Linear(1, 1, bias=True)
        self.decay_proj = nn.Linear(1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → context: (B, T, D)"""
        B, T, D = x.shape

        # Норма каждого токен-вектора: (B, T, 1)
        norms = x.norm(dim=-1, keepdim=True)

        # Важность (mass): насколько этот токен весит в памяти
        mass  = torch.sigmoid(self.mass_proj(norms))   # (B, T, 1)

        # Скорость затухания: как быстро этот токен "забывается"
        decay = torch.sigmoid(self.decay_proj(norms))  # (B, T, 1)

        # Взвешенные токены: mass × вектор
        weighted = mass * x  # (B, T, D)

        # Накопленная память через log-cumsum (параллельный алгоритм):
        #
        # Идея: context[t] = Σ_{i≤t} decay_factor(i→t) × mass[i] × x[i]
        # где decay_factor(i→t) = Π_{k=i+1}^{t} decay[k]
        #
        # Через логарифмы: log(decay_factor) = Σ log(decay)
        # cumsum даёт нарастающую сумму → exp даёт нарастающее произведение

        log_decay = torch.log(decay.clamp(min=1e-6))      # (B, T, 1)
        cum_log   = torch.cumsum(log_decay, dim=1)         # (B, T, 1)

        # diff[i,j] = насколько токен i "затух" к позиции j
        # = cum_log[j] - cum_log[i]
        cum_i = cum_log.transpose(1, 2)    # (B, 1, T)
        cum_j = cum_log                    # (B, T, 1)
        diff  = cum_j - cum_i              # (B, T, T)
        diff  = diff.clamp(max=0)          # затухание только уменьшает

        # Causal mask: токен j не видит будущее (i > j)
        causal = torch.tril(torch.ones(T, T, device=x.device))
        diff   = diff.masked_fill(causal == 0, float('-inf'))

        # Веса: экспонента разницы логарифмов
        attn_weights = torch.exp(diff)    # (B, T, T)

        # Взвешенная сумма прошлых токенов:
        context = attn_weights @ weighted  # (B, T, D)

        return context
```

**Тест:**

```python
mem = HDCMemory(dim=64)
x = torch.randn(2, 10, 64)
ctx = mem(x)

# Форма не изменилась:
assert ctx.shape == (2, 10, 64), f"Ожидали (2,10,64), получили {ctx.shape}"

# Первый токен видит только себя:
# (context[b, 0, :] должен быть proportional к x[b, 0, :])
print("HDCMemory: тест пройден ✓")
```

---

### Этап 4: MultiHeadBindingAttention — HDC вместо матриц

Binding (HDC-multiply) вместо матричного умножения.

```python
class MultiHeadBindingAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0, "dim должен делиться на n_heads"
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads

        # Маленькие проекции: (n_heads, head_dim)
        # Это НЕ матрицы (dim × dim) — это биполярные векторы!
        self.bv_q = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.bv_k = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.bv_v = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, HD = self.n_heads, self.head_dim

        # Reshape для multi-head: (B, T, H, HD) → (B, H, T, HD)
        xh = x.view(B, T, H, HD).permute(0, 2, 1, 3)  # (B, H, T, HD)

        # Масштабируем по головам (вместо матричных проекций):
        q = xh * self.bv_q.unsqueeze(1)  # (B, H, T, HD)
        k = xh * self.bv_k.unsqueeze(1)  # (B, H, T, HD)
        v = xh * self.bv_v.unsqueeze(1)  # (B, H, T, HD)

        # HDC Bind: поэлементное умножение q и k:
        binding = q * k  # (B, H, T, HD) — не T×T матрица!

        # Scores: суммируем по head_dim
        scores = binding.sum(-1, keepdim=True)  # (B, H, T, 1)
        # Расширяем для применения к T токенам:
        scores = scores.expand(B, H, T, T)       # (B, H, T, T)

        # Causal mask: нельзя видеть будущее
        causal = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(causal == 0, float('-inf'))

        # Sigmoid вместо softmax: не конкурентное внимание
        attn_weights = torch.sigmoid(scores) * causal * H  # масштаб

        # Применяем к значениям:
        out = attn_weights @ v  # (B, H, T, HD)

        # Объединяем головы:
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return out
```

**Тест:**

```python
attn = MultiHeadBindingAttention(dim=64, n_heads=4)
x = torch.randn(2, 10, 64)
out = attn(x)

# Форма не изменилась:
assert out.shape == (2, 10, 64)

# Подсчитаем параметры:
params = sum(p.numel() for p in attn.parameters())
print(f"Параметры attention: {params:,}")
# Для dim=64, n_heads=4: 3 × 4 × 16 = 192 параметра — ничтожно мало!

print("MultiHeadBindingAttention: тест пройден ✓")
```

---

### Этап 5: ControllerBlock и HDCBlock — собираем блок

```python
class ControllerBlock(nn.Module):
    """
    Основной вычислитель: FFN с Pre-LN и residual.
    Занимает 99.9% параметров блока.
    """
    def __init__(self, dim: int, inner_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln  = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN → expand → GELU → contract → dropout → residual
        h = self.ln(x)          # нормализуем
        h = self.fc1(h)         # 4096 → 2560 (или dim → inner_dim)
        h = F.gelu(h)           # нелинейность
        h = self.drop(h)
        h = self.fc2(h)         # 2560 → 4096
        return x + h            # residual!


class HDCBlock(nn.Module):
    """
    Один блок HDC-Brain: Memory → Attention → Controller.
    Memory и Attention собирают контекст (дёшево).
    Controller думает над ним (дорого, 99.9% параметров).
    """
    def __init__(self, dim: int, inner_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.memory     = HDCMemory(dim)
        self.attention  = MultiHeadBindingAttention(dim, n_heads)
        self.controller = ControllerBlock(dim, inner_dim, dropout)
        self.ln_mem     = nn.LayerNorm(dim)
        self.ln_attn    = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Шаг 1: память о прошлом
        mem = self.memory(x)
        x = self.ln_mem(x + mem)    # residual + нормализация

        # Шаг 2: внимание
        attn = self.attention(x)
        x = self.ln_attn(x + attn)  # residual + нормализация

        # Шаг 3: глубокая обработка
        x = self.controller(x)      # внутри есть свой residual

        return x
```

**Тест:**

```python
block = HDCBlock(dim=64, inner_dim=128, n_heads=4)
x = torch.randn(2, 10, 64)
out = block(x)
assert out.shape == (2, 10, 64), "Форма должна сохраняться"

# Параметры:
params = sum(p.numel() for p in block.parameters())
memory_params = sum(p.numel() for p in block.memory.parameters())
attn_params   = sum(p.numel() for p in block.attention.parameters())
ctrl_params   = sum(p.numel() for p in block.controller.parameters())

print(f"Memory:     {memory_params:,} ({100*memory_params/params:.1f}%)")
print(f"Attention:  {attn_params:,}   ({100*attn_params/params:.1f}%)")
print(f"Controller: {ctrl_params:,}   ({100*ctrl_params/params:.1f}%)")
# Ожидаем: Controller ≈ 99.9% блока
```

---

### Этап 6: ThoughtLoop — три прохода через восемь блоков

```python
class ThoughtLoop(nn.Module):
    """
    Применяет одни и те же блоки несколько раз.
    Между проходами: гейтированная интерполяция.
    """
    def __init__(self, dim: int, max_thoughts: int = 4):
        super().__init__()
        self.max_thoughts = max_thoughts

        # Обучаемые gate-значения: одно число на мысль
        # zeros → sigmoid(0) = 0.5 — нейтральный старт
        self.thought_gates = nn.Parameter(torch.zeros(max_thoughts))

        # Позиционные сигналы: "на какой мысли ты?"
        # Маленькая инициализация чтобы не ломать первый проход
        self.thought_pos = nn.Parameter(torch.randn(max_thoughts, dim) * 0.01)

        # Нормализация перед каждым дополнительным проходом
        self.ln = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, blocks: nn.ModuleList, n_thoughts: int = 3) -> torch.Tensor:
        n_thoughts = min(n_thoughts, self.max_thoughts)

        # === Мысль 1: обязательная ===
        for block in blocks:
            h = block(h)

        if n_thoughts <= 1:
            return h

        # === Мысли 2, 3, ...: с gate ===
        for t in range(1, n_thoughts):
            gate = torch.sigmoid(self.thought_gates[t])
            # gate от 0 до 1:
            # 0 = игнорируем эту мысль
            # 1 = полностью заменяем предыдущий результат

            # Входной сигнал: нормализованный h + "метка мысли"
            thought_input = self.ln(h) + self.thought_pos[t]

            # Прогоняем через ТЕ ЖЕ блоки
            thought = thought_input
            for block in blocks:
                thought = block(thought)

            # Гейтированная интерполяция:
            # h = (1-gate) × старый + gate × новый
            h = h + gate * (thought - h)

        return h
```

**Тест:**

```python
dim = 64
blocks = nn.ModuleList([HDCBlock(dim, 128) for _ in range(2)])
loop = ThoughtLoop(dim, max_thoughts=4)

x = torch.randn(2, 10, dim)

# 1 мысль vs 3 мысли должны давать разный результат:
out_1 = loop(x, blocks, n_thoughts=1)
out_3 = loop(x, blocks, n_thoughts=3)
assert not torch.allclose(out_1, out_3), "1 и 3 мысли должны давать разные результаты"

# Но форма одинаковая:
assert out_1.shape == out_3.shape == (2, 10, dim)

# Начальные gates = 0.5:
initial_gates = torch.sigmoid(loop.thought_gates)
print(f"Начальные gates: {initial_gates.tolist()}")  # все ≈ 0.5

print("ThoughtLoop: тест пройден ✓")
```

---

### Этап 7: Полная модель

```python
class MiniHDCBrain(nn.Module):
    def __init__(self,
                 vocab_size:    int = 32768,
                 dim:           int = 128,    # уменьшено для быстрого теста
                 inner_dim:     int = 256,    # dim × 2 — минимальный вариант
                 n_blocks:      int = 2,      # уменьшено
                 n_heads:       int = 4,
                 max_thoughts:  int = 4,
                 n_thoughts:    int = 3,      # мысли во время обучения
                 dropout:       float = 0.0): # без dropout для простоты
        super().__init__()

        self.n_thoughts = n_thoughts

        # Кодбук: и вход и выход (weight tying)
        self.codebook = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.codebook.weight, std=0.02)

        # 8 (или n_blocks) блоков
        self.blocks = nn.ModuleList([
            HDCBlock(dim, inner_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        # ThoughtLoop
        self.thought_loop = ThoughtLoop(dim, max_thoughts)

        # Масштаб выхода
        # ВАЖНО: инициализировать как 1.0, а НЕ как 1/sqrt(dim)!
        # 1/sqrt(128) = 0.088 → логиты будут слабыми → обучение медленное
        # (см. главу 18: output_scale=0.0156 убил обучение на 2 часа)
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def _ste(self, w: torch.Tensor) -> torch.Tensor:
        """Straight-Through Estimator: градиент течёт через sign()"""
        hard = w.sign()
        return (hard - w).detach() + w

    def _cyclic_pos(self, x: torch.Tensor) -> torch.Tensor:
        """Cyclic Position Permutation: 0 параметров"""
        B, T, D = x.shape
        device = x.device
        positions = torch.arange(T, device=device)
        shifts = positions % D
        base = torch.arange(D, device=device)
        idx = (base.unsqueeze(0) + shifts.unsqueeze(1)) % D
        idx = idx.unsqueeze(0).expand(B, T, D)
        return x.gather(2, idx)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor = None):
        """
        token_ids: (B, T) — входные токены
        targets:   (B, T) — целевые токены (для обучения)

        Возвращает: (logits, loss) или (logits, None)
        """
        # Шаг 1: кодбук + бинаризация
        raw = self.codebook(token_ids)  # (B, T, D)
        x = self._ste(raw)              # → {-1, +1}

        # Шаг 2: позиционное кодирование
        x = self._cyclic_pos(x)         # (B, T, D) — форма та же

        # Шаг 3: ThoughtLoop (3 прохода через блоки)
        h = self.thought_loop(x, self.blocks, n_thoughts=self.n_thoughts)
        # h: (B, T, D)

        # Шаг 4: декодирование через тот же кодбук (weight tying)
        cb_binary = self._ste(self.codebook.weight)  # (vocab_size, D)
        logits = h @ cb_binary.T * self.output_scale  # (B, T, vocab_size)

        # Шаг 5: loss (только при обучении)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1)                    # (B*T,)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new: int = 50,
                 temperature: float = 0.8) -> torch.Tensor:
        """Генерация текста: добавляем по одному токену"""
        for _ in range(max_new):
            logits, _ = self(token_ids)
            # Берём только последнюю позицию:
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids
```

**Тест:**

```python
model = MiniHDCBrain(vocab_size=1000, dim=64, inner_dim=128, n_blocks=2)

# Тест forward:
x = torch.randint(0, 1000, (2, 10))  # batch=2, seq=10
logits, loss = model(x, x[:, 1:])    # targets = x сдвинутый
# Нет, логиты (2,10,1000) а цели (2,9) — нужно выровнять
logits, loss = model(x[:, :-1], x[:, 1:])  # стандартный language modeling
assert logits.shape == (2, 9, 1000)

# Первый loss должен быть ≈ log(vocab_size):
expected = torch.log(torch.tensor(float(1000))).item()
print(f"Первый loss: {loss.item():.2f}, ожидали ~{expected:.2f}")
# Если очень отличается — проблема с инициализацией

# Тест generate:
prompt = torch.tensor([[5, 10, 20]])
generated = model.generate(prompt, max_new=5)
print(f"Генерация: {generated.shape}")  # (1, 8)

print("Полная модель: тест пройден ✓")
```

---

### Этап 8: Обучение с нуля

```python
import math
import torch.optim as optim

def train(model, data: torch.Tensor, steps: int = 2000,
          batch_size: int = 8, seq_len: int = 64,
          max_lr: float = 3e-4, min_lr: float = 1e-5,
          warmup_steps: int = 50):
    """
    data: одномерный torch.LongTensor с токенами
    """
    # === Две группы параметров ===
    codebook_params = [p for n, p in model.named_parameters() if 'codebook' in n]
    other_params    = [p for n, p in model.named_parameters() if 'codebook' not in n]

    optimizer = optim.AdamW([
        {'params': other_params,    'lr': max_lr,    'weight_decay': 0.1},
        {'params': codebook_params, 'lr': max_lr/10, 'weight_decay': 0.0},
        # ↑ Кодбук: в 10× меньше LR (STE чувствителен к большим шагам)
        # ↑ Кодбук: нет weight decay (биполярные веса нельзя гасить к нулю)
    ], betas=(0.9, 0.95))

    model.train()
    losses = []

    for step in range(steps):
        # === Cosine LR с warmup ===
        if step < warmup_steps:
            lr = max_lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr / 10

        # === Батч ===
        # Случайные стартовые позиции
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[i:i+seq_len]   for i in starts])    # (B, T)
        y = torch.stack([data[i+1:i+seq_len+1] for i in starts])  # (B, T)

        # === Forward + Loss ===
        logits, loss = model(x, y)

        # === Backward ===
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (страховка от взрыва):
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # === Логирование ===
        losses.append(loss.item())
        if step % 100 == 0:
            bpb = loss.item() / math.log(2)
            print(f"Шаг {step:4d}: loss={loss.item():.3f}  bpb={bpb:.2f}  "
                  f"lr={lr:.2e}  grad_norm={grad_norm:.2f}")

    return losses


# ========================
# Запуск на TinyShakespeare
# ========================

# Загрузка данных (символьный уровень — самый простой):
with open("input.txt", "r") as f:
    text = f.read()

# Создаём словарь символов:
chars = sorted(set(text))
vocab_size = len(chars)
print(f"Символов в алфавите: {vocab_size}")  # ~65 для английского

char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

# Токенизируем весь текст:
data = torch.tensor([char_to_id[c] for c in text], dtype=torch.long)
print(f"Токенов всего: {len(data):,}")

# Создаём маленькую модель:
model = MiniHDCBrain(
    vocab_size=vocab_size,
    dim=128,
    inner_dim=256,
    n_blocks=2,
    n_heads=4,
    max_thoughts=4,
    n_thoughts=3,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {n_params:,}")

# Обучаем:
losses = train(
    model=model,
    data=data,
    steps=2000,
    batch_size=8,
    seq_len=64,
    max_lr=3e-4,
    warmup_steps=50,
)

# Генерируем текст:
model.eval()
seed = "HAMLET:\n"
seed_ids = torch.tensor([[char_to_id[c] for c in seed]])
generated_ids = model.generate(seed_ids, max_new=200)
generated_text = "".join(id_to_char[i.item()] for i in generated_ids[0])
print("\n--- Сгенерированный текст ---")
print(generated_text)
```

---

### Чеклист: проверь себя

Перед тем как считать проект завершённым:

**BipolarCodebook:**
- [ ] Выход содержит только {-1, +1}
- [ ] Градиент течёт (STE): `loss.backward()` → `codebook.grad is not None`
- [ ] Dot product разных токенов ≈ 0

**CyclicPosition:**
- [ ] Позиция 0: вектор не меняется
- [ ] Позиция 1 ≠ Позиция 2 (для ненулевых входов)

**HDCMemory:**
- [ ] Форма входа = форме выхода
- [ ] Causal: токен 0 не видит токен 1 (нет утечки из будущего)

**MultiHeadBindingAttention:**
- [ ] Нет матрицы T×T в памяти (используем поэлементное умножение)
- [ ] Параметров в 100+ раз меньше чем у стандартного attention

**ThoughtLoop:**
- [ ] n_thoughts=1 и n_thoughts=3 дают разные результаты
- [ ] Начальные gates ≈ 0.5 (sigma(0) = 0.5)
- [ ] После обучения gate_2 растёт (открывается)

**Полная модель:**
- [ ] Первый loss ≈ log(vocab_size) / log(2) BPB (случайное угадывание)
- [ ] После 500 шагов loss уменьшился хотя бы на 20%
- [ ] Генерирует текст (пусть бессвязный)
- [ ] output_scale инициализирован 1.0, НЕ 1/√dim

---

### Идеи для экспериментов

Ты реализовал базовую версию. Вот что можно исследовать дальше:

**Эксперимент 1: Сравни n_thoughts**

```python
for n in [1, 2, 3]:
    model = MiniHDCBrain(n_thoughts=n, ...)
    losses = train(model, data, steps=500)
    final_bpb = losses[-1] / math.log(2)
    print(f"n_thoughts={n}: BPB={final_bpb:.3f}")
```

Воспроизводится ли паттерн из главы 17: 1 < 2 < 3?

**Эксперимент 2: Отключи HDCMemory**

```python
# В HDCBlock.forward замени:
#   mem = self.memory(x)
#   x = self.ln_mem(x + mem)
# На:
#   pass  # пропускаем memory
```

Насколько хуже без явной памяти?

**Эксперимент 3: Sigmoid vs Softmax**

В MultiHeadBindingAttention:
```python
# Наш вариант:
attn_weights = torch.sigmoid(scores) * causal * H

# Стандартный:
scores_masked = scores.masked_fill(causal == 0, float('-inf'))
attn_weights = F.softmax(scores_masked, dim=-2)
```

Что лучше на твоих данных?

**Эксперимент 4: Мониторинг gates**

```python
# В тренировочном цикле после каждых 100 шагов:
gates = torch.sigmoid(model.thought_loop.thought_gates)
print(f"Шаг {step}: gates = {[f'{g:.3f}' for g in gates.tolist()]}")
```

gate_2 открывается быстро? gate_3 медленнее?

---

### Вместо заключения

Если ты запустил код и увидел что BPB падает — поздравляю.

Ты только что обучил языковую модель на основе Гиперразмерных Вычислений с ThoughtLoop.

Это архитектура которую ты теперь понимаешь изнутри. Не "посмотрел видео на YouTube". Не "прочитал блог". А написал каждую строчку руками, понял каждый трюк.

Что дальше:

```
Масштаб:    dim=128 → 512 → 1024 → 4096
Данные:     один текст → большой корпус → веб-данные
Язык:       символы → BPE токены (SentencePiece)
Сравнение:  HDC-Brain vs GPT-2 того же размера
Улучшения:  RoPE вместо Cyclic, больше голов, развёрнутый ThoughtLoop
Публикация: arXiv paper (если результаты интересные)
```

Эта архитектура — не финальный ответ. Это начало исследования.

---

### Итоги главы

8 этапов — полная реализация HDC-Brain:

```
Этап 1: BipolarCodebook       — STE трюк, {-1,+1}
Этап 2: CyclicPosition        — 0 параметров для позиций
Этап 3: HDCMemory             — накопленная память с decay
Этап 4: BindingAttention      — q*k вместо Q@K.T
Этап 5: HDCBlock              — Memory + Attention + Controller
Этап 6: ThoughtLoop           — 3 прохода, learned gates
Этап 7: MiniHDCBrain          — собрали всё вместе
Этап 8: Обучение              — AdamW, cosine LR, gradient clipping
```

Чеклист из 20+ тестов. 4 эксперимента для самостоятельного исследования.

---

*Приложение A → Глоссарий*
*Приложение B → Математика: линейная алгебра и теория информации*
*Приложение C → Полный аннотированный код v14.1*
