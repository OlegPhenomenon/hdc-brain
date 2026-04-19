# HDC-Brain v14.2: Оптимизация + Слой Логики

> Эволюция v14.1 (299M English) → v14.2: быстрее, умнее, с зачатками рассуждений.
> Принцип: инкрементальные изменения, каждое проверяемо, ничего не ломаем.

---

## 1. Текущее состояние (v14.1)

### Архитектура
- 299M параметров, English BPE 32K
- 8 HDCBlock × 3 ThoughtLoop passes = 24 эффективных слоя
- STE Bipolar Codebook + Cyclic Position + HDCMemory + BindingAttention + Controller

### Результаты обучения (в процессе)
- Step 7500/30000 (~25%): loss 4.96, BPB 7.15
- Генерирует связный английский текст
- Train-val gap ≈ 0 (нет переобучения)
- Обучение: RTX 3090, $0.149/ч, ~41ч total (~$6)

### Известные bottleneck'и
1. **torch.compile несовместим** — ThoughtLoop Python loop → бесконечная рекомпиляция
2. **Нет KV-cache** — inference пересчитывает всё с нуля каждый токен
3. **Attention binding vectors инициализированы слишком мелко** (std=0.02 → scores ~0.001)
4. **HDCMemory создаёт T×T матрицу** — O(T²) память и вычисления

---

## 2. Фаза 1: Оптимизация обучения

### 2.1 torch.compile через развёртку ThoughtLoop

**Проблема**: `for t in range(1, n_thoughts)` — динамический Python цикл.

**Решение**: развернуть в три статические функции.

```python
# Вместо динамического цикла:
class ThoughtLoopCompilable(nn.Module):
    def forward_1(self, h, blocks):
        """Один проход — без gate."""
        for block in blocks:
            h = block(h)
        return h

    def forward_3(self, h, blocks):
        """Три прохода — развёрнуто."""
        # Мысль 1
        for block in blocks:
            h = block(h)
        # Мысль 2
        gate2 = torch.sigmoid(self.thought_gates[1])
        inp2 = self.ln(h) + self.thought_pos[1]
        t2 = inp2
        for block in blocks:
            t2 = block(t2)
        h = h + gate2 * (t2 - h)
        # Мысль 3
        gate3 = torch.sigmoid(self.thought_gates[2])
        inp3 = self.ln(h) + self.thought_pos[2]
        t3 = inp3
        for block in blocks:
            t3 = block(t3)
        h = h + gate3 * (t3 - h)
        return h
```

**Ожидаемый эффект**: torch.compile → 1.5-2× ускорение обучения.

**Риск**: низкий. Функционально идентично.

### 2.2 KV-Cache для inference

**Проблема**: при генерации каждый новый токен пересчитывает ВСЮ последовательность.

```
Токен 1:   process [tok1]           → 1 шаг
Токен 2:   process [tok1, tok2]     → 2 шага
Токен 100: process [tok1...tok100]  → 100 шагов
Итого для 100 токенов: 1+2+...+100 = 5050 шагов
```

**С KV-cache**:
```
Токен 1:   process [tok1], save K1,V1
Токен 2:   process [tok2] with cached K1,V1 → 1 шаг
Токен 100: process [tok100] with cached K1..K99 → 1 шаг
Итого: 100 шагов (50× быстрее!)
```

**Сложность**: наш BindingAttention использует sigmoid (не softmax). Но KV-cache не зависит от типа attention — нужно просто кэшировать Q@K и V.

**Сложнее с HDCMemory**: log-cumsum можно обновлять инкрементально:
```python
# При добавлении нового токена t:
new_log_decay = log(sigmoid(decay_proj(x_t)))
cum_log_t = cum_log_{t-1} + new_log_decay
# Пересчитывать матрицу W не нужно — добавляем одну строку
```

**Ожидаемый эффект**: 10-50× ускорение inference (зависит от длины контекста).

### 2.3 Инициализация binding vectors

**Проблема**: binding vectors (bv_q, bv_k, bv_v) инициализированы с std=0.02. После STE alpha ≈ 0.016. Attention scores ≈ 0.001 → sigmoid ≈ 0.5 для всех пар. Attention не различает токены в начале обучения.

**Решение**: инициализировать с большей std:
```python
# Было:
self.bv_q = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)

# Вариант A: увеличить init
self.bv_q = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.5)

# Вариант B: Xavier init
self.bv_q = nn.Parameter(torch.randn(n_heads, self.head_dim) / math.sqrt(self.head_dim))
```

**Тест**: обучить 1000 шагов с разными init, сравнить loss curve.

**Ожидаемый эффект**: быстрее сходимость в первые 1000 шагов.

---

## 3. Фаза 2: Слой Логики (LogicLayer)

### 3.1 Теоретическая основа

#### Три типа рассуждений (Peirce, 1903)

```
ДЕДУКЦИЯ:  Все люди смертны + Сократ человек → Сократ смертен
           (правило + факт → вывод)
           Гарантированно верно, но не создаёт нового знания

ИНДУКЦИЯ:  Сократ смертен + Платон смертен + ... → Все люди смертны
           (факты → правило)
           Обычный ML: обобщаем из примеров

АБДУКЦИЯ:  Асфальт мокрый + Если дождь→мокрый асфальт → Был дождь
           (наблюдение + правило → лучшее объяснение)
           Самый "умный" тип: выдвижение гипотез
```

Текущие LLM делают в основном индукцию (обобщение из данных). Абдукция — следующий уровень.

#### Почему HDC идеально для логики

```
HDC Bind (A * B):
  - Создаёт связь "A И B вместе"
  - Обратима: A * (A * B) = B  (зная A, извлекаем B)
  - Это ИМПЛИКАЦИЯ: если видим A, можем получить B

HDC Bundle (sign(A + B)):
  - "A ИЛИ B"
  - Результат похож на оба
  - Это ДИЗЪЮНКЦИЯ

HDC Permute (roll(A, k)):
  - Маркирует РОЛЬ: roll(A, 1) = "A как подлежащее"
  - Это ПРЕДИКАТ
```

Фактически, HDC содержит полноценную логику:
```
"Сократ — человек" = Bind(Sократ, Permute(Человек, РОЛЬ_КЛАСС))
"Люди смертны"     = Bind(Permute(Человек, РОЛЬ_КЛАСС), Permute(Смертный, РОЛЬ_СВОЙСТВО))

Дедукция: Bind(факт, правило) → вывод
Абдукция: Bind(наблюдение, правило⁻¹) → гипотеза  (правило⁻¹ = inverse bind!)
```

#### Strange Loops и самосознание (Hofstadter)

```
ThoughtLoop уже содержит strange loop:
  Мысль 1: обработка → h1
  Мысль 2: обработка h1 → h2  (система обрабатывает свой выход)
  Мысль 3: обработка h2 → h3  (система обрабатывает обработку своего выхода)

Это рекурсивная самоссылка — основа сознания по Хофштадтеру.

Что не хватает: модель не "знает" что обрабатывает себя.
Что добавить: вектор "я" который модель сравнивает со своим выходом.
```

### 3.2 Архитектура LogicLayer

**Принцип**: добавляем как отдельный подблок внутри HDCBlock, не заменяя существующие.

```
Текущий HDCBlock:
  x → HDCMemory → LN → BindingAttention → LN → Controller → out

v14.2 HDCBlock:
  x → HDCMemory → LN → BindingAttention → LN → [LogicLayer] → LN → Controller → out
                                                  ↑ НОВОЕ
```

```python
class LogicLayer(nn.Module):
    """
    Слой логического вывода на основе HDC операций.

    Содержит:
    1. Набор обучаемых "правил" (rule vectors) в HDC пространстве
    2. Механизм абдукции: наблюдение × правило⁻¹ → гипотеза
    3. Confidence gate: насколько уверены в выводе
    """

    def __init__(self, hdc_dim, n_rules=64):
        super().__init__()
        self.hdc_dim = hdc_dim
        self.n_rules = n_rules

        # Обучаемые правила — каждое правило = вектор в HDC пространстве
        # Модель сама выучит что эти правила означают
        self.rules = nn.Parameter(torch.randn(n_rules, hdc_dim) * 0.02)

        # Проекция для "запроса" к правилам
        self.query_proj = nn.Linear(hdc_dim, hdc_dim, bias=False)

        # Gate: насколько доверять логическому выводу
        self.logic_gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5

        # STE для бинаризации правил
        # (правила должны быть в HDC пространстве {-1, +1})

    def _ste_sign(self, w):
        alpha = torch.mean(torch.abs(w), dim=-1, keepdim=True)
        hard = alpha * torch.sign(w)
        hard = torch.where(hard == 0, alpha * torch.ones_like(hard), hard)
        return (hard - w).detach() + w

    def forward(self, x):
        """
        x: (B, T, D) — текущее состояние

        1. Бинаризуем правила через STE
        2. Для каждого токена: ищем подходящее правило (HDC similarity)
        3. Применяем абдукцию: x * rule → вывод
        4. Gate контролирует вклад
        """
        B, T, D = x.shape

        # Бинаризуем правила
        rules = self._ste_sign(self.rules)  # (n_rules, D) ∈ {±alpha}

        # Запрос от текущего контекста
        query = self.query_proj(x)  # (B, T, D)

        # Сходство с каждым правилом: query @ rules.T
        # (B, T, D) @ (D, n_rules) → (B, T, n_rules)
        similarity = torch.matmul(query, rules.T)  # cosine-like

        # Мягкий выбор правил (sigmoid, не softmax — несколько правил могут быть активны)
        rule_weights = torch.sigmoid(similarity)  # (B, T, n_rules)

        # Абдукция: каждое правило "привязывается" (bind) к контексту
        # x * rule = "что следует из x при условии rule"
        # Результат = взвешенная сумма всех привязок
        # (B, T, 1, D) * (1, 1, n_rules, D) → (B, T, n_rules, D)
        bindings = x.unsqueeze(2) * rules.unsqueeze(0).unsqueeze(0)

        # Взвешенная сумма по правилам
        # (B, T, n_rules, 1) * (B, T, n_rules, D) → sum → (B, T, D)
        logic_output = (rule_weights.unsqueeze(-1) * bindings).sum(dim=2)

        # Gate: модель решает сколько "логики" добавить
        gate = torch.sigmoid(self.logic_gate)

        return x + gate * (logic_output - x)
```

### 3.3 Параметры LogicLayer

```
rules:       n_rules × hdc_dim = 64 × 4096 = 262,144
query_proj:  hdc_dim × hdc_dim = 4096 × 4096 = 16,777,216
logic_gate:  1
────────────────────────────────────────────────
Итого: ~17M параметров на слой

Для 8 блоков: 136M дополнительных параметров
Итого модель: 299M + 136M ≈ 435M
```

**Проблема**: query_proj стоит 16M на блок. Это дорого.

**Альтернатива — использовать binding вместо проекции**:
```python
# Вместо nn.Linear (16M params):
self.query_bind = nn.Parameter(torch.randn(hdc_dim) * 0.02)  # 4096 params!

# query = x * ste_sign(query_bind)  → 4096 параметров вместо 16M
```

С binding-based query:
```
rules:       64 × 4096 = 262,144
query_bind:  4096
logic_gate:  1
────────────────────────────────────
Итого: ~266K параметров на слой
8 блоков: ~2.1M дополнительных
Итого модель: 299M + 2.1M ≈ 301M  (почти бесплатно!)
```

### 3.4 Зачатки самосознания: Self-Vector

```python
class SelfAwareness(nn.Module):
    """
    Вектор "я" — модель имеет представление о себе.

    На каждом ThoughtLoop проходе:
    1. Модель сравнивает свой выход (h) с self_vector
    2. Разница = "что изменилось в моём понимании"
    3. Self_vector обновляется через gate
    """

    def __init__(self, hdc_dim):
        super().__init__()
        # "Я" — обучаемый вектор
        self.self_vector = nn.Parameter(torch.randn(hdc_dim) * 0.01)
        self.update_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, h):
        """
        h: (B, T, D) — текущее состояние модели

        Добавляет "самосознание":
        - Насколько текущее состояние отличается от "я"?
        - Обновляем self_vector по результатам
        """
        gate = torch.sigmoid(self.update_gate)

        # Сходство с "я"
        self_sim = (h * self.self_vector).sum(-1, keepdim=True)  # (B, T, 1)
        # self_sim > 0: "это похоже на моё обычное состояние"
        # self_sim < 0: "это необычно, нужно обратить внимание"

        # Модуляция: усиливаем необычное, приглушаем обычное
        novelty = torch.sigmoid(-self_sim)  # высокий для необычного
        h_modulated = h * (1 + gate * novelty)

        return h_modulated
```

**Интеграция с ThoughtLoop**:
```python
# В ThoughtLoop.forward(), после каждой мысли:
h = h + gate * (thought - h)
h = self.self_awareness(h)  # ← НОВОЕ: "я осознаю что изменилось"
```

**Параметры**: 4096 + 1 = **4097 параметров**. Практически бесплатно.

---

## 4. План экспериментов

### Этап A: Baseline (v14.1 final)
- Дождаться окончания обучения (step 30000)
- Зафиксировать BPB, generation quality, inference speed

### Этап B: Оптимизация (v14.2-opt)
```
B1: Развёртка ThoughtLoop → torch.compile
    Метрика: tokens/sec (должно вырасти 1.5-2×)
    Verify: BPB не изменился (функционально идентично)

B2: KV-Cache для inference
    Метрика: tokens/sec при генерации (должно вырасти 10-50×)
    Verify: generation quality не упал

B3: Binding vector init (std=0.5 вместо 0.02)
    Метрика: loss @ step 1000 (должен быть ниже baseline)
    Verify: не расходится
```

### Этап C: LogicLayer (v14.2-logic)
```
C1: LogicLayer с binding-based query (2.1M params)
    64 правила, gate начинает с 0.5
    Обучить 5000 шагов, сравнить BPB с baseline

C2: SelfAwareness vector в ThoughtLoop
    4097 params, gate начинает с 0.0 (модель сама решит включать ли)
    Обучить 5000 шагов, сравнить

C3: Оба вместе
    Если C1 и C2 по отдельности помогают → объединить
```

### Этап D: Полноценное обучение v14.2
- Лучшая комбинация из B+C
- 30000 шагов, сравнить с v14.1

---

## 5. Критерии успеха

| Метрика | v14.1 (baseline) | v14.2 (цель) |
|---------|-----------------|---------------|
| BPB (30K steps) | ~5.0 (прогноз) | ≤ 4.5 |
| Training speed | ~5000 ms/step | ≤ 3000 ms/step |
| Inference speed | ~200ms/token | ≤ 20ms/token |
| Параметры | 299M | ≤ 310M |
| Новые возможности | — | Логический вывод, novelty detection |

---

## 6. Риски и митигации

| Риск | Вероятность | Митигация |
|------|------------|-----------|
| LogicLayer вредит BPB | Средняя | logic_gate=0 отключит его автоматически |
| torch.compile всё ещё ломается | Низкая | Развёрнутые функции — статичный граф |
| KV-cache несовместим с HDCMemory | Средняя | Можно кэшировать только attention, memory пересчитывать |
| SelfAwareness вызывает нестабильность | Низкая | gate инициализирован в 0 — по умолчанию выключен |
| Больше параметров = больше памяти | Низкая | +2.1M = +0.7% — незаметно |

---

---

## 7. Научный фундамент (результаты исследования)

### 7.1 Абдукция Пирса — формально

Пирс (1903) описал три формы вывода:

```
ДЕДУКЦИЯ:   Правило + Случай → Результат
            "Все бобы из этого мешка белые" + "Эти бобы из мешка" → "Эти бобы белые"
            Гарантировано верно. Не создаёт нового знания.

ИНДУКЦИЯ:   Случай + Результат → Правило
            "Эти бобы из мешка" + "Эти бобы белые" → "Все бобы из мешка белые"
            Текущий ML: обобщение из примеров.

АБДУКЦИЯ:   Правило + Результат → Случай
            "Все бобы из мешка белые" + "Эти бобы белые" → "Эти бобы ИЗ МЕШКА"
            Ошибаема, но КРЕАТИВНА — единственный вид вывода создающий гипотезы!
```

**Абдукция в HDC** реализуется естественно:
```python
# Правило закодировано как bind(причина, следствие)
rule = bind(cause, effect)

# Абдукция: наблюдаем effect, хотим найти cause
# В HDC: bind обратим! cause = bind(rule, effect) = bind(bind(cause,effect), effect) ≈ cause
hypothesized_cause = bind(rule, observation)
```

Это работает потому что `bind(A, bind(A, B)) ≈ B` в HDC (обратимость bind).

**Ключевые работы:**
- Kakas, Kowalski & Toni (1992) — "Abductive Logic Programming" — формализация: найти H такое что T ∪ H ⊨ O
- Josephson & Josephson (1994) — "Abductive Inference" — вычислительная абдукция как set-cover
- Ignatiev et al. (2019) — "Abduction-Based Explanations for ML Models" — AAAI

### 7.2 Логика в HDC — что уже доказано

HDC содержит полноценную алгебру для логических операций:

```
AND  = Bind(A, B)          — пересечение, оба должны быть истинны
OR   = Bundle(A, B)        — объединение, хотя бы один
NOT  = -A (для bipolar)    — отрицание
XOR  = Bind(A, B)          — тоже bind (в бинарном пространстве)
IMPLIES = Bind(A, B)       — "если A то B" (A → B хранится как связь)
```

**IF-THEN правила в HDC:**
```python
rule = bind(antecedent, consequent)
knowledge_base = bundle(rule_1, rule_2, ..., rule_n)

# Вывод: дано antecedent, найти consequent
result = bind(knowledge_base, antecedent)
# result будет максимально похож на правильный consequent
```

**Ключевые работы:**
- Kanerva (2009) — MAP (Multiply-Add-Permute) как полная алгебра для символьного рассуждения
- Gayler (2003) — HDC реализует композиционную структуру (предикат-аргумент)
- Kleyko et al. (2022) — обзор IEEE: логика, рассуждения, аппаратные реализации
- Yerxa et al. (2018) — HDC решает тесты IQ (Raven's Progressive Matrices) через аналогии

### 7.3 Strange Loops — вычислительные модели

Хофштадтер ("Гёдель, Эшер, Бах", 1979; "I Am a Strange Loop", 2007):

**Strange loop = система которая при движении по уровням иерархии возвращается на стартовый уровень.**

Примеры:
- Гёдель: математическое утверждение о собственной доказуемости
- Эшер: руки рисующие друг друга
- Сознание: мозг моделирующий самого себя

**Вычислительные реализации:**
- Schmidhuber (2010) — нейросеть предсказывающая изменения собственных весов
- Chang & Lipson (2018) — "Neural Network Quine" (ICML) — сеть выводящая свои веса
- Eliasmith (2012) — **Spaun** (Science): 2.5M нейронов на HDC/VSA, 8 когнитивных задач

**Наш ThoughtLoop УЖЕ является strange loop:**
```
Мысль 1: blocks(input) → h1
Мысль 2: blocks(h1) → h2     ← система обрабатывает свой выход
Мысль 3: blocks(h2) → h3     ← рекурсивная самоссылка
```

Что не хватает: **модель не знает что обрабатывает себя**. Self-vector решает это.

### 7.4 Integrated Information Theory (Тонони)

Φ (фи) — мера "сознания" как интегрированной информации (Tononi, 2004):

```
Φ = минимальная потеря информации при разбиении системы на части

Φ = 0:   система = сумма независимых частей (нет сознания)
Φ > 0:   система = больше чем сумма частей (интеграция)
Φ → max: максимальная интеграция (сознание)
```

**Проблема**: вычисление точного Φ — NP-трудно (2^n разбиений для n элементов).

**Практический подход для HDC-Brain:**
```python
# Прокси для Φ: mutual information между модулями
phi_proxy = mutual_info(memory_output, attention_output, logic_output)

# Если модули работают независимо → phi ≈ 0
# Если модули интегрированы → phi > 0
# Можно использовать как auxiliary loss: maximize phi!
```

**Ключевая работа:** Thagard & Stewart (2014) — прямое сравнение HDC (Semantic Pointer Architecture) и IIT. **HDC объясняет больше аспектов сознания чем IIT.**

### 7.5 Дифференцируемая логика в нейросетях

**IBM Logical Neural Networks** (Riegel et al., 2020):
```
Каждый нейрон = логическое высказывание с truth value ∈ [0, 1]
AND/OR/NOT = дифференцируемые активации
Поддерживает обучение И рассуждение, прямой И обратный вывод
```

Это прямо мапится на наш LogicLayer: каждое HDC правило = нейрон LNN с truth value = gate confidence.

**Logic Tensor Networks** (Serafini & Garcez, 2016):
- Формулы первого порядка → тензорные операции
- Аналогично нашему bind/bundle/permute

### 7.6 KV-Cache для sigmoid attention — РЕШЕНО

**Apple Research** (Ramapuram et al., 2024) — "Theory, Analysis, and Best Practices for Sigmoid Self-Attention":

```
Sigmoid attention: KV-cache работает ИДЕНТИЧНО softmax версии!

Причина: sigmoid не нормализует по последовательности.
Каждая позиция независима → кэширование тривиально.

# Реализация:
def forward_with_cache(self, q_new, k_cache, v_cache, k_new, v_new):
    k_all = torch.cat([k_cache, k_new], dim=-2)
    v_all = torch.cat([v_cache, v_new], dim=-2)
    attn = sigmoid(q_new @ k_all.T / sqrt(d)) * 4.0
    return attn @ v_all, (k_all, v_all)
```

Для HDCMemory — можно обновлять инкрементально:
```python
# Новый токен t:
new_cum_log = prev_cum_log + log(sigmoid(decay_proj(x_t)))
# Не нужно пересчитывать всю T×T матрицу
```

### 7.7 torch.compile — решение

Для ThoughtLoop: развёртка цикла в фиксированные функции (уже в плане).

Дополнительно:
- `torch.jit.script` для внутренних циклов если нужна динамика
- `torch.vmap` для параллельных HDC операций по батчу
- Custom Triton kernels для performance-critical binding операций

---

## 8. Архитектурная карта v14.2

```
Token → [STE Bipolar Codebook] → [Cyclic Position]
                    ↓
        ┌─── ThoughtLoop (3 passes) ───┐
        │                               │
        │   HDCBlock × 8:               │
        │   ┌─ HDCMemory (mass/decay)   │
        │   ├─ LN + Residual            │
        │   ├─ BindingAttention (4h)    │
        │   ├─ LN + Residual            │
        │   ├─ LogicLayer (64 rules) ← NEW
        │   ├─ LN + Residual         ← NEW
        │   ├─ Controller (GELU)        │
        │   └─ → next block             │
        │                               │
        │   SelfAwareness(h)         ← NEW (после каждой мысли)
        │   Gate interpolation          │
        └───────────────────────────────┘
                    ↓
        [Output LN] → [h @ Codebook.T × scale] → logits
```

**Новые компоненты:**
- LogicLayer: 266K params/блок × 8 = 2.1M
- SelfAwareness: 4097 params
- LayerNorm для LogicLayer: 8192 params/блок × 8 = 65K
- **Итого новое: ~2.2M параметров (0.7% от модели)**

---

## 9. Ссылки

### Абдукция и логика
1. Kakas, Kowalski & Toni (1992) — Abductive Logic Programming
2. Josephson & Josephson (1994) — Abductive Inference: Computation, Philosophy, Technology
3. Peirce C.S. (1903) — Harvard Lectures on Pragmatism

### HDC и символьные вычисления
4. Kanerva (2009) — Hyperdimensional Computing: An Introduction — Cognitive Computation
5. Gayler (2003) — Vector Symbolic Architectures Answer Jackendoff's Challenges
6. Kleyko et al. (2022) — VSA as Computing Framework — Proceedings of the IEEE
7. Plate (2003) — Holographic Reduced Representations

### Сознание и самосознание
8. Hofstadter (1979) — Gödel, Escher, Bach
9. Hofstadter (2007) — I Am a Strange Loop
10. Tononi (2004) — An Information Integration Theory of Consciousness — BMC Neuroscience
11. Oizumi, Albantakis & Tononi (2014) — IIT 3.0 — PLoS Computational Biology
12. Thagard & Stewart (2014) — Semantic Pointer Competition vs IIT — Consciousness and Cognition

### Когнитивные архитектуры на HDC
13. Eliasmith (2013) — How to Build a Brain — Oxford University Press
14. Eliasmith et al. (2012) — A Large-Scale Model of the Functioning Brain — Science

### Нейро-символическая интеграция
15. Riegel et al. (2020) — Logical Neural Networks — IBM
16. Serafini & Garcez (2016) — Logic Tensor Networks
17. Evans & Grefenstette (2018) — Differentiable Inductive Logic Programming

### Оптимизация
18. Ramapuram et al. (2024) — Sigmoid Self-Attention — Apple Research
19. Yang et al. (2024) — Gated Linear Attention — ICML
20. Schmidhuber (2010) — Formal Theory of Creativity and Intrinsic Motivation

---

*Документ будет обновляться по мере результатов обучения v14.1 и экспериментов v14.2.*
