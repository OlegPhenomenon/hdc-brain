# HDC-Brain v14 — Реальная архитектура

## Источник
Файл: `/Users/oleghasjanov/Documents/learning/hoffman_swarm/hdc-brain-v14/hdc_brain_v14.py`
Тренируется на vast.ai, BPB 3.896 (iter 78K/100K), генерирует связный русский текст.

## Ключевой принцип: НИЧЕГО от трансформера

Нет softmax. Нет learned position embeddings. Нет projection matrices (W_q, W_k, W_v).
Все операции основаны на HDC: binding, permutation, bundling, similarity.

## Пайплайн

```
BPE Token → STE Bipolar Codebook → Cyclic Permutation → N × HDCBlock → ThoughtLoop → Codebook^T → logits
```

## Компоненты

### 1. STE Bipolar Codebook (строки 206, 221-226)
```python
real = self.codebook[idx]                    # float параметры
alpha = mean(abs(real))                       # масштаб
hard = alpha * sign(real)                     # bipolar: +α / -α
return (hard - real).detach() + real          # STE: forward=bipolar, backward=float
```
- Токен → bipolar вектор (+1/-1, масштабированный)
- STE (Straight-Through Estimator): градиенты текут через sign()
- Weight-tying: тот же codebook используется для output (logits = h @ codebook^T)

### 2. Cyclic Permutation (строки 228-232)
```python
indices = (arange(D) - positions) % D        # циклический сдвиг
x = x.gather(2, indices)                      # переставить биты
```
- Позиция кодируется через циклический сдвиг HDC-вектора
- permute(vec, pos) — стандартная HDC операция для "роли/позиции"
- НЕ learned embeddings, НЕ sinusoidal

### 3. HDCMemory — рабочая память (строки 77-99)
```python
mass = sigmoid(mass_proj(x))                  # важность токена [0,1]
decay = sigmoid(decay_proj(x))                # скорость забывания [0,1]
W = exp(cumsum(log(decay)))                   # каузальная матрица затухания
context = W @ (mass * x)                      # взвешенная кумулятивная сумма
```
- Параллельный scan (НЕ self-attention)
- Каждый токен имеет "массу" (importance) и "decay" (forgetting rate)
- Data-dependent: decay зависит от содержания, не фиксированный
- Параметры: 2 × D (mass_proj + decay_proj)
- Похоже на SSM/linear attention, но с HDC-мотивацией

### 4. HDCAttention — binding-based внимание (строки 40-74)
```python
# 3 role vectors (learned, D параметров каждый, НЕ D×D матрицы)
bv_q = Parameter(randn(D) * 0.02)
bv_k = Parameter(randn(D) * 0.02)
bv_v = Parameter(randn(D) * 0.02)

# Binding = назначение роли (element-wise multiply в bipolar space)
Q = x * sign(bv_q)                           # "этот токен в роли query"
K = x * sign(bv_k)                           # "этот токен в роли key"
V = x * sign(bv_v)                           # "этот токен в роли value"

# Attention scores = similarity между bound-векторами
scores = Q @ K^T * scale                     # dot product = HDC similarity
attn = sigmoid(scores * 4.0)                 # sigmoid, НЕ softmax
attn = causal_mask(attn)                     # каузальная маска

out = attn @ V                               # взвешенная комбинация
```

**Почему это HDC, а не трансформер:**
- Binding (x * sign(bv)) = HDC bind операция в bipolar space
- sign(bv) — bipolar вектор, назначает "роль" каждому биту
- 3×D параметров vs 3×D×D в трансформере (в 4096 раз меньше!)
- sigmoid вместо softmax — не конкурирующее внимание
- Q·K^T между bound-векторами = "насколько query-аспект позиции i совместим с key-аспектом позиции j"

### 5. ControllerBlock (строки 24-37)
```python
h = LayerNorm(x)
h = GELU(down(h))       # D → inner_dim
h = dropout(up(h))       # inner_dim → D
return x + h             # residual
```
- Feed-forward с residual
- Единственный компонент с "обычными" матрицами (down/up projection)

### 6. HDCBlock = Memory + Attention + Controller (строки 102-118)
```python
mem = HDCMemory(x)
x = LayerNorm(x + mem)
attn = HDCAttention(x)
x = LayerNorm(x + attn)
x = Controller(x)
```

### 7. ThoughtLoop — многопроходное рассуждение (строки 121-181)
```python
# Первый проход — обязательный
h = blocks(tokens)

# Дополнительные "мысли" (2-4 прохода через ТЕ ЖЕ блоки)
for t in range(1, n_thoughts):
    gate = sigmoid(thought_gates[t])          # learned gate [0,1]
    thought = blocks(LayerNorm(h) + thought_pos[t])
    h = h + gate * (thought - h)              # gated residual
```
- Модель "думает" несколько раз через те же блоки
- Каждый thought уточняет понимание
- gate[2] = 0.721 → третий thought активен и полезен
- Уникальная концепция, нет в стандартных трансформерах

## Параметры
- hdc_dim=4096, n_blocks=6, controller_dim=768, max_thoughts=4
- Codebook: 16K × 4096 = 65.5M
- Blocks: 6 × (Memory + Attention + Controller) ≈ 50M
- Total: ~116M параметров

## Результаты (iter 78K/100K)
- BPB: 3.896 (train loss 2.99)
- Генерирует связный русский текст
- "Москва — это столица России, ее статус и место в мире"
- "В 1945 году закончилась Великая Отечественная война"
- Инференс в 3-4 раза быстрее GPT-2 при сравнимом качестве

## Что перенести в v18

HDC Binding Attention для червей должен использовать ТОТ ЖЕ принцип:
- **Role vectors** (binding vectors) для назначения ролей
- **Binding = XOR** в binary space (аналог * sign() в bipolar)
- **Similarity = XNOR+POPCNT** (аналог dot product)
- **Soft attention** по памяти червей (не hard LSH lookup)
