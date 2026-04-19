# v19: Binary HDC Brain — от поисковика к мозгу

## Проблема

v18 = поисковая система: trigram → lookup в хэш-таблицу → результат.
v14 = мозг: обработка последовательности → T×T attention → генерация.

v18 не может генерировать связный текст потому что у него нет:
1. Обработки последовательности (только точечный trigram lookup)
2. Обучаемых представлений (codebook фиксированный, роли random)
3. Функции потерь (черви не знают "хорошо" vs "плохо")

## Цель v19

Binary HDC модель которая:
- Генерирует связный грамматический текст (как v14)
- Использует ТОЛЬКО binary HDC операции (XOR, XNOR+POPCNT, permute, bundle)
- Обучается через Hebbian + reinforcement (НЕ backprop)
- Имеет долгосрочную память червей (уникальное преимущество над v14)
- Работает на CPU (XNOR+POPCNT вместо float matmul)

## Три компонента (A + B + C)

### A. Binary HDCMemory — каузальная память

Аналог HDCMemory из v14 (`hdc_brain_v14.py:77-99`), но binary.

v14 (float):
```python
mass = sigmoid(mass_proj(x))           # важность токена [0,1]
decay = sigmoid(decay_proj(x))         # скорость забывания [0,1]
W = exp(cumsum(log(decay)))            # каузальная матрица затухания
context = W @ (mass * x)              # взвешенная кумулятивная сумма
```

v19 (binary):
```
mass_role = обучаемый binary vec (Hebbian)
decay_role = обучаемый binary vec (Hebbian)

Для каждой позиции t:
  mass[t] = similarity(token[t], mass_role) / dim    // [-1,1] → "важность"
  decay[t] = similarity(token[t], decay_role) / dim  // [-1,1] → "забывание"

  // Каузальный контекст: bundle прошлых токенов с весами
  context_acc = BundleAccumulator
  for s = 0..t:
    weight = mass[s] * decay^(t-s)    // экспоненциальное затухание
    if weight > threshold:
      repeat = (weight * 10).round()  // дискретизация веса для bundle
      for _ in 0..repeat:
        context_acc.add(token[s])

  context[t] = context_acc.to_binary()
```

Параметры: 2 binary vec (mass_role, decay_role) = 2 × 4096 бит.
v14 аналог: 2 × D float = 2 × 4096 float.

Обучение mass_role/decay_role: Hebbian — если правильный predict, bundle role с контекстом который привёл к успеху.

### B. Binary Attention T×T — позиционное внимание

Аналог HDCAttention из v14 (`hdc_brain_v14.py:40-74`), но binary.

v14 (float):
```python
Q = x * sign(bv_q)                    # binding с ролью query
K = x * sign(bv_k)                    # binding с ролью key
V = x * sign(bv_v)                    # binding с ролью value
scores = Q @ K^T * scale              # T×T attention scores
attn = sigmoid(scores * 4.0)          # causal, sigmoid не softmax
out = attn @ V                        # взвешенная комбинация
```

v19 (binary):
```
role_query = обучаемый binary vec (Hebbian)
role_key = обучаемый binary vec (Hebbian)

Для каждой позиции t (target):
  Q = token[t] XOR role_query          // binding = назначение роли

  // Attention scores ко всем прошлым позициям
  attended_acc = BundleAccumulator
  for s = 0..t:                        // causal: только прошлое
    K = token[s] XOR role_key
    score = similarity(Q, K) / dim     // [-1,1]

    if score > 0:                      // только положительное внимание
      repeat = (score * 5).round()     // дискретизация для bundle
      for _ in 0..repeat:
        attended_acc.add(token[s])      // value = сам токен

  attended[t] = attended_acc.to_binary()
```

Параметры: 2 binary vec = 2 × 4096 бит.
v14 аналог: 3 × D float.

Сложность: O(T²) binary similarity operations.
При T=512, dim=4096: 512×512 × 64 XNOR+POPCNT = ~17M ops.
На M3 Pro (~10 GOPS): ~1.7ms per position set. Приемлемо.

### C. Hebbian Training Loop

v14 обучается через cross-entropy + Adam optimizer (backprop).
v19 обучается через Hebbian association + reinforcement (НЕ backprop).

```
for epoch in 0..n_epochs:
  for batch in data.chunks(batch_size):
    for pos in 0..batch.len()-1:
      // 1. Forward
      context = HDCMemory.forward(batch, pos)
      attended = Attention.forward(batch, pos)
      combined = bundle(context, attended)

      // 2. Predict
      scores = [similarity(combined, codebook[t]) for t in vocab]
      predicted = argmax(scores)
      expected = batch[pos + 1]

      // 3. Hebbian update (если правильно)
      if predicted == expected:
        // "Что срабатывает вместе — связывается"
        role_query = bundle(role_query × inertia, successful_Q)
        role_key = bundle(role_key × inertia, successful_K)
        mass_role = bundle(mass_role × inertia, high_mass_tokens)
        codebook[expected] = bundle(codebook[expected] × inertia, combined)

      // 4. Anti-Hebbian (если неправильно — слабое ослабление)
      if predicted != expected:
        // Не трогать role vectors (асимметричное обучение как в мозге)
        // Но: обновить codebook expected чтобы он был ближе к combined
        codebook[expected] = bundle(codebook[expected] × inertia, combined)
```

Ключевые отличия от backprop:
- Нет градиентов, нет chain rule, нет error propagation
- Только локальные обновления (Hebbian = "увидел вместе → связал")
- Асимметрия: правильный predict → усиление, неправильный → слабое обновление codebook
- Мозг так и работает: дофамин при успехе, слабая коррекция при ошибке

### D. Интеграция с WormMind (уникальное преимущество)

v14 не имеет долгосрочной памяти. v19 = v14-like processing + worm memory.

```
Уровень 1: Binary HDCMemory + Attention (быстрая обработка контекста)
  → "интуиция" — мгновенная реакция на паттерн

Уровень 2: WormMind reasoning (глубокий анализ при неуверенности)
  → "рассуждение" — если уровень 1 неуверен, черви думают

Уровень 3: Long-term worm memory (факты, правила, абстракции)
  → "знание" — черви помнят то что уровень 1 забыл

Predict pipeline:
  1. Level 1 forward → top-10 candidates + confidence
  2. IF confidence > 0.8: return Level 1 answer (быстро, ~1ms)
  3. ELSE: WormMind.think() → reasoning over candidates (~5ms)
  4. Combine: Level 1 scores × WormMind evidence → final answer
```

Это даёт v19 преимущество над v14:
- Факты которые модель видела 1 раз → сохранены в worm memory (v14 забудет)
- Рассуждение при неуверенности → лучше чем чистая "интуиция"
- Background learning → черви учатся пока модель работает

## Архитектура v19

```
Input: BPE tokens → Bipolar Codebook (Hebbian-trained)
  ↓
Cyclic Permutation (позиционное кодирование)
  ↓
N × BinaryHDCBlock:
  Binary HDCMemory (каузальный контекст)
  Binary Attention (T×T позиционное внимание)
  Controller: bundle(down_projection, GELU-like, up_projection)
  ↓
ThoughtLoop: K проходов через те же блоки (рассуждение)
  ↓
Output: similarity(hidden, codebook) → logits
  ↓
[optional] WormMind reasoning при низкой confidence
```

## Параметры (оценка)

| Компонент | v14 (float) | v19 (binary) | Экономия |
|-----------|-------------|--------------|----------|
| Codebook | 16K × 4096 float = 256MB | 16K × 4096 bit = 8MB | 32× |
| Attention roles | 3 × 4096 float = 48KB | 2 × 4096 bit = 1KB | 48× |
| Memory roles | 2 × 4096 float = 32KB | 2 × 4096 bit = 1KB | 32× |
| Controller | 4096 × 768 × 2 float = 24MB/block | binary equivalent TBD | TBD |
| Total (6 blocks) | ~116M params (float) | ~10M params (binary) | ~12× |

## Открытые вопросы

1. **Controller в binary**: down→GELU→up требует float matmul. Binary аналог?
   - Вариант: bundle-based projection через random binary matrices
   - Вариант: skip controller, компенсировать больше blocks

2. **Hebbian сходимость**: будут ли роли сходиться без градиентов?
   - Нужен эксперимент: обучить role vectors на toy task
   - Инерция 8:1 может быть слишком медленной или слишком быстрой

3. **T×T на длинных последовательностях**: O(T²) при T=512 = 262K ops
   - На CPU с XNOR+POPCNT: ~1-2ms
   - При T=2048: ~6-8ms — приемлемо?

4. **Codebook training**: Hebbian update codebook[token] = bundle(old, context)
   - Может ли codebook деградировать? (все вектора сходятся к среднему)
   - Нужен diversity mechanism?

## Порядок реализации

```
Этап 1: Binary HDCMemory (mass/decay parallel scan) ✅ DONE (pipeline.rs: PipelineMemory)
  → тест: test_pipeline_memory_forward ✅

Этап 2: Binary T×T Attention ✅ DONE (pipeline.rs: PipelineAttention)
  → тест: test_pipeline_attention_forward ✅

Этап 3: Hebbian Training Loop ✅ DONE (pipeline.rs: BinaryPipeline::train_chunk)
  → тест: test_pipeline_train_chunk ✅
  → cmd: cargo run --release -- train-pipeline

Этап 4: ThoughtLoop (multi-pass) — НЕ НАЧАТ
  → тест: 2+ thoughts дают лучший predict?

Этап 5: WormMind интеграция — НЕ НАЧАТ
  → тест: fast path (Level 1) + slow path (WormMind)

Этап 6: Full training на Wikipedia — ЗАПУЩЕН (sr=100, chunk=128)
  → цель: BPB < 5.0, связный русский текст
```

## Эталон (v14 результаты для сравнения)

- BPB: 3.896 (iter 78K/100K, ещё учится)
- Генерация: "Москва — это столица России", "В 1945 году закончилась ВОВ"
- Параметры: ~116M (float)
- Тренировка: GPU (vast.ai), ~4 дня
- Файл: `/Users/oleghasjanov/Documents/learning/hoffman_swarm/hdc-brain-v14/hdc_brain_v14.py`
