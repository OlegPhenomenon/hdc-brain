# HDC-Brain v13: Architecture Plan

## Based on 5 papers (2025-2026)

### Sources
1. **THDC** (Jan 2026) — Trainable codebook with STE, D=64 works
2. **BiHDTrans** (Sep 2025) — HDC attention: binding Q/K/V, boolean mask, no FFN
3. **HDC Probe** (ICLR 2026) — LLM residuals map to HDC (94% binary accuracy)
4. **Optimal Encoding** (Feb 2026) — Learning vs cognition geometry, learnable kernel width
5. **AR-HDC** (2025) — Autoregressive HDC for sequences, detach trick

---

## Core Architecture

```
Input: tokens (char-level для прототипа, BPE для production)
  |
  v
[Trainable HDC Codebook] — STE: forward=sign(w), backward=gradient через real w
  |                         Scaling: alpha * sign(w) где alpha = mean(|w|)
  v                         (из THDC paper)
{-1,+1}^D bipolar hypervectors
  |
  v
[Position: Cyclic Permutation] — rho^t(hv) = roll(hv, t)
  |                               Бесплатно, без параметров (из BiHDTrans)
  v
[HDC Memory Block] x N — ОСНОВНОЙ ВЫЧИСЛИТЕЛЬНЫЙ БЛОК
  |
  |  1. Context Mass: mass = sigmoid(learned_probe @ token)
  |     "Россия" -> 0.95, "это" -> 0.05
  |     (идея пользователя + learnable kernel width из Optimal Encoding)
  |
  |  2. Memory Accumulation:
  |     memory[t] = decay[t] * memory[t-1] + mass[t] * token_bound[t]
  |     (element-wise recurrence, O(D) per step, NO matmul)
  |
  |  3. HDC Attention (optional, from BiHDTrans):
  |     Q = memory ⊗ BV_q   (binding, D params)
  |     K = memory ⊗ BV_k   (binding, D params)
  |     V = memory ⊗ BV_v   (binding, D params)
  |     scores = Q @ K^T     (integer dot-product)
  |     mask = (scores > 0) AND causal_mask  (boolean, no softmax!)
  |     attn_out = sign(mask @ V)  (selective bundling + binarize)
  |
  |  4. Bundle: output = memory + attn_out + token
  |
  v
[Controller] — маленькая NN (из нашей идеи)
  LayerNorm -> Linear(D->inner) -> GELU -> Linear(inner->D) + residual
  (нужен для non-linear transform, BiHDTrans говорит что для classification
   не нужен, но для generation — скорее всего нужен)
  |
  v
[Output Head] — weight-tied с codebook
  logits = output @ codebook^T * scale
  scale = 1/sqrt(D) или learnable
```

---

## Два режима: Prototype и Production

### Prototype (начинаем с него)
- Tokenizer: char-level (vocab=170, уже есть)
- D = 4096 (validated by HDC Probe paper)
- N = 1 memory block (без HDC attention, только recurrence)
- Controller: 4 blocks x 512d inner
- ~34M params
- VRAM: ~2GB -> batch=256

### Production (после валидации)
- Tokenizer: BPE 8K-32K (sentencepiece)
- D = 4096
- N = 4 memory blocks (с HDC attention в каждом)
- Controller: 6 blocks x 1024d inner
- ~100-200M params
- VRAM: 6-10GB

---

## Key Design Decisions (из статей)

### 1. STE с scaling factor (из THDC)
```python
# НЕ просто sign(w)
alpha = torch.mean(torch.abs(w))
hard = alpha * torch.sign(w)
return (hard - w).detach() + w
```
Alpha сохраняет масштаб градиентов. Без него — нестабильность.

### 2. Position = Cyclic Permutation (из BiHDTrans)
```python
# Бесплатно, без параметров, без обучения
pos_encoded = torch.roll(token_hv, shifts=t, dims=-1)
```
BiHDTrans доказал: permutation сохраняет все HDC свойства.
Vs learned pos_embed: экономим vocab*D параметров.

### 3. Boolean Attention вместо Softmax (из BiHDTrans)
```python
scores = Q @ K.T        # integer dot-product
mask = (scores > 0)      # boolean threshold
mask = mask & causal      # нижний треугольник
out = sign(mask.float() @ V)  # selective bundling
```
39x быстрее softmax на FPGA. На GPU тоже быстрее (нет exp/div).

### 4. Dual-mode encoding НЕ нужен при D>=1500 (из Optimal Encoding)
При D=4096 single encoding strategy работает для ОБОИХ:
- Learning (backprop через codebook) — w~1.0
- Memory (retrieval из accumulated memory) — w~1.0
Не нужно усложнять с разными w.

### 5. Weight tying codebook↔output (из HDC Probe)
LLM residual streams ≈ HDC пространство.
Значит: один и тот же codebook для embed и output = правильно.
```python
logits = h @ self.codebook.weight.T  # output = поиск ближайшего слова
```

### 6. Detach в autoregressive generation (из AR-HDC)
```python
# При генерации: detach предотвращает gradient explosion
for step in range(gen_len):
    logits = model(context)
    next_token = sample(logits[:, -1])
    context = cat([context, next_token.detach()])  # detach!
```

---

## Параметры для прототипа

```python
CONFIG = {
    'hdc_dim': 4096,
    'n_memory_blocks': 1,       # только recurrence, без attention
    'n_controller_blocks': 4,
    'controller_dim': 512,
    'use_hdc_attention': False,  # включим в production
    'use_ste': True,             # bipolar codebook с STE
    'use_cyclic_pos': True,      # permutation вместо learned pos
    'dropout': 0.1,
    'max_seq_len': 512,
}
```

### Подсчет параметров (char-level, vocab=170):
- Codebook: 170 x 4096 = 696K
- Mass probe: 4096 = 4K
- Decay probe: 4096 = 4K
- Controller (4 blocks): 4 x (4096x512 + 512x4096 + LN) = 16.8M
- Output LN: 8K
- **Total: ~17.5M** (очень маленькая!)

### VRAM estimate (batch=256, seq=256):
- Model: 17.5M x 12B (params+adam+grads) = 210MB
- Activations: 256 x 256 x 4096 x 2B x ~8 = 4.3GB
- **Total: ~4.5GB** -> fits easily in 12GB

### Speed estimate:
- Memory loop: 256 steps x element-wise ops = ~5ms
- Controller: 4 blocks MLP = ~2ms
- Output matmul: 64K x 4096 x 170 = ~1ms
- Backward: ~15ms
- **Total: ~25-40ms/iter**
- **Epoch (1GB): ~15K iters x 35ms = ~9 minutes**

---

## Что делает эту архитектуру УНИКАЛЬНОЙ

1. **Memory через bind/bundle** — не нейросеть, а алгебра (никто не делал для текста)
2. **Context mass** — learned "важность" слова для контекста (новая идея)
3. **Всё в одном HDC пространстве** — нет проекций туда-обратно
4. **Boolean attention** (опционально) — из BiHDTrans, но для генерации текста
5. **Cyclic permutation** вместо learned positional encoding
6. **STE bipolar codebook** — из THDC, но для language model

Ни одна существующая публикация не комбинирует все эти элементы.

---

## План реализации

### Phase 1: Prototype (1-2 часа)
1. Закодить hdc_brain.py с recurrence + controller (без attention)
2. Char-level tokenizer (уже есть)
3. Запустить на GPU, batch=256
4. Цель: loss падает, генерация хоть что-то осмысленное

### Phase 2: Validation (4-8 часов обучения)
1. Мониторить BPB, quality генерации, context mass
2. Сравнить с v11 baseline (BPB 3.25)
3. Если работает — переходим к Phase 3

### Phase 3: HDC Attention (если Phase 2 ОК)
1. Добавить boolean causal attention из BiHDTrans
2. N=4 memory blocks с attention
3. Ожидание: значительное улучшение quality

### Phase 4: BPE (если Phase 3 ОК)
1. Обучить sentencepiece на данных
2. Увеличить модель для BPE vocab
3. Финальная модель для dubbing app
