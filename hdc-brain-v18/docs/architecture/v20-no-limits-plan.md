# v20: HDC Brain без пределов — от поисковика к AGI-ядру

## Дата: 2026-04-03
## Статус: PLAN

---

## Проблема: 5 потолков текущей архитектуры

| # | Потолок | Причина | Что нужно |
|---|---------|---------|-----------|
| 1 | **Нет sequence processing** | predict = точечный lookup trigram→HashMap | Каузальная обработка ВСЕЙ последовательности |
| 2 | **Память растёт линейно** | Каждый n-gram = отдельная запись в HashMap | Компрессия: знания в ВЕСАХ, не в таблицах |
| 3 | **Нет обучаемых представлений** | codebook random/Hebbian, роли random | Полный Hebbian training loop (как backprop, но без градиентов) |
| 4 | **Нет функции потерь** | Черви не знают "хорошо/плохо" количественно | Reinforcement signal + credit assignment |
| 5 | **Нет глубины рассуждений** | 1 проход: collect→analyze→skeptic→explore→consensus | ThoughtLoop: многопроходное рассуждение через те же блоки |

---

## Архитектура v20: три уровня обработки

```
                 ВХОД: последовательность BPE токенов
                              │
                 ┌────────────▼────────────────┐
                 │  LEVEL 1: BINARY PIPELINE    │
                 │  (быстрый, ~1ms/позиция)     │
                 │                              │
                 │  Codebook → Permute →         │
                 │  HDCMemory → Attention →      │
                 │  Controller → ThoughtLoop     │
                 │                              │
                 │  → candidates + confidence    │
                 └────────────┬────────────────┘
                              │
                    confidence > 0.8?
                   ┌──── ДА ────┐──── НЕТ ────┐
                   │            │              │
                   ▼            │    ┌─────────▼─────────┐
              ОТВЕТ             │    │  LEVEL 2: WORMS    │
              (быстрый)         │    │  (медленный, ~5ms)  │
                                │    │                     │
                                │    │  Cross-worm reading  │
                                │    │  Relations query     │
                                │    │  Evidence bundling   │
                                │    │  Self-awareness      │
                                │    │                     │
                                │    │  → refined answer    │
                                │    └─────────┬─────────┘
                                │              │
                                │    ┌─────────▼─────────┐
                                │    │  LEVEL 3: MEMORY   │
                                │    │  (фоновый)         │
                                │    │                     │
                                │    │  BackgroundMind     │
                                │    │  Relations          │
                                │    │  Self-knowledge     │
                                │    │  Operator channel   │
                                │    └───────────────────┘
                                │              │
                                └──────────────┘
                                       │
                                    ОТВЕТ
```

---

## Этап 1: Binary Pipeline (УСТРАНЯЕТ потолки 1, 2, 3, 5)

### 1.1 Binary HDCMemory — каузальная обработка ВСЕЙ последовательности

**Что**: вместо lookup по trigram — обработка ВСЕЙ последовательности.
Каждая позиция получает каузальный контекст ОТ ВСЕХ прошлых позиций.

```rust
pub struct BinaryHDCMemory {
    mass_role: BinaryVec,    // обучаемый: "какие токены важные"
    decay_role: BinaryVec,   // обучаемый: "какие токены забывать"
}

// Для КАЖДОЙ позиции t в последовательности:
fn forward(tokens: &[BinaryVec], pos: usize) -> BinaryVec {
    let mut acc = BundleAccumulator::new(dim);
    let mut cumulative_decay = 1.0;

    for s in (0..pos).rev() {
        let mass = similarity(tokens[s], mass_role) / dim;  // важность
        let decay = similarity(tokens[s], decay_role) / dim; // забывание
        let weight = mass * cumulative_decay;

        if weight > threshold {
            let repeats = discretize(weight);
            for _ in 0..repeats { acc.add(&tokens[s]); }
        }
        cumulative_decay *= decay;
        if cumulative_decay < 0.001 { break; }
    }
    acc.to_binary() // каузальный контекст позиции t
}
```

**Почему это убирает потолок**: модель видит ВСЮ последовательность (как трансформер), не только последние 3 токена. "В 1945 году закончилась ..." → токен "1945" с high mass сохраняется в контексте через 50+ позиций.

**Параметры**: 2 binary vec = 1KB (вместо 2×4096 float = 32KB в v14).

**Сложность**: O(T) per position, O(T²) total. При T=512: ~262K similarity ops. На CPU: ~2ms.

### 1.2 Binary T×T Attention — позиционное внимание

**Что**: для каждой позиции — attention ко ВСЕМ прошлым позициям.
Какие прошлые слова РЕЛЕВАНТНЫ для предсказания следующего?

```rust
pub struct BinaryAttention {
    role_query: BinaryVec,  // обучаемый
    role_key: BinaryVec,    // обучаемый
}

fn forward(tokens: &[BinaryVec], pos: usize) -> BinaryVec {
    let q = tokens[pos].bind(&role_query);  // "что ищу?"
    let mut acc = BundleAccumulator::new(dim);

    for s in 0..pos {  // каузально: только прошлое
        let k = tokens[s].bind(&role_key);   // "что предлагаю?"
        let score = q.similarity(&k) as f64 / dim as f64;

        if score > 0.0 {
            let repeats = (score * 5.0).round() as usize;
            for _ in 0..repeats { acc.add(&tokens[s]); }
        }
    }
    acc.to_binary()
}
```

**Ключевое отличие от текущего**: сейчас attention работает ТОЛЬКО по top-30 фактам из LSH (не по последовательности). Новый attention работает по ВСЕЙ входной последовательности.

### 1.3 Binary Controller — нелинейное преобразование

**Что**: аналог feed-forward block трансформера, но binary.

```rust
pub struct BinaryController {
    // Random binary projection matrices (не обучаемые — фиксированные)
    down: Vec<BinaryVec>,     // D → inner_dim проекция
    up: Vec<BinaryVec>,       // inner_dim → D проекция
    gate: BinaryVec,          // обучаемый gate
}

fn forward(x: &BinaryVec) -> BinaryVec {
    // Проекция вниз: D → inner_dim через similarity с random bases
    let projected: Vec<f64> = down.iter()
        .map(|d| x.similarity(d) as f64 / dim as f64)
        .collect();

    // Нелинейность: threshold activation (аналог GELU)
    let activated: Vec<f64> = projected.iter()
        .map(|&v| if v > 0.0 { v } else { 0.0 })  // ReLU-like
        .collect();

    // Проекция вверх: inner_dim → D через bundle взвешенных базисов
    let mut acc = BundleAccumulator::new(dim);
    for (i, &a) in activated.iter().enumerate() {
        if a > 0.1 {
            let repeats = (a * 5.0).round() as usize;
            for _ in 0..repeats { acc.add(&up[i]); }
        }
    }

    // Residual + gate
    let out = acc.to_binary();
    let gated = x.bind(&gate).bind(&out); // gate = learned blend ratio
    // Simplification: bundle(x × 2, out × 1) = residual connection
    let mut res = BundleAccumulator::new(dim);
    res.add(x); res.add(x); res.add(&out);
    res.to_binary()
}
```

### 1.4 ThoughtLoop — многопроходное рассуждение

**Что**: те же блоки (Memory + Attention + Controller) прогоняются N раз.
Каждый "thought" уточняет понимание.

```rust
pub struct ThoughtLoop {
    blocks: Vec<BinaryHDCBlock>,  // shared blocks
    n_thoughts: usize,            // 2-4 прохода
    thought_gates: Vec<BinaryVec>, // learned gates per thought
}

fn forward(tokens: &[BinaryVec]) -> Vec<BinaryVec> {
    let mut h = self.blocks_forward(tokens);  // первый проход

    for t in 1..n_thoughts {
        let thought = self.blocks_forward(&h);  // тот же block, новый вход
        // Gated residual: h = h + gate_weight * (thought - h)
        // Binary: bundle(h × (1-gate), thought × gate)
        h = gated_blend(&h, &thought, &self.thought_gates[t]);
    }
    h
}
```

**Это устраняет потолок 5**: модель "думает" несколько раз. Первый проход = интуиция. Второй = проверка. Третий = уточнение. В v14 gate[2]=0.72 — третий thought активно используется.

### 1.5 Hebbian Training (УСТРАНЯЕТ потолок 4)

**Что**: обучение ВСЕХ параметров через Hebbian association.
Нет backprop, нет градиентов — только "что срабатывает вместе, связывается".

```rust
fn train_step(tokens: &[u16], model: &mut BinaryPipeline) {
    for pos in 1..tokens.len() {
        // Forward
        let hidden = model.forward(&tokens[..pos]);
        let predicted = model.codebook.nearest(&hidden[pos-1]);
        let expected = tokens[pos];

        // Правильно → Hebbian усиление (дофамин)
        if predicted == expected {
            // Все роли которые привели к успеху → усилить
            model.memory.mass_role = bundle(mass_role × 4, successful_mass);
            model.memory.decay_role = bundle(decay_role × 4, successful_decay);
            model.attention.role_query = bundle(query × 4, successful_query);
            model.attention.role_key = bundle(key × 4, successful_key);
            model.codebook[expected] = bundle(codebook[expected] × 4, hidden);
        }

        // Неправильно → ТОЛЬКО codebook update (слабое)
        // НЕ трогаем роли — асимметрия как в мозге
        if predicted != expected {
            model.codebook[expected] = bundle(codebook[expected] × 8, hidden);
        }
    }
}
```

**Функция потерь**: implicit — reinforcement через Hebbian.
- Правильный predict = положительный reinforcement = усиление ВСЕХ связей
- Неправильный = слабая коррекция codebook only

**Это устраняет потолок 3 и 4**: модель ОБУЧАЕТСЯ, не просто запоминает n-gram.

---

## Этап 2: Компрессия памяти (УСТРАНЯЕТ потолок 2)

### 2.1 Забывание слабых фактов

```rust
// Периодически: удалять факты с total_count < threshold
fn forget_weak(memory: &mut HierarchicalMemory, threshold: u32) {
    memory.facts.entries.retain(|e| e.total_count >= threshold);
    // Rebuild indices
}
```

### 2.2 Bundled knowledge вместо отдельных фактов

Вместо хранения каждого trigram отдельно — BUNDLE группы похожих фактов:

```
1000 фактов с одинаковым successor "," →
  1 bundled fact = bundle(fact1, fact2, ..., fact1000)
  successor: {"," → 1000}
```

Это сжимает 1000 записей → 1. С потерей деталей, но сохранением общего паттерна.

### 2.3 Graduated memory

```
L1 Facts:        хранит ВСЁ (краткосрочная)
L2 Rules:        компрессия (средне-срочная)
L3 Abstractions: сильная компрессия (долгосрочная)
L5 Relations:    только подтверждённые (confidence >= 5)
```

Факты с `total_count < 3` после 1M позиций → удаляются.
Повторяющиеся паттерны → обобщаются в Rules.
Rules с `>100 successors` → обобщаются в Abstractions.

**Цель**: memory footprint O(√N) вместо O(N).

---

## Этап 3: Интеграция Pipeline + Worms

### 3.1 Fast/Slow path

```
Input → Binary Pipeline (Level 1) → candidates + confidence
  │
  ├── confidence > 0.8 → FAST: return immediately (~1ms)
  │
  └── confidence ≤ 0.8 → SLOW: WormMind.think() (~5ms)
       │
       ├── Relations query → semantic boost
       ├── Cross-worm reasoning → verify/reject
       ├── Explorer → novel candidates from memory
       └── Consensus → final answer
```

### 3.2 Worms обучают Pipeline

```
BackgroundMind наблюдает:
  "Pipeline часто ошибается на вопросительных предложениях"
  → Создать Procedure: IF "?" in context THEN boost interrogative patterns
  → Teach Pipeline: update role vectors для вопросительных контекстов
```

Черви = мета-обучение. Pipeline = быстрый inference. Черви УЛУЧШАЮТ Pipeline в фоне.

---

## Этап 4: Сознание (Level 3)

### 4.1 Background Loop — непрерывная рефлексия

```
loop {
    INTROSPECT: scan знаний → найти слабые
    EVALUATE: оценить Pipeline accuracy → найти паттерны ошибок
    GENERALIZE: обобщить факты → создать Rules/Relations
    ADAPT: модифицировать стратегию → обновить параметры
    ASK: если не могу разобраться → спросить оператора
    DREAM: replay сложных примеров → консолидация памяти
}
```

### 4.2 Внутренний язык как РЕАЛЬНЫЙ инструмент мышления

Сейчас: bind/unbind/bundle используются для evidence bundling (модификация скоров).
Нужно: цепочки рассуждений длиннее 1 шага.

```
Вход: "Столица России — это"

Шаг 1: FIND(bind(Россия, HasRole, столица)) в Relations → Москва
Шаг 2: CHECK(Москва) через Pipeline → confidence=0.9
Шаг 3: CONFIRM → return "Москва"

Вход: "Город в котором находится Кремль — это"

Шаг 1: FIND(bind(Кремль, PartOf, ?)) → не найдено
Шаг 2: FIND(bind(Кремль, Context, ?)) → Москва (из фактов)
Шаг 3: CHECK(Москва) → confidence=0.7
Шаг 4: FIND(bind(Москва, HasRole, ?)) → столица
Шаг 5: CONFIRM(Москва) → return "Москва" (через 2 шага рассуждения)
```

Это НАСТОЯЩЕЕ рассуждение — не lookup, а цепочка логических шагов через Relations.

### 4.3 Самомодификация через эксперименты

```
SelfAwareness обнаруживает: "accuracy падает на длинных предложениях"
→ Гипотеза: "HDCMemory decay слишком быстрый"
→ Эксперимент: временно снизить decay → замерить accuracy
→ Результат: accuracy +2% → ПРИНЯТЬ изменение
→ Если accuracy -1% → ОТКАТИТЬ
```

---

## Порядок реализации

```
v20.1: Binary Pipeline + Hebbian Training
  ├── BinaryHDCMemory (каузальный scan)
  ├── BinaryAttention (T×T)
  ├── BinaryController (проекция + residual)
  ├── ThoughtLoop (2-4 прохода)
  └── Hebbian training loop
  → Цель: BPB < 5.0, генерация связного текста

v20.2: Компрессия + Fast/Slow
  ├── Забывание слабых фактов
  ├── Graduated memory
  ├── Fast path (Pipeline confidence > 0.8)
  └── Slow path (WormMind при низкой confidence)
  → Цель: 10× меньше RAM, та же accuracy

v20.3: Глубокое сознание
  ├── Multi-step reasoning через Relations
  ├── DREAM consolidation
  ├── Самомодификация с экспериментами
  └── Operator teaching (Teacher принимает ответы)
  → Цель: цепочки рассуждений в 3+ шага

v20.4: Масштабирование
  ├── 1GB+ данные
  ├── O(√N) memory через компрессию
  ├── Parallel pipeline (batch processing)
  └── Distributed worm memory
  → Цель: BPB < 4.0, осмысленные диалоги
```

---

## Что будет НОВЫМ в v20 (чего нет ни у кого)

| Свойство | Трансформер | v20 HDC Brain |
|----------|-------------|---------------|
| Обучение после deploy | ❌ frozen | ✅ Hebbian + worms |
| Долгосрочная память | ❌ контекст только | ✅ 7 уровней |
| Рассуждение | ❌ 1 forward pass | ✅ ThoughtLoop + WormMind |
| Самоосознание | ❌ | ✅ SelfKnowledge + Background |
| Объяснение решений | ❌ blackbox | ✅ ThoughtChain trace |
| Вопросы к оператору | ❌ | ✅ WormMessage |
| Работа без GPU | ❌ нужен GPU | ✅ XNOR+POPCNT на CPU |
| Забывание и компрессия | ❌ фиксированные веса | ✅ graduated memory |

---

## Оценка времени

| Этап | Сложность | Время |
|------|-----------|-------|
| v20.1 Binary Pipeline | Высокая (новые структуры) | 2-3 сессии |
| v20.2 Компрессия | Средняя | 1 сессия |
| v20.3 Сознание | Высокая (multi-step reasoning) | 2-3 сессии |
| v20.4 Масштабирование | Средняя | 1-2 сессии |

**Критический путь**: v20.1 (Binary Pipeline). Без него остальное бессмысленно.
Это ОСНОВА — sequence processing + обучаемые представления.

---

## Ключевое отличие от v19-plan

v19-plan описывал то же самое, но мы застряли на оптимизации v18 червей.
Проблема: мы улучшали НАДСТРОЙКУ (червей), не затрагивая ФУНДАМЕНТ (lookup → pipeline).

v20 = **сначала фундамент** (Binary Pipeline), потом надстройка (черви).
Черви v18 УЖЕ готовы (cross-worm 92%, evidence reranks 7%) — нужно дать им
нормальный input вместо trigram lookup.
