# v18 Optimization Changes — итерация 2

## Дата: 2026-04-03

## Что реализовано

### 1. Relations (уровень 5 памяти) — семантические связи

**Файл**: `src/memory.rs`

- `RelationEntry` — структура связи: bind(subject, bind(role, object))
- `RelationMemory` — хранилище с LSH + subject/object/subject_role индексами
- Дедупликация: повторная связь → confidence++
- HDC unbind тест: unbind(relation, query) → recovered ≈ object_vec (подтверждено)

**Интеграция в WormMind** (`src/worm_mind.rs`):
- Phase 3.5: Relations QUERY — Sequence/Similar связи усиливают кандидатов
- Phase 7: СОЗДАНИЕ RELATIONS — при правильном predict: Sequence + Similar
- Только confidence >= 3 relations влияют (фильтрация шума)
- Boost через raw_score × (1 + boost) вместо evidence bundling (безопаснее)

### 2. Background Loop — автономная саморефлексия

**Файл**: `src/worm_mind.rs` — `BackgroundMind`

- INTROSPECT цикл: сканирование слабых фактов + конфликтов в Relations
- GENERALIZE цикл: поиск пар токенов с общими successors → Similar relations
- Вызывается каждые 1000 predict'ов из evaluate_mind()
- Не запускается в readonly eval (evaluate_mind_readonly)

### 3. Канал связи с оператором

**Файл**: `src/worm_mind.rs` — `WormMessage` enum

- 5 типов сообщений: Ask, NeedData, Report, Discovery, Confused
- Генерируются в WormMind::think():
  - ASK при confidence < 0.05
  - REPORT при высокой confidence + novel discovery
  - CONFUSED при > 5 contradictions
- BackgroundMind генерирует: NeedData, Confused, Discovery
- Вывод в stderr с emoji маркерами

### 4. Очистка warnings

- Было: 25 warnings
- Стало: 0 warnings
- Подход: `#[allow(dead_code)]` для API struct/methods, `_` для unused vars

## Результаты (sr=10, 27M tokens)

| Метрика | Baseline v18.0 | v18+awareness | v18+relations |
|---------|---------------|---------------|---------------|
| Train Top-1 (5K) | 23.4% | 22.2% | 22.7% |
| Train Top-5 (5K) | 37.2% | 35.9% | 36.0% |
| Confidence | 0.24 | 0.39 | 0.39 |
| Relations | — | — | 2130+ |
| Warnings | 25 | 15 | 0 |
| Tests | 29/29 | 29/29 | 31/31 |

## Самокритика

1. **Relations boost на sr=10 слабый** — мало данных → relations шумные.
   При confidence < 3 relations игнорируются. На sr=1 ожидается больше пользы.

2. **Final eval accuracy ниже train eval** — это НОРМАЛЬНО, разные позиции.
   Baseline показывал ту же разницу.

3. **Зацикливание генерации** НЕ решено — нужна HDCMemory (v14 каузальная память).

4. **Опкоды НЕ удалены** — изменение индексов опкодов нарушит детерминированность
   HDC языка (seed = 0x1DC18 + index). Оставлены с пометкой deprecated.

## Итерация 2 — HDCMemory + Самопознание

### 5. Binary HDCMemory (Фаза 4) ✅
- mass_role + decay_role: 2 обучаемых binary vec
- build_causal_context: exponential decay, data-dependent weights
- Заменяет фиксированный context_bundle в WormMind::think()
- Hebbian обучение при правильном predict

### 6. Самопознание (Фаза 5) ✅
- SelfKnowledge struct: счётчики, health, accuracy, confidence
- EVALUATE цикл в BackgroundMind
- describe() → "Я — HDC Brain v18. У меня 1.5M фактов..."
- observed_accuracy обновляется из eval

### Аудит фиктивности ✅
- SelfAwareness: НЕ декорация — реально меняет k параметры
- Cross-worm reading: НЕ декорация — unbind математически корректен
- Relations: ЧАСТИЧНО декорация на sr=10 (confidence<3 фильтруются)
- BackgroundMind: НЕ декорация — реально сканирует и генерирует
- HDCMemory: при random roles ≈ равномерное окно, но Hebbian меняет
- WormMessage: НЕ декорация — генерируют осмысленные сообщения

## Что НЕ сделано (следующие итерации)

1. **Самомодификация** (Фаза 6) — черви настраивают параметры
2. **Процедуры** (уровень 6) — bind(IF, trigger, THEN, action_chain)
3. Тест на sr=1 для валидации на полных данных
4. bind(Self, HasRole, ...) HDC vectors для самопознания
5. Teacher для приёма ответов оператора
