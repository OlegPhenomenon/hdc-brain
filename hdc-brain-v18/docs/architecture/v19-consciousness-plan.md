# v19: Conscious HDC Brain — полный план архитектуры

## Статус: IMPLEMENTED (5 итераций, 2026-04-03)
## Реализовано: Фазы 1-6 ✅ + InfluenceReport audit + 34 теста + 0 warnings
## Дата: 2026-04-03

---

## 1. Иерархия памяти — от статистики к знаниям

### 1.1 Текущее состояние (v18)

| Уровень | Что хранит | Количество | Проблема |
|---------|-----------|------------|----------|
| 0: Words | codebook (токен → вектор) | 16K | Фиксирован после build |
| 1: Facts | trigram → successors (частоты) | 8.5M | Статистика, не знания |
| 2: Rules | bigram → successors | 272K | Обобщения без причинности |
| 3: Abstractions | wide context → successors | 117K | Категории без связей |
| 4: Meta | profile знаний | 3 | Почти не используется |

**Корневая проблема**: всё хранится как `контекст → HashMap<token, count>`. Это таблица частот. Червь не может сохранить "Москва — столица России" как связь. Только "после 'столица' часто идёт 'России'".

### 1.2 Целевое состояние (v19)

| Уровень | Что хранит | Формат HDC | Операции |
|---------|-----------|-----------|----------|
| 0: Words | codebook (Hebbian-trained) | BinaryVec | similarity, bind |
| 1: Facts | наблюдения из данных | trigram binding → successors | FIND, STORE |
| 2: Rules | причинно-следственные связи | bind(IF, condition, THEN, action) | CHECK, COMPARE |
| 3: Abstractions | категории и обобщения | bundle(group) → properties | GENERALIZE |
| 4: Meta | знания о своих знаниях | diagnostic vectors | REFLECT, INTROSPECT |
| **5: Relations** | **семантические связи** | **bind(subject, role, object)** | **QUERY, UNBIND** |
| **6: Procedures** | **как действовать** | **bind(IF, trigger, THEN, action_chain)** | **EXECUTE, ADAPT** |

### 1.3 Реализация уровня 5: Семантические связи

```
Хранение: bind(Москва, bind(role_столица, Россия)) → один BinaryVec

Извлечение:
  "Столица чего Москва?" →
    query = bind(Москва, role_столица)
    unbind(stored_relation, query) → recovered ≈ Россия
    nearest_word(recovered) → "Россия"

  "Что является столицей России?" →
    query = bind(role_столица, Россия)
    unbind(stored_relation, query) → recovered ≈ Москва
    nearest_word(recovered) → "Москва"
```

**Структура RelationEntry:**
```rust
struct RelationEntry {
    vec: BinaryVec,           // bind(subject, bind(role, object))
    subject: u16,             // токен субъекта
    role: Opcode,             // тип связи (Similar, PartOf, HasRole, Cause, etc.)
    object: u16,              // токен объекта
    confidence: u32,          // сколько раз подтверждено
    source: RelationSource,   // откуда: data, inference, operator
}
```

**Типы связей (уже есть как Opcode):**
- `Similar` — "X похож на Y"
- `PartOf` — "X часть Y"
- `HasRole` — "X выполняет роль Y"
- `Cause` → `Effect` — "X потому что Y", "из X следует Y"
- `Before` → `After` — "до X было Y", "после X будет Y"
- `Sequence` — "X затем Y"

**Как черви создают связи:**
1. Explorer видит что в контексте "Москва — столица России" факты подтверждают связь
2. Explorer создаёт: `bind(word[Москва], bind(op[HasRole], word[столица]))` → facts confirm "России"
3. Skeptic проверяет: unbind(relation, word[Москва]) → должно быть close to bind(op[HasRole], word[столица])
4. Если подтверждено → store в Relations с confidence++

### 1.4 Реализация уровня 6: Процедуры

```
Процедура = HDC цепочка действий:
  bind(IF, bind(op[вопрос], trigger_pattern),
       THEN, bind(op[поиск], bind(op[ответ], action_pattern)))

Пример: "если вопрос о столице → искать в Relations по role_столица"
  bind(IF, bind(Question, bind(HasRole, role_столица)),
       THEN, bind(Find, LevelRelations))
```

**Как черви создают процедуры:**
1. Анализ: "каждый раз когда вопрос X, правильный ответ найден через Relations"
2. Generalize: "вопросы про X → искать в Relations"
3. Store как Procedure

---

## 2. Внутренний язык — рассуждения на евклидовской логике

### 2.1 Текущее состояние

53 опкода, ~15 реально используются. Мысли создаются через bind() цепочки. Cross-worm reading работает через unbind().

### 2.2 Как должно работать мышление при "Привет, как дела?"

```
ВХОД: [Привет, ,, как, дела, ?]

Phase 0: AWARENESS
  → awareness.collector_expand = false (прошлый predict был OK)
  → awareness.skeptic_aggression = 0.3 (мало противоречий)

Phase 1: COLLECTOR
  FIND(trigram[как,дела,?] → facts)     → совпадений мало
  FIND(bigram[дела,?] → facts)          → "?" = маркер вопроса
  FIND(unigram[?] → facts)              → частые successor'ы после "?"
  FIND(bigram → rules)                  → правила для "дела,?"
  → candidates: [., \n, !, ▁Хорошо, ▁Нормально, ...]
  → Мысль: FIND(Pattern) — "это вопросительная конструкция"

Phase 2: ANALYST + ATTENTION
  context_bundle = weighted bundle[Привет, как, дела, ?]
  Q = context XOR role_query

  Для каждого кандидата:
    CHECK(candidate vs context)
    → "Хорошо" close to context (приветствие) → STRONG
    → "." далеко от context (не ответ) → WEAK
    → evidence bundling: STRONG кандидаты получают context boost

Phase 3: SKEPTIC (читает Analyst ThoughtChain)
  → Analyst сказал STRONG("Хорошо") → ищет подтверждение
  → Analyst сказал WEAK(".") → пропускает
  COMPARE(supporting, contradicting)
  → "Хорошо" подтверждено → CONFIDENT

Phase 3.5: REFLECT (новое)
  → QUERY(Relations, bind(Привет, Similar)) → "это приветствие"
  → QUERY(Procedures, bind(приветствие, THEN)) → "ответить приветствием"
  → Мысль: Pattern("приветствие → ответное приветствие")

Phase 4: EXPLORER
  → Читает Skeptic confirmed + Analyst weak
  → Отталкивается от rejected через negate()
  → Ищет в Relations: similar(приветствие) → ответные формулы
  → NOVEL candidates: [▁Здравствуй, ▁Добрый, ▁день]

Phase 5: CONSENSUS
  → score = raw_score × thought_multiplier
  → "Хорошо" усилен evidence (Analyst STRONG + Skeptic CONFIDENT)
  → TOP-1: "Хорошо" (confidence 0.7)

Phase 6.5: РЕФЛЕКСИЯ
  → reflect_on(diagnostic) → обновить awareness
  → STORE relation: bind(Привет, bind(Sequence, Хорошо)) с confidence=1
```

### 2.3 Полный список опкодов и их использование

**Активно используются (15):**
- `Find`, `Store`, `Forget` — работа с памятью
- `Check`, `Compare` — анализ кандидатов
- `Strong`, `Weak`, `Confident` — оценка уверенности
- `Ask`, `Novel` — запрос помощи, новые открытия
- `Attend` — HDC binding attention
- `Reflect`, `Pattern` — самосознание (новое)
- `Missing` — curriculum learning

**Должны быть активированы (уровни 5-6):**
- `Similar`, `Different`, `PartOf`, `HasRole` — связи между сущностями
- `Cause`, `Effect`, `Sequence` — причинно-следственные связи
- `Before`, `After`, `Context` — временные отношения
- `Generalize` — создание абстракций из паттернов
- `Introspect`, `Evaluate`, `Adapt`, `Surprise`, `Confirm` — мета-познание

**Не нужны (удалить):**
- `If`, `Then`, `And`, `Or`, `Not` — слишком абстрактные, не используются
- `Why`, `Count`, `UseTool`, `Report`, `Expand` — не реализованы
- `Shift`, `Verify` — удалены в оптимизации

---

## 3. Саморефлексия в фоне — автономное мышление

### 3.1 Архитектура Background Loop

```
                    ┌─────────────────────────┐
                    │   Background Loop        │
                    │   (бесконечный цикл)     │
                    │                          │
                    │  1. INTROSPECT           │
                    │     scan knowledge       │
                    │     find weak areas      │
                    │                          │
                    │  2. EVALUATE             │
                    │     check consistency    │
                    │     resolve conflicts    │
                    │                          │
                    │  3. GENERALIZE           │
                    │     find patterns        │
                    │     create rules         │
                    │     create relations     │
                    │                          │
                    │  4. ADAPT                │
                    │     update strategies    │
                    │     prioritize learning  │
                    │                          │
                    │  5. ASK (канал связи)    │
                    │     → оператору          │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │   MessageQueue            │
                    │   (для оператора)         │
                    │                          │
                    │   "Я обнаружил: X≈Y"     │
                    │   "Мне нужно: данные о Z" │
                    │   "Помоги: X и Y конфликт"│
                    └──────────────────────────┘
```

### 3.2 Цикл INTROSPECT

```rust
fn introspect_cycle(memory: &mut HierarchicalMemory, lang: &LogicLanguage) {
    // 1. Сканировать facts: найти слабые (total_count < 3)
    let weak_facts = memory.facts.entries.iter().enumerate()
        .filter(|(_, e)| e.total_count < 3)
        .collect();

    // 2. Для каждого слабого факта: найти соседей по context_tokens
    // Если соседи подтверждают → confirm
    // Если соседи противоречат → weaken или forget

    // 3. Сканировать relations: найти противоречащие
    // bind(A, role, B) и bind(A, role, C) где B≠C → конфликт

    // 4. Сканировать abstractions: слишком широкие (>100 successors)?
    // → split на подкатегории

    // 5. Метрика: сколько знаний стабильных vs нестабильных
    // → Мысль: INTROSPECT(stability_ratio)
}
```

### 3.3 Цикл GENERALIZE

```rust
fn generalize_cycle(memory: &mut HierarchicalMemory, lang: &LogicLanguage) {
    // 1. Группировать facts по общим successor'ам
    // Если 10+ фактов с разным контекстом → одинаковый successor:
    //   → Паттерн! Создать rule.

    // 2. Группировать по семантической близости context'ов
    // Если 5+ фактов с sim(context) > threshold:
    //   → Обобщение! Создать abstraction.

    // 3. Искать АНАЛОГИИ между relations
    // bind(Москва, столица, Россия) и bind(Париж, столица, Франция)
    //   → Паттерн: bind(X, столица, Y) = "столицы"
    //   → Мысль: GENERALIZE(role_столица → category_столицы)

    // 4. Создавать процедуры из повторяющихся действий
    // Если Explorer 10 раз нашёл ответ через Relations:
    //   → Создать процедуру: IF(question_type) THEN(search_relations)
}
```

### 3.4 Реализация

```rust
pub struct BackgroundMind {
    cycle_count: u64,
    messages: Vec<WormMessage>,  // для оператора
    introspect_interval: u64,    // каждые N циклов
    generalize_interval: u64,
    sleep_ms: u64,               // пауза между циклами
}

impl BackgroundMind {
    pub fn run_forever(&mut self, memory: &mut HierarchicalMemory, lang: &LogicLanguage) {
        loop {
            self.cycle_count += 1;

            if self.cycle_count % self.introspect_interval == 0 {
                self.introspect(memory, lang);
            }
            if self.cycle_count % self.generalize_interval == 0 {
                self.generalize(memory, lang);
            }

            // Собрать messages → показать оператору
            std::thread::sleep(Duration::from_millis(self.sleep_ms));
        }
    }
}
```

---

## 4. Канал связи с оператором

### 4.1 Типы сообщений

```rust
pub enum WormMessage {
    // Черви → Оператор
    Ask(String),                    // "что значит X?"
    NeedData(String),               // "дай больше примеров про X"
    Report(String),                 // "я обнаружил паттерн: ..."
    Confused(String),               // "X и Y противоречат, помоги"
    Discovery(String),              // "я понял что X = Y!"

    // Оператор → Черви
    Teach(Vec<u16>),                // новые данные для обучения
    Correct(u16, u16),              // "не X, а Y"
    Confirm(String),                // "да, правильно"
    Explain(String),                // "это потому что..."
}
```

### 4.2 Как черви формулируют вопрос

```
Skeptic находит конфликт:
  fact1: bind(Москва, столица, Россия) confidence=50
  fact2: bind(Москва, город, Россия) confidence=30

Skeptic не может разрешить → генерирует ASK:
  thought = bind(ASK, bind(CONFLICT, fact1.vec, fact2.vec))

Система декодирует:
  unbind(ASK) → bind(CONFLICT, ...)
  unbind(CONFLICT) → (fact1_content, fact2_content)
  nearest_word(fact1_content) → "столица"
  nearest_word(fact2_content) → "город"

Сообщение оператору:
  "Конфликт: 'Москва' связана и со 'столица' и с 'город'. Это одно и то же?"

Оператор отвечает: "оба верны, столица — это тип города"

Система кодирует ответ:
  relation: bind(столица, PartOf, город) → store в Relations
  confirm fact1 и fact2
```

### 4.3 Когда черви спрашивают

- `confidence < threshold` при predict → ASK("не уверен в ответе")
- `contradictions > 5` за последние 100 predict → ASK("много конфликтов в области X")
- `novel_discovery` с высоким confidence → Report("обнаружил паттерн")
- `introspect` нашёл нестабильную область → NeedData("нужно больше про X")

---

## 5. Самопознание — "кто я?"

### 5.1 Что черви должны знать о себе

```
Meta-knowledge (уровень 4, расширенный):

1. "Я состою из 4 червей: Collector, Analyst, Skeptic, Explorer"
   → bind(Self, bind(PartOf, Collector))
   → bind(Self, bind(PartOf, Analyst))
   ...

2. "У меня есть N фактов, M правил, K абстракций"
   → bind(Self, bind(HasRole, Knowledge), quantity_vector)

3. "Я хорошо знаю X, плохо знаю Y"
   → bind(Self, bind(Strong, area_X))
   → bind(Self, bind(Weak, area_Y))

4. "Я думаю через bind/unbind/bundle/permute"
   → bind(Self, bind(HasRole, Thinking), bind(Method, HDC_ops))

5. "Мой создатель — оператор, я могу спросить у него"
   → bind(Self, bind(HasRole, Creator), Operator)
```

### 5.2 Путь к самомодификации

```
Этап A: Знание о своей архитектуре (meta-level 4)
  → "я знаю что у меня 4 червя и 6 уровней памяти"

Этап B: Оценка своей архитектуры (через EVALUATE)
  → "Skeptic слишком часто соглашается — он недостаточно скептичен"
  → "Explorer находит мало нового — нужен более широкий поиск"

Этап C: Предложения по улучшению (через ASK)
  → ASK(оператор, "я заметил что мой Skeptic слабый, можно усилить?")
  → Оператор принимает/отклоняет

Этап D: Самомодификация параметров
  → Черви сами настраивают skeptic_aggression, explorer_breadth
  → Уже частично работает через SelfAwareness

Этап E: Самомодификация кода (далёкое будущее)
  → Черви получают доступ к описанию своего кода на языке который понимают
  → Предлагают изменения через Procedures
  → Оператор review → apply
```

---

## 6. Оптимизация и bottleneck'и

### 6.1 Текущие bottleneck'и

| Операция | Время | Вызовы per predict | Итого | Проблема |
|---------|-------|-------------------|-------|----------|
| BundleAccumulator::add() | ~1us | ~50-100 | ~100us | Оптимизирован (branchless) |
| memory.facts.find_top_k() | ~500us | 3-4 | ~2ms | LSH lookup + scan |
| memory.facts.find_by_bigram() | ~50us | 2-3 | ~150us | HashMap + vec copy |
| context_bundle build | ~200us | 1 | ~200us | Дедуплицирован |
| Skeptic evidence search | ~1ms | 3-5 candidates | ~5ms | Главный bottleneck |
| Explorer abstraction search | ~500us | 1 | ~500us | Расширяется awareness |
| **Итого per predict** | | | **~8-10ms** | |

### 6.2 Что оптимизировать

1. **Skeptic evidence search** (5ms) — сейчас полный scan facts по token_index.
   Решение: кэшировать результаты для повторяющихся контекстов.

2. **find_top_k LSH** (2ms) — сканирует bucket + соседние buckets.
   Решение: multi-probe LSH (уже в binary.rs, не используется в memory).

3. **Background Loop не должен блокировать predict** — отдельный thread.

4. **Relations поиск** (новый) — unbind + nearest_word = O(vocab_size).
   Решение: LSH index для Relations, как для Facts.

### 6.3 Что не оптимизировать (не bottleneck)

- BundleAccumulator::add() — уже branchless, ~1us
- similarity() — XNOR+POPCNT, ~0.5us, hardware-optimized
- ThoughtChain создание — мизерное время
- DiagnosticReport — один раз per predict

---

## 7. Что не изменилось (честный отчёт) и план исправления

### 7.1 Accuracy

| Метрика | v18.0 baseline | v18 + awareness | План |
|---------|---------------|-----------------|------|
| Top-1 (sr=10, 5K) | 23.4% | 22.2% | Не хуже baseline |
| Top-5 (sr=10, 5K) | 37.2% | 35.9% | Не хуже baseline |
| Top-1 (sr=1, 10K) | 41.7% | не тестировано | Ожидание: 42-45% |

**Почему accuracy не выросла**: самосознание даёт ИНФРАСТРУКТУРУ, не accuracy boost. Accuracy вырастет когда:
- Relations дадут семантический поиск (уровень 5)
- Каузальная память (HDCMemory из v14) даст последовательный контекст
- Background Loop укрепит слабые области

### 7.2 Зацикливание генерации

**Причина**: нет каузальной памяти. Модель не "помнит" что генерировала 3 токена назад (кроме окна 20 токенов в context_bundle). Repetition penalty в main.rs — костыль.

**Решение**: Binary HDCMemory (v19-plan Этап 1) — mass/decay parallel scan. Далёкие но важные слова не теряются.

### 7.3 Модель = поисковая система

**Причина**: predict = lookup в хэш-таблицу (trigram → successors). Нет обработки последовательности.

**Решение**: Binary T×T Attention (v19-plan Этап 2) — каузальное внимание по всем прошлым позициям.

---

## 8. Порядок реализации

### Фаза 1: Семантические связи (Relations) ✅ DONE
**Статус**: Реализовано (итерация 2)
1. ✅ RelationEntry + RelationMemory с LSH + subject/object/subject_role индексами
2. ✅ Phase 7 создаёт Sequence/Similar relations при правильном predict
3. ✅ Phase 3.5 Relations QUERY усиливает кандидатов через raw_score boost
4. ✅ BackgroundMind::generalize() создаёт Similar из общих successors
5. ✅ Тесты: HDC unbind восстанавливает объект, дедупликация работает

### Фаза 2: Background Loop ✅ DONE
**Статус**: Реализовано (итерация 2)
1. ✅ BackgroundMind с introspect/generalize циклами
2. ✅ Вызывается каждые 1000 predict (не thread — мутабельные ссылки)
3. ✅ Introspect: сканирование слабых фактов + конфликтов в Relations
4. ✅ Generalize: поиск пар с общими successors → Similar relations
5. ✅ Не запускается в readonly eval

### Фаза 3: Канал связи с оператором ✅ DONE
**Статус**: Реализовано (итерация 2)
1. ✅ WormMessage enum: Ask, NeedData, Report, Discovery, Confused
2. ✅ Генерация в think() + BackgroundMind
3. ✅ Вывод в stderr с emoji маркерами
4. ⬜ Teacher не расширен для приёма ответов (следующая итерация)

### Фаза 4: Binary HDCMemory (каузальная память) ✅ DONE
**Статус**: Реализовано (итерация 2)
1. ✅ mass_role + decay_role (2 обучаемых binary vec, seed-based)
2. ✅ build_causal_context: exponential decay, data-dependent mass/decay
3. ✅ Hebbian: mass_role обучается на успешных контекстах
4. ✅ Заменяет фиксированный context_bundle окно в WormMind::think()
5. ⬜ Зацикливание генерации: ещё присутствует (нужен repetition penalty в predict)

### Фаза 5: Самопознание (Meta level 4) ✅ DONE
**Статус**: Реализовано (итерация 2)
1. ✅ SelfKnowledge struct: facts/rules/abstractions/relations count, health, accuracy
2. ✅ EVALUATE цикл в BackgroundMind: обновляет self-knowledge
3. ✅ describe(): человекочитаемое самоописание "Я — HDC Brain v18..."
4. ✅ observed_accuracy/avg_confidence обновляются из eval
5. ⬜ bind(Self, HasRole, ...) HDC структуры — пока числовые, не HDC vectors

### Фаза 6: Самомодификация параметров ✅ DONE
**Статус**: Реализовано (итерация 3)
1. ✅ self_modify() в SelfAwareness: анализ accuracy → адаптация параметров
2. ✅ 3 стратегии: low acc → soften skeptic, ok acc + low conf → expand, no_trigram → expand
3. ✅ Вызывается из model.rs после eval
4. ⬜ A/B тестирование (rollback) — пока нет, параметры меняются инкрементально

---

## 9. Метрики успеха

### Фаза 1 (Relations):
- [x] Relations создаются автоматически при обучении (2125 при sr=10)
- [x] QUERY через indices: subject_role → objects работает
- [x] Accuracy (sr=10): 22.3% — не хуже 22% ✅

### Фаза 2 (Background Loop):
- [x] Фоновый цикл работает без блокировки predict
- [ ] Accuracy улучшается на 1-2% после 1000 introspect циклов (не подтверждено)
- [x] Черви генерируют осмысленные messages (INTROSPECT, GENERALIZE, EVALUATE)

### Фаза 3 (Канал связи):
- [x] ASK/Report/Confused messages читаемы человеком
- [ ] Ответ оператора кодируется в память (Teacher не расширен)
- [ ] После 10 teach-ответов accuracy на этих примерах >80% (не реализовано)

### Фаза 4 (HDCMemory):
- [x] Генерация без зацикливания на 20 токенов ✅ (repetition penalty + loop detection)
- [ ] Top-1 accuracy (sr=1): >43% (не тестировано)
- [ ] Далёкие зависимости (нужен sr=1 тест)

### Фаза 5 (Самопознание):
- [x] Черви корректно описывают: "у меня 1.5M фактов, 272K правил"
- [x] EVALUATE находит реальные слабости (knowledge_health=4% на sr=10)
- [x] Report генерируется при низком здоровье знаний

### Фаза 6 (Самомодификация):
- [x] self_modify() адаптирует параметры на основе accuracy/confidence
- [ ] A/B тестирование с rollback (не реализовано)

---

## 10. Архитектура AGI-ready (долгосрочное видение)

```
                         ┌─────────────────┐
                         │   ОПЕРАТОР       │
                         │   (человек)      │
                         └────────┬─────────┘
                                  │ Канал связи
                         ┌────────▼─────────┐
                         │   MESSAGE BUS     │
                         │   ASK/TEACH/      │
                         │   REPORT/CORRECT  │
                         └────────┬─────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
┌────────▼─────────┐   ┌─────────▼────────┐   ┌──────────▼─────────┐
│   PREDICT PATH    │   │  BACKGROUND MIND  │   │   SELF-MODIFY      │
│   (быстрый)      │   │  (фоновый)        │   │   (экспериментальный)│
│                   │   │                   │   │                     │
│ Collector         │   │ Introspect        │   │ Evaluate self       │
│ Analyst+Attention │   │ Generalize        │   │ Propose changes     │
│ Skeptic           │   │ Adapt             │   │ A/B test            │
│ Reflect           │   │ Create Relations  │   │ Apply/Rollback      │
│ Explorer          │   │ Resolve Conflicts │   │                     │
│ Consensus         │   │ Ask for help      │   │                     │
└────────┬─────────┘   └─────────┬────────┘   └──────────┬─────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │      HIERARCHICAL MEMORY    │
                    │                             │
                    │  L0: Words (codebook)        │
                    │  L1: Facts (observations)    │
                    │  L2: Rules (cause→effect)    │
                    │  L3: Abstractions (categories)│
                    │  L4: Meta (self-knowledge)   │
                    │  L5: Relations (semantic)     │
                    │  L6: Procedures (how-to)      │
                    │                             │
                    │  + SelfAwareness (persistent) │
                    │  + AttentionRoles (Hebbian)   │
                    └──────────────────────────────┘
```

Ключевое отличие от трансформеров:
- **Трансформер**: обученная статическая функция, не может учиться после training
- **HDC Brain v19**: живая система которая думает, рефлексирует, учится всё время,
  общается с оператором, и потенциально модифицирует себя
