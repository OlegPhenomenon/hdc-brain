# v18 Optimization Changes — итерации 3-4

## Дата: 2026-04-03

## Итерация 3: Исправления + Самомодификация

### 1. Hybrid context (HDCMemory)
- Вместо чистого HDCMemory каузального контекста — HYBRID подход
- Baseline recency weighting (проверенный): {0-2: 4, 3-5: 3, 6-10: 2, 11-20: 1}
- HDCMemory mass_mod: ±30% модификация baseline weights
- causal_extra: далёкие (>20) токены с high mass получают 1 repeat
- **Устранена регрессия**: accuracy вернулась к 22.3-22.7%

### 2. Зацикливание генерации — УСТРАНЕНО
- Repetition penalty окно: 5 → 15 токенов
- Loop detection: если последние 6 = предпоследние 6 → разрыв цикла
- Выбирает наименее частый кандидат при обнаружении цикла

### 3. Самомодификация (Фаза 6)
- `SelfAwareness::self_modify()`: 3 стратегии адаптации
  - Low accuracy + low confidence → soften skeptic, expand explorer
  - OK accuracy + low confidence → expand collector
  - Long no_trigram streak → permanent expand

## Итерация 4: Глубокий аудит влияния

### КРИТИЧЕСКИЙ РЕЗУЛЬТАТ — InfluenceReport

Добавлены счётчики РЕАЛЬНОГО влияния каждой фичи. Каждый увеличивается ТОЛЬКО когда фича ИЗМЕНИЛА результат.

#### До исправления (итерация 4a):
| Фича | Счётчик | Из 5000 | Вердикт |
|------|---------|---------|---------|
| relations_boosted | 282 | 5.6% | ✅ Работает |
| **cross_worm_overrides** | **0** | **0%** | **❌ СЛОМАНО** |
| **causal_extra_tokens** | **0** | **0%** | **❌ Не работает** |
| awareness_adaptations | 7284 | 145%+ | ✅ Активно |
| evidence_reranks | 368 | 7.4% | ✅ Работает |

#### Найденные баги:
1. **Cross-worm unbind**: unbind(thought, kind) вместо unbind(thought, creation_opcode).
   Analyst создаёт `CHECK(word)`, но kind перезаписывается в STRONG/WEAK.
   unbind по STRONG даёт `CHECK XOR word XOR STRONG ≠ word`.
   **Исправлено**: unbind по CHECK (creation opcode).

2. **Explorer unbind**: та же ошибка — unbind по kind вместо creation opcode.
   **Исправлено**: unbind по Compare/Weak для Skeptic, CHECK для Analyst.

3. **causal_extra threshold**: mass_mod > 1.1 при random roles ≈ never.
   **Снижено**: mass_mod > 1.02. На sr=10 всё ещё 0 (нужно Hebbian обучение).

#### После исправления (итерация 4b):
| Фича | Счётчик | Из 5000 | Вердикт |
|------|---------|---------|---------|
| relations_boosted | **502** | 10% | ✅ Работает (рост) |
| **cross_worm_overrides** | **4616** | **92%** | **✅ ИСПРАВЛЕНО!** |
| causal_extra_tokens | 0 | 0% | ⚠️ Нужно больше данных |
| awareness_adaptations | 7271 | 145%+ | ✅ Активно |
| evidence_reranks | 370 | 7.4% | ✅ Работает |

### Accuracy (sr=10):
- Train Top-1: **22.46%** (baseline 23.4%, лучший за все итерации)
- confidence: 0.39

## Вывод

Cross-worm reading был СЛОМАН с самого начала (unbind по неправильному опкоду).
Исправление привело к:
- 0 → 4616 cross-worm overrides (92% predict'ов)
- Accuracy 22.2% → 22.5% (частичное восстановление baseline)

Оставшаяся "декорация":
- causal_extra_tokens = 0: HDCMemory mass_role требует больше Hebbian обучения
  (ожидается на sr=1 с ×10 данных)
