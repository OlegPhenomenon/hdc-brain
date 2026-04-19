# v18 Optimization Changes — итерация 1

## Дата: 2026-04-03

## Проблемы найденные при анализе

### КРИТИЧЕСКИЕ — холостая работа

1. **Evidence multiplier почти не работал** (`worm_mind.rs`)
   - Было: `multiplier = 1.0 + evidence_strength.clamp(-0.5, 0.5)` → [0.5, 1.5]
   - Мысли червей через evidence bundling давали ~±25% коррекцию к raw_score
   - Но raw_score у кандидатов различается в 8× → мысли НЕ МОГЛИ переранжировать
   - **Решение**: расширенный multiplier с attention + forward + reinforcement
   - Формула: `score = raw_score × (1.0 + evidence±0.5 + attn+0.25 + fwd+0.15 + reinf)`
   - **Результат**: confidence 0.24 → 0.39 (+63%) — мысли реально влияют

2. **Двойной context bundle** — строился 2 раза (в HdcBindingAttention::build() и Analyst)
   - **Решение**: строим ОДИН РАЗ, передаём как параметр в HdcBindingAttention::build()

3. **attention.rs полностью мёртвый** (694 строки float v14 port) → удалён

4. **ThoughtQueue** никогда не использовался → удалён

5. **11+ методов** никогда не вызывались → удалены

### Исследования и отвергнутые подходы

1. **Hebbian codebook при КАЖДОМ predict**: деградирует codebook при accuracy <50%
   (anti-Hebbian updates доминируют). Оставлен только correct-predict Hebbian.

2. **Codebook update во время eval**: даже correct-only Hebbian при eval деградирует
   codebook за 5K-10K позиций. Отключен для eval (learn_codebook=false).

3. **Аддитивная формула evidence**: `score = norm_raw*0.4 + evidence*0.3 + attn*0.2`
   хуже multiplier на ~1.5% — слабые кандидаты получают слишком большой boost.

4. **Recency weighting баг**: при дедупликации context_bundle потерялись веса
   {0-2: 4, 3-5: 3} → {0-2: 3}. Исправлено.

## Оптимизации

### BundleAccumulator::add() — branchless
- Было: побитовый цикл с branching per bit (4096 branches per add())
- Стало: word-level shift + branchless `(bit*2-1)`, ~3× меньше инструкций
- to_binary() тоже оптимизирован: прямое построение u64 words

### Hebbian Codebook Update (из v19-plan)
- Codebook обучается при правильном predict: `codebook[exp] = bundle(old × 8, context)`
- Anti-Hebbian ОТКЛЮЧЕН: при accuracy <50% разрушает codebook
- Включается только с флагом learn_codebook=true (training eval)
- evaluate_mind_readonly() — eval без обновления codebook

### API: learn_codebook параметр
- WormMind::think() принимает learn_codebook: bool
- evaluate_mind() → learn_codebook=false (безопасный default)
- evaluate_mind_readonly() → всегда false

## Удалённый код
- `src/attention.rs` — 694 строки
- `ThoughtQueue`, 11 методов LogicLanguage
- `EvalResult`, `evaluate()`, `predict_mind()`, `predict_with_context()`
- `BundleAccumulator::reset()`, `MemoryEntry::confidence()/weaken()`
- `MemoryLevel::find_weak()/is_empty()`, `Config::fact_lsh_bits`

## Результаты (sr=10, 27M tokens)

| Метрика | Baseline (v18.0) | After optimization |
|---------|------------------|--------------------|
| Top-1 (5K) | 23.4% | 22.4% (-1%) |
| Top-5 (5K) | 37.2% | 36.1% (-1.1%) |
| Confidence | 0.24 | **0.39** (+63%) |
| Thoughts effect | минимальный | **значимый** |
| Tests | 29/29 | 29/29 |
| Warnings | 25 | 15 |

**Вывод**: ~1% регрессия на sr=10, но мысли червей теперь РЕАЛЬНО влияют на результат
(confidence +63%). На sr=1 (полные данные) ожидается улучшение т.к. evidence signals
качественнее. Нужен тест на sr=1 для подтверждения.

## Следующие шаги
1. Тест на sr=1 (~2 часа) для валидации на полных данных
2. Binary HDCMemory (каузальная память из v19-plan)
3. Fast/Slow path (skip WormMind при confidence > threshold)
