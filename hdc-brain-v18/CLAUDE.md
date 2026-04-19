# HDC-Brain v18 — Инструкции

## Критические правила (НЕ НАРУШАТЬ)

### HDC ≠ Трансформер
- v14 НЕ содержит ни одного компонента трансформера
- НЕТ softmax, НЕТ learned position embeddings, НЕТ projection matrices W_q/W_k/W_v
- HDC Binding Attention = element-wise multiply с role vectors (3×D параметров, не 3×D²)
- Sigmoid вместо softmax. Cyclic permutation вместо positional embeddings.
- **Прочитай docs/architecture/v14-hdc-brain.md перед любой работой с attention**

### Время выполнения
- sample_rate=1 на 27M токенов = ~35 мин тренинг + ~55 мин worm cycle + ~30 мин eval = **~2 часа**
- sample_rate=10 = ~3 минуты total
- **ВСЕГДА сначала тестируй на sr=10**, sr=1 только после подтверждения что sr=10 работает
- **Честно предупреждай** о реальном времени выполнения

### Не галлюцинировать
- Не описывай код который не читал
- Не придумывай архитектуру — читай файлы
- Если не помнишь — перечитай, не выдумывай

## Архитектура

### Документация (читай перед работой)
- `docs/architecture/v14-hdc-brain.md` — архитектура v14 (HDC Binding Attention, ThoughtLoop, HDCMemory)
- `docs/architecture/worm-mind.md` — архитектура WormMind v18

### Ключевые файлы
- `src/worm_mind.rs` — WormMind: черви думают при predict + HdcBindingAttention
- `src/worms.rs` — Collector/Analyst/Skeptic/Explorer + WormOrchestrator (тренировка)
- `src/model.rs` — HDCBrainV18: train(), evaluate_mind(), predict()
- `src/memory.rs` — HierarchicalMemory: facts/rules/abstractions + indices
- `src/binary.rs` — BinaryVec: XOR/XNOR/POPCNT/permute/bundle
- `src/language.rs` — 38 опкодов внутреннего языка червей
- `src/attention.rs` — порт v14 float attention (не интегрирован)
- `src/working_memory.rs` — Working Memory (evidence pipeline, model level)
- `src/logger.rs` — файловое логирование

### v14 проект (эталон)
- `/Users/oleghasjanov/Documents/learning/hoffman_swarm/hdc-brain-v14/hdc_brain_v14.py`
- Тренируется на vast.ai, BPB 3.896
- HDC Binding Attention через role vectors (bv_q, bv_k, bv_v)
- ThoughtLoop: 4 прохода через те же блоки
- HDCMemory: mass/decay parallel scan

## Метрики

### v18.1 (текущая, sr=1 с attention)
- Train Top-1: 41.7%, Top-5: 60.5%
- Final Top-1: 40.7%, Top-5: 59.5% (10K eval)
- Memory: 8.5M facts, 1.57M rules, 932K abstractions

### Baseline (sr=10 без attention)
- Top-1: 23.4%, Top-5: 37.2%

## Запуск
```bash
# Быстрый тест (3 мин)
cargo run --release -- train --data ../hdc-brain-v15/data_big_bpe.bin --vocab ../hdc-brain-v15/vocab_16k.txt --sample-rate 10 --log test.log

# Полный запуск (~2 часа)
cargo run --release -- train --data ../hdc-brain-v15/data_big_bpe.bin --vocab ../hdc-brain-v15/vocab_16k.txt --log full.log

# Тесты
cargo test --release
```
