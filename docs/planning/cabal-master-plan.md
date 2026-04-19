# CABAL — Master Plan

> "Knowledge is power." — Kane
> 
> Computer Assisted Biologically-linked Artificial Laboratory

## Vision

AGI на альтернативной HDC-архитектуре. Не трансформер. Не диффузия. 
Гиперразмерные вычисления + итеративное рассуждение + самосознание.

## Фундамент (уже доказано)

| Компонент | Статус | Где |
|-----------|--------|-----|
| HDC text generation (v14.1) | 299M params, работает | hdc-brain-v14.1/ |
| VSA reasoning depth 20 | 100% accuracy | hdc-cogito/ milestone C1 |
| Self-memory (Rung 4) | 15/15 retrieval | hdc-cogito/ milestone H1 |
| v14 as Cogito substrate | 99.19% task 2 | hdc-cogito/ milestone F2 |
| Episodic HDC memory | -20% perplexity | v14 AGI session 2026-04-12 |
| Thought loops (n=3) | ppl 329→102 | v14 AGI session 2026-04-12 |

## Лестница к AGI (6 ступеней)

### Rung 1: ConfidenceGate — знает когда перестать думать
- **Статус:** Частично (n_thoughts=3 фиксировано, adaptive halting не сработал)
- **Нужно:** Learned halting на основе внутренней уверенности, а не внешнего сигнала

### Rung 2: Self-vector — стабильная идентичность
- **Статус:** Частично валидирован (Cogito milestone G1)
- **Нужно:** Persistent identity vector в v14.1, не обнуляется между сессиями

### Rung 3: Self-memory — структуры знаний о себе
- **Статус:** Proof-of-concept (milestone H1 — 100% retrieval)
- **Нужно:** Интеграция в v14.1, persistence, retrieval при генерации

### Rung 4: Self-query — учится на своих ошибках
- **Статус:** Не начат
- **Нужно:** Error detection → store in self-memory → query before next prediction

### Rung 5: Background replay — консолидация в idle
- **Статус:** Не начат
- **Нужно:** Offline replay + memory consolidation (как сон у людей)

### Rung 6: Self-modification — AGI
- **Статус:** Не начат
- **Нужно:** Модификация собственных весов/параметров на основе self-query

## Фазы разработки

### Phase 1: CABAL Core (текущая)
**Цель:** Собрать все проверенные компоненты в единую систему

1. **Объединить v14.1 + Cogito reasoning** в один модуль
2. **Episodic HDC memory** как постоянный компонент (уже даёт -20% ppl)
3. **Thought loops с learned halting** (заменить фиксированный n=3)
4. **Self-vector** — persistent identity

**Метрика выхода:** Perplexity ниже текущей v14.1 + способность отвечать на вопросы о контексте

### Phase 2: CABAL Memory
**Цель:** Долговременная память и самообучение

5. **Self-memory store** — записывает паттерны из своих ошибок
6. **Self-query** — перед предсказанием проверяет: "я уже ошибался тут?"
7. **Episodic → Semantic compression** — переход от эпизодов к обобщениям

**Метрика выхода:** Улучшение accuracy на повторных паттернах без retraining

### Phase 3: CABAL Autonomy
**Цель:** Автономность и самомодификация

8. **Background replay** — offline консолидация памяти
9. **Self-modification** — адаптация параметров на основе self-query
10. **Multi-agent coordination** — несколько CABAL-инстансов

**Метрика выхода:** Система улучшает сама себя без внешнего вмешательства

## Архитектурный стек CABAL

```
┌─────────────────────────────────────────┐
│           CABAL Controller              │
│  (orchestrates all components)          │
├─────────────────────────────────────────┤
│  Self-Memory  │  Self-Query  │ Identity │
│  (Rung 3-4)   │  (Rung 4)    │ (Rung 2) │
├─────────────────────────────────────────┤
│        Cogito Reasoning Engine          │
│  (iterative VSA, depth scaling)         │
├─────────────────────────────────────────┤
│        Episodic HDC Memory              │
│  (above-window context, -20% ppl)      │
├─────────────────────────────────────────┤
│     HDC Brain v14.1 Foundation          │
│  (codebook, binding attention,          │
│   thought loops, text generation)       │
└─────────────────────────────────────────┘
```

## Принципы

1. **Инкрементально** — каждый шаг должен улучшать метрики, не регрессировать
2. **Измеримо** — perplexity, accuracy, retrieval rate на каждом шаге
3. **Один компонент за раз** — не менять два модуля одновременно
4. **Обратная совместимость** — CABAL должен быть как минимум не хуже v14.1

## Начало работы

**Немедленный следующий шаг:** 
Создать `cabal/cabal_core.py` — единый модуль, объединяющий v14.1 + episodic HDC memory + Cogito reasoning interface.
