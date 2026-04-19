# WormMind Architecture

## Overview

WormMind = черви ДУМАЮТ при predict. Не просто ищут в памяти — рассуждают на внутреннем языке, проверяют себя, и учатся попутно.

## Pipeline

```
Input: tokens[..pos] → "какой следующий токен?"

1. COLLECTOR (3-5 шагов)
   FIND(trigram → facts)     → top candidates by exact match
   FIND(bigram → facts)      → always, not just fallback
   FIND(unigram → facts)     → if few candidates
   FIND(bigram → rules)      → Explorer's learned rules
   → Передаёт top-20 кандидатов

2. ANALYST (20-30 шагов)
   Для каждого кандидата:
     CHECK(candidate vs context_vector)
     IF(high_score AND context_compatible) → STRONG(conf=0.7+)
     IF(moderate_score) → CHECK(conf=0.3-0.8)
     IF(low_score) → WEAK(conf=0.05)

3. SKEPTIC (5-10 шагов)
   Для top-3:
     ASK(counter-evidence)
     Ищет факты с тем же контекстом но ДРУГИМ successor
     COMPARE(supporting, contradicting)
     IF(ratio > 0.6) → укрепить confidence × 1.3
     IF(ratio < 0.3) → снизить confidence × 0.5

   Background learning:
     IF(ratio > 0.7) → confirm_successor (укрепить факт)
     IF(ratio < 0.2) → FORGET (ослабить ошибочный факт)

4. EXPLORER (1-5 шагов)
   FIND(abstraction → novel candidates)
   FIND(medium context → confirmation)
   IF(top candidate confident) → STORE(new abstraction)

5. CONSENSUS
   final_score = Σ(score × confidence) для каждого кандидата
   → top-10 отсортированных
```

## Internal Language Usage

Opcodes реально используемые WormMind:
- `FIND` — поиск в памяти (facts, rules, abstractions)
- `CHECK` — проверка кандидата по контексту
- `COMPARE` — сравнение support vs contradict
- `IF/THEN` — условная логика
- `STRONG/WEAK` — оценка уверенности
- `ASK` — запрос контр-доказательств
- `NOVEL` — новый кандидат из абстракций
- `STORE` — создание новой абстракции
- `FORGET` — ослабление ошибочного факта

## Background Learning

При каждом predict черви попутно обновляют долгосрочную память:
- Skeptic укрепляет подтверждённые факты
- Skeptic ослабляет опровергнутые
- Explorer создаёт абстракции для уверенных предсказаний

За 5000 eval positions = ~2500 background updates.

## HDC Binding Attention (v18.1)

Три механизма внимания интегрированы в фазу Analyst:

1. **Semantic Attention** — контент-адресный взвешенный bundle
   - Каждый токен контекста взвешен по similarity к anchor (последние 2 токена)
   - Далёкие но семантически связанные слова получают больший вес
   - Заменяет простой BundleAccumulator с recency weighting

2. **Shift Prediction** — последовательная инерция через XOR-шифты
   - shift_i = word[t_i] ⊕ word[t_{i-1}] — "что изменилось" между шагами
   - Если шифты consistent → есть паттерн → предсказываем продолжение
   - predicted_next = word[t_pos] ⊕ avg_shift

3. **Forward Verification** — кандидат создаёт известный паттерн вперёд?
   - Проверяет: есть ли в памяти биграмма (t_pos, candidate)?
   - Если да → кандидат является "началом следующего паттерна" → бонус

4. **Chain Verification** — более широкий контекст подтверждает через rules?
   - Проверяет: rules для (t_{pos-2}, t_{pos-1}) предсказывают t_pos?
   - Если да → широкий контекст consistent → бонус кандидату

Opcodes: `Attend`, `Shift`, `Verify` (38 total)

## Metrics

### v18.0 (без attention, sr=10)
- Top-1: 23.4%
- Top-5: 37.2%
- avg_uncertain: 14.1
- need_more_data: 3826/5000

### v18.1 (с HDC Binding Attention, sr=1)
- **Top-1: 41.7%** (+78%)
- **Top-5: 60.5%** (+63%)
- no_trigram_match: 0.0% (полное покрытие)
- avg_uncertain: 8.5 (-40%)
- avg_contradictions: 1.3 (-52%)
- need_more_data: 1623 (-58%)
- Memory: 8.5M facts, 1.57M rules, 932K abstractions

## Self-Diagnostic (v18.1)
- no_trigram_match: 0.0% (полное покрытие при sr=1)
- avg_uncertain: 8.5 из 20 кандидатов (улучшение с 14.1)
- avg_contradictions: 1.3 (улучшение с 2.7)
- need_more_data: 1623 из 5000 (33% — было 76%)

## Roadmap
1. [DONE] Codebook cache (375s → 0s)
2. [DONE] Расширение языка (+6 опкодов самоосознания + 3 attention)
3. [DONE] Самодиагностика (DiagnosticReport)
4. [DONE] HDC Binding Attention (semantic + shift + forward/chain verify)
5. [DONE] Full sample_rate=1 run: Top-1 41.7%, Top-5 60.5%
6. [TODO] Оптимизация скорости attention (eval 10K = ~30 мин)
7. [TODO] Структурные правила Explorer ("X — столица → country")
8. [TODO] Автономность червей (self-directed exploration)
9. [TODO] Расширение внутреннего языка (более сложные цепочки IF/THEN)
10. [TODO] Механизм "что червям нужно" → автоматическое curriculum learning
