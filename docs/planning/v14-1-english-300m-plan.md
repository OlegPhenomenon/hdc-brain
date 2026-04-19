# HDC-Brain v14.1: English 300M — План

## Цель
Обучить HDC-Brain 300M на английском датасете, сравнить с SmolLM2-360M, опубликовать arXiv.

## Архитектура v14.1 (299M params) — РЕАЛИЗОВАНО

| Параметр | v14 (103M) | v14.1 (299M) | SmolLM2-360M |
|----------|-----------|-------------|-------------|
| hdc_dim | 4096 | 4096 | 960 (hidden) |
| controller_dim | 768 | 2560 | 2560 (MLP) |
| n_blocks | 6 | 8 | 32 |
| attention | 1 head binding | 4 heads binding | 15 heads GQA |
| attn params/layer | 12K | 12K | 2.7M |
| max_thoughts | 4 | 4 | N/A |
| SEQ_LEN | 256 | 512 | 8192 |
| vocab | BPE 16K (рус) | BPE 32K (англ) | BPE 49K |
| dropout | 0.1 | 0.1 | 0.0 |
| total params | 103M | **299M** | **362M** |

## Данные
- **FineWeb** или **OpenWebText** — 5-10B токенов английского текста
- BPE 32K vocab (sentencepiece или tiktoken)
- Pre-tokenized binary для скорости

## Сервер
- A100 80GB или H100 — vast.ai / Lambda / RunPod
- Бюджет: ~$2-5/час × 72-120 часов = $150-600
- Альтернатива: 2× A6000 48GB

## Этапы

### 1. Подготовка (1-2 дня, локально) — ВЫПОЛНЕНО
- [x] Масштабировать архитектуру: controller 2560, 8 блоков, 4-head binding attention
- [x] Скрипт скачивания и токенизации FineWeb (`prepare_data.py`)
- [x] Тест на CPU: forward+backward работает, все 128 params имеют градиенты
- [x] Точное количество параметров: **299,290,629** (299M)
- [x] Тренировочный скрипт (`train.py`) с bf16, gradient checkpointing, cosine LR
- [x] Benchmark скрипт (`benchmark.py`) для сравнения с SmolLM2
- [x] Server setup скрипт (`server_setup.sh`)
- [ ] BPE 32K tokenizer — тренируется на сервере при первом запуске

### 2. Обучение (3-5 дней, сервер)
- [ ] Арендовать A100
- [ ] Загрузить код + данные
- [ ] Pre-training: cosine LR, 100K+ iters
- [ ] Мониторинг BPB каждые 500 iter
- [ ] Цель: BPB < 3.5 (конкурентно с GPT-2)

### 3. Бенчмарки (1 день)
- [ ] WikiText-103 perplexity
- [ ] LAMBADA accuracy
- [ ] Сравнительная таблица vs GPT-2 124M
- [ ] Inference speed benchmark (CPU, GPU)
- [ ] Model size comparison (binary codebook vs float)

### 4. SFT (опционально, 1 день)
- [ ] Alpaca/Dolly English instruction dataset
- [ ] LoRA fine-tuning (не full SFT!)
- [ ] Демо-чат для показа

### 5. arXiv статья (3-5 дней)
- [ ] Формат: NeurIPS/ICML style
- [ ] Секции: Abstract, Introduction, Architecture, Experiments, Results, Conclusion
- [ ] Графики: BPB curves, attention visualization, thought gate dynamics
- [ ] Таблицы: сравнение с GPT-2, model size, inference speed
- [ ] LaTeX оформление

### 6. Публикация
- [ ] GitHub репозиторий (код + веса)
- [ ] arXiv submit
- [ ] Reddit r/MachineLearning пост
- [ ] Опционально: YouTube видео

## Ключевые метрики для статьи
- BPB на WikiText-103 vs GPT-2
- Params: 300M vs 124M (GPT-2)
- Inference memory: ~160MB (binary codebook) vs 500MB
- Attention params: 4096x меньше на слой
- Thought loops: gate dynamics, ablation study (1 vs 2 vs 3 thoughts)

## Риски
- BPB может не дойти до 3.5 → увеличить данные или params
- A100 дорого → начать с A6000, масштабировать если работает
- Multi-head binding attention — новый компонент, может потребовать отладку

## Бюджет
- Сервер: $200-400
- Домен/хостинг: $0 (GitHub Pages)
- arXiv: бесплатно
