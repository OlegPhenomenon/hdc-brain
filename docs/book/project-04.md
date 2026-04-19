# Проект IV. Мини-HDCBrain: собери и обучи

---

## Что строим

Мини-версию HDC-Brain — полноценную языковую модель на 4M параметров. Она будет предсказывать следующий токен, как GPT, но на архитектуре HDC. Обучим на коротких текстах (стихи, сказки) и увидим генерацию.

Это кульминация Экспедиции IV: все блоки из глав 12–16 собираются в одну работающую модель.

---

## Что тебе нужно знать

- **HDC Memory** — глава 12, секция целиком. Отвечает за "память" модели — хранение предыдущего контекста.
- **Attention** — глава 13, секция "Attention — самый важный механизм в NLP". Механизм, который решает на что "смотреть".
- **HDC Binding Attention** — глава 14, секция целиком. Наша версия attention через Bind — ключевое отличие от трансформера.
- **HDCBlock** — глава 15, секция "HDCBlock — собираем блок целиком". Один блок = память + attention + feed-forward. Здесь складываем несколько блоков в стек.
- **Thought Loops** — глава 16, секция целиком. Многопроходное мышление — модель "думает" несколько раз над одним входом.
- **Кодбук** — глава 11 и проект III. Входной слой модели.
- **Loss и обучение** — глава 4, секция "Полный цикл обучения". Обучаем cross-entropy loss.

---

## Спецификация

### Архитектура

```python
import torch
import torch.nn as nn

class MiniHDCBrain(nn.Module):
    """
    Мини-версия HDC-Brain.

    Архитектура:
    - Codebook: vocab_size → dim (bipolar через STE)
    - Positional encoding: permute-based (глава 10)
    - N блоков HDCBlock (глава 15)
    - Thought loops: K проходов через блоки (глава 16)
    - Output head: dim → vocab_size (предсказание следующего токена)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 512,
        n_blocks: int = 2,
        n_thoughts: int = 2,
        context_len: int = 64,
    ):
        super().__init__()
        pass

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len)
        Возвращает: (batch, seq_len, vocab_size) — logits
        """
        pass

    def generate(self, prompt: torch.Tensor, max_len: int = 50, temperature: float = 1.0) -> torch.Tensor:
        """
        Автогрессивная генерация.
        prompt: (1, prompt_len) — начальные токены
        Возвращает: (1, prompt_len + generated_len)
        """
        pass


class HDCBlock(nn.Module):
    """
    Один блок HDC-Brain (глава 15).

    Содержит:
    - HDC Memory (глава 12)
    - HDC Binding Attention (глава 14)
    - Feed-forward с residual connection (глава 3)
    """

    def __init__(self, dim: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

### Обучение

```python
def train_mini_brain(
    texts: list[str],
    vocab_size: int = 256,
    dim: int = 512,
    n_blocks: int = 2,
    n_thoughts: int = 2,
    epochs: int = 20,
    lr: float = 3e-4,
) -> MiniHDCBrain:
    """
    Обучает мини-HDCBrain на текстах.

    1. Токенизируй тексты (символьный токенизатор или BPE из проекта III)
    2. Нарежь на окна context_len
    3. Loss = cross_entropy(model(tokens[:-1]), tokens[1:])
    4. Оптимизатор: AdamW
    5. Выводи loss каждую эпоху

    Возвращает обученную модель.
    """
    pass
```

---

## Тесты

```python
model = MiniHDCBrain(vocab_size=128, dim=256, n_blocks=2, n_thoughts=2, context_len=32)

# Forward pass
x = torch.randint(0, 128, (2, 16))
logits = model(x)
assert logits.shape == (2, 16, 128)

# Генерация
prompt = torch.randint(0, 128, (1, 5))
generated = model.generate(prompt, max_len=20)
assert generated.shape[1] > 5

# Подсчёт параметров
n_params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {n_params:,}")
# Должно быть порядка 1-5M при dim=256, n_blocks=2
```

---

## Данные для обучения

Возьми что-нибудь короткое и с повторяющимися паттернами:

```python
texts = [
    "Идёт бычок качается, вздыхает на ходу. Ох, доска кончается, сейчас я упаду.",
    "Наша Таня громко плачет, уронила в речку мячик. Тише Танечка не плачь, не утонет в речке мяч.",
    "Уронили мишку на пол, оторвали мишке лапу. Всё равно его не брошу, потому что он хороший.",
]
```

На таком объёме модель не станет гением, но ты увидишь что loss падает и генерация становится осмысленнее.

---

## Улучшения

- Обучи на большем тексте (сказки Пушкина, 50-100 КБ). Видна ли разница?
- Попробуй n_thoughts=1 vs n_thoughts=3. Сравни loss после 20 эпох.
- Визуализируй attention weights — на что модель "смотрит" при предсказании?
- Измерь BPB (bits per byte) — как в главе 18. Сравни с биграммной моделью из проекта I.
