"""
Binary Voting Codebook — обучение кодбука без gradient descent.

Вместо float32 + STE + backprop:
  Кодбук хранится как бинарный {-1, +1} (int8).
  Обучение через Error-Weighted Bit Voting:
    1. Forward pass → предсказание → ошибка (Hamming distance)
    2. Для каждого бита: пример "голосует" за правильное значение
    3. Голоса накапливаются в flip_pressure (счётчик)
    4. Когда давление превышает порог → бит переключается

Аналогии с gradient descent:
  flip_pressure ≈ momentum
  threshold     ≈ learning rate (высокий = осторожно)
  confidence    ≈ loss weighting
  Hamming dist  ≈ loss function
"""
import torch
import torch.nn as nn
import numpy as np


class BinaryVotingCodebook(nn.Module):
    """Бинарный кодбук с обучением через голосование.

    Хранит:
      codebook: (V, D) int8 {-1, +1} — бинарные векторы токенов
      pressure: (V, D) float32 — накопленное давление на каждый бит
        (float потому что confidence дробный, но это НЕ градиенты —
         это счётчик голосов, который можно сделать int16 если нужно)

    Обучение:
      После forward pass получаем hidden state h и target token.
      sign(h) = "какими битами ДОЛЖЕН быть target" (по мнению модели).
      Сравниваем с текущим codebook[target] — где расхождение,
      накапливаем давление. Когда давление > threshold → FLIP.
    """

    def __init__(self, vocab_size: int, dim: int, threshold: float = 5.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.threshold = threshold

        # Бинарный кодбук {-1, +1}
        init = torch.randint(0, 2, (vocab_size, dim)) * 2 - 1  # {-1, +1}
        self.register_buffer("codebook", init.to(torch.int8))

        # Давление на переключение каждого бита
        self.register_buffer("pressure", torch.zeros(vocab_size, dim))

        # Статистика
        self.register_buffer("total_flips", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_votes", torch.tensor(0, dtype=torch.long))

    def encode(self, idx: torch.Tensor) -> torch.Tensor:
        """Token IDs → бинарные векторы (как float для совместимости с блоками).

        idx: (B, T) long
        return: (B, T, D) float {-1.0, +1.0}
        """
        return self.codebook[idx].float()

    def decode_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Hidden state → logits через dot product с кодбуком.

        h: (B, T, D) float
        return: (B, T, V) float — similarity scores
        """
        # Бинаризуем h перед сравнением
        h_sign = torch.sign(h)
        h_sign = torch.where(h_sign == 0, torch.ones_like(h_sign), h_sign)

        # Dot product с бинарным кодбуком
        # (B, T, D) @ (D, V) → (B, T, V)
        cb_float = self.codebook.float()  # (V, D)
        logits = torch.matmul(h_sign, cb_float.T)

        return logits

    @torch.no_grad()
    def vote(self, h: torch.Tensor, target_ids: torch.Tensor):
        """Бинарное голосование — обновить давление на биты кодбука.

        h: (B, T, D) float — hidden state перед output
        target_ids: (B, T) long — правильные токены

        Для каждой позиции (b, t):
          1. sign(h[b,t]) = "модель считает что target должен выглядеть так"
          2. codebook[target] = "а на самом деле target выглядит так"
          3. Где расхождение → голосуем за переключение
          4. Уверенность = насколько h далёк от нуля (чем дальше, тем увереннее)
        """
        B, T, D = h.shape

        # Что модель "хочет" видеть на месте target
        h_sign = torch.sign(h)
        h_sign = torch.where(h_sign == 0, torch.ones_like(h_sign), h_sign)
        h_sign = h_sign.to(torch.int8)  # (B, T, D) {-1, +1}

        # Уверенность: |h| — чем дальше от 0, тем увереннее
        confidence = torch.abs(h).clamp(max=5.0)  # (B, T, D)
        # Нормализуем по среднему
        confidence = confidence / (confidence.mean() + 1e-8)

        # Текущие коды target-токенов
        target_codes = self.codebook[target_ids]  # (B, T, D) int8

        # Где модель не согласна с текущим кодбуком
        # disagreement[i] = +1 если модель хочет +1 а кодбук говорит -1
        # disagreement[i] = -1 если модель хочет -1 а кодбук говорит +1
        # disagreement[i] = 0 если согласны
        desired = h_sign  # (B, T, D) int8
        current = target_codes  # (B, T, D) int8
        disagree_mask = (desired != current)  # (B, T, D) bool

        # Направление: в какую сторону голос (+1 = "сделай +1", -1 = "сделай -1")
        vote_direction = desired.float()  # (B, T, D)

        # Взвешенный голос: направление × уверенность, только где не согласны
        weighted_vote = vote_direction * confidence * disagree_mask.float()  # (B, T, D)

        # Аккумулируем давление на каждый токен
        # Нужно scatter_add по target_ids
        flat_targets = target_ids.reshape(-1)  # (B*T,)
        flat_votes = weighted_vote.reshape(-1, D)  # (B*T, D)

        # Для каждого уникального target_id суммируем голоса
        self.pressure.index_add_(0, flat_targets, flat_votes)

        self.total_votes += B * T

    @torch.no_grad()
    def apply_flips(self):
        """Применить накопленное давление: переключить биты где порог превышен."""
        # Где давление превышает порог
        flip_pos = self.pressure > self.threshold   # сделать +1
        flip_neg = self.pressure < -self.threshold  # сделать -1

        n_flips = flip_pos.sum() + flip_neg.sum()

        if n_flips > 0:
            # Применяем переключения
            self.codebook[flip_pos] = 1
            self.codebook[flip_neg] = -1

            # Сбрасываем давление где переключили
            self.pressure[flip_pos | flip_neg] = 0.0

            self.total_flips += n_flips

        return int(n_flips)

    @torch.no_grad()
    def decay_pressure(self, factor: float = 0.95):
        """Затухание давления — старые голоса теряют силу."""
        self.pressure *= factor

    def stats(self) -> dict:
        """Диагностика состояния кодбука."""
        p = self.pressure
        return {
            "total_flips": int(self.total_flips),
            "total_votes": int(self.total_votes),
            "pressure_mean": float(p.abs().mean()),
            "pressure_max": float(p.abs().max()),
            "pressure_above_half": int((p.abs() > self.threshold / 2).sum()),
            "ones_ratio": float((self.codebook == 1).float().mean()),
        }
