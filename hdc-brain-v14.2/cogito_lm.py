"""
Cogito-LM: языковая модель на чистом HDC с итеративным рассуждением.

Идея:
  1. Берём обученный кодбук из v14.1 (backprop научил хорошие представления)
  2. Строим ассоциативные памяти из данных (кто за кем идёт)
  3. Инференс: Cogito-стиль рассуждение — итеративный поиск по памятям

НЕ RAG. Вся "память" — это статистика из данных, закодированная в HDC.
Никаких блоков, никаких матриц. Только bind, bundle, unbind, cleanup.

Памяти (строятся из данных):
  - pos_memories[i]: ассоциации "токен на расстоянии i → следующий токен"
    Для каждого токена t хранится СУММА кодов всех токенов, которые
    шли после t на расстоянии i. Это (V, D) матрица.

Рассуждение (Cogito-стиль):
  Шаг 1: Запрашиваем pos_memory[0] (ближайший контекст) → первый guess
  Шаг 2: Объединяем guess с контекстом → уточнённый зап��ос
  Шаг 3: Запрашиваем pos_memory[1] (второй контекст) → второй guess
  Шаг 4: Голосование между guess'ами → финальный ответ
  ...до N шагов
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import sentencepiece as spm

V14_1_DIR = Path(__file__).parent.parent / "hdc-brain-v14.1"
sys.path.insert(0, str(V14_1_DIR))

from hdc_brain_v14_1 import create_model

WEIGHTS = V14_1_DIR / "weights" / "best_hdc_brain_v14_1.pt"
SP_MODEL = V14_1_DIR / "bpe_en_32k.model"
VAL_BIN = V14_1_DIR / "val.bin"


def binarize_codebook(codebook):
    """STE-бинаризация как в модели: float → bipolar {-1, +1}."""
    return torch.sign(codebook).clamp(min=-1, max=1)


class CognitoLM:
    """Языковая модель на чистом HDC с Cogito-рассуждением.

    Все операции — целочисленные после бинаризации:
      bind = elementwise multiply (XOR для бинарных)
      bundle = sum
      permute = cyclic shift
      cleanup = argmax(dot product)
    """

    def __init__(self, codebook_float, n_pos_memories=5, device="cpu"):
        """
        codebook_float: (V, D) обученный кодбук (float)
        n_pos_memories: сколько позиционных памятей строить
        """
        self.device = device
        self.V, self.D = codebook_float.shape

        # Бинарный кодбук {-1, +1}
        self.codebook = binarize_codebook(codebook_float).to(device)

        self.n_pos = n_pos_memories

        # Позиционные ассоциативные памяти
        # pos_mem[i][tok] = сумма кодов всех токенов, которые шли через i позиций после tok
        # Форма: (n_pos, V, D)
        self.pos_memories = torch.zeros(n_pos_memories, self.V, self.D, device=device)

        # Статистика: сколько примеров для каждой пози��ии
        self.pos_counts = torch.zeros(n_pos_memories, self.V, device=device)

    def build_memories(self, data, max_tokens=500_000):
        """Построить ��ссоциативные памяти из токенизированных данных.

        data: numpy array of token ids
        """
        n = min(len(data), max_tokens)
        print(f"  Building memories from {n:,} tokens, {self.n_pos} positions...")
        t0 = time.time()

        # Строим на CPU чтобы не взорвать MPS память
        cb_cpu = self.codebook.cpu().float()
        pos_mem_cpu = torch.zeros(self.n_pos, self.V, self.D)
        pos_cnt_cpu = torch.zeros(self.n_pos, self.V)
        chunk_size = 50_000

        for pos in range(self.n_pos):
            offset = pos + 1
            if offset >= n:
                break

            for start in range(0, n - offset, chunk_size):
                end = min(start + chunk_size, n - offset)
                ctx_chunk = torch.from_numpy(data[start:end].astype(np.int64))
                tgt_chunk = torch.from_numpy(data[start + offset:end + offset].astype(np.int64))

                tgt_codes = cb_cpu[tgt_chunk]  # (chunk, D)
                pos_mem_cpu[pos].index_add_(0, ctx_chunk, tgt_codes)
                pos_cnt_cpu[pos].index_add_(0, ctx_chunk, torch.ones(len(ctx_chunk)))

        # Нормализуем на CPU, потом переносим на device
        for pos in range(self.n_pos):
            counts = pos_cnt_cpu[pos].unsqueeze(1).clamp(min=1)
            pos_mem_cpu[pos] /= counts

        self.pos_memories = pos_mem_cpu.to(self.device)
        self.pos_counts = pos_cnt_cpu.to(self.device)

        elapsed = time.time() - t0
        print(f"  Built in {elapsed:.1f}s")

        # Статистика
        for pos in range(self.n_pos):
            active = (self.pos_counts[pos] > 0).sum().item()
            print(f"    pos_memory[{pos}]: {active}/{self.V} tokens have associations")

    @torch.no_grad()
    def query_memory(self, pos_idx, token_id):
        """Запросить позиционную память: что обычно идёт после token_id на pos_idx+1 позиций?

        Возвращает вектор (D,) — усреднённый код "следующих" токенов.
        """
        return self.pos_memories[pos_idx, token_id]  # (D,)

    @torch.no_grad()
    def cleanup(self, noisy_vec, top_k=10):
        """Найти ближайшие токены к зашумлённому вектору.

        Возвращает (token_ids, similarities) для top_k кандидатов.
        """
        # Cosine similarity
        noisy_norm = F.normalize(noisy_vec.unsqueeze(0).float(), dim=-1)
        cb_norm = F.normalize(self.codebook.float(), dim=-1)
        sims = (noisy_norm @ cb_norm.T).squeeze(0)  # (V,)
        top_sims, top_ids = sims.topk(top_k)
        return top_ids, top_sims

    @torch.no_grad()
    def predict_simple(self, context_ids):
        """Простое предсказание: голосование по позиционным памятям.

        context_ids: list или 1D tensor — последние N токенов контекста
        """
        if isinstance(context_ids, list):
            context_ids = torch.tensor(context_ids, device=self.device)

        T = len(context_ids)
        D = self.D

        # Собираем "голоса" от каждой позиционной памяти
        vote = torch.zeros(D, device=self.device)
        n_votes = 0

        for pos in range(min(self.n_pos, T)):
            tok_id = context_ids[-(pos + 1)].item()  # от ближнего к дальнему
            mem_vec = self.query_memory(pos, tok_id)

            if mem_vec.abs().sum() < 1e-6:
                continue  # нет данных для этого токена

            # Вес: ближний контекст важнее
            weight = 1.0 / (pos + 1)
            vote += weight * mem_vec
            n_votes += 1

        if n_votes == 0:
            # Fallback: ближайший токен в кодбуке
            return context_ids[-1].unsqueeze(0), torch.tensor([0.0])

        return self.cleanup(vote, top_k=1)

    @torch.no_grad()
    def predict_cogito(self, context_ids, max_steps=5):
        """Cogito-рассуждение: итеративное уточнение предсказания.

        Как работае��:
          1. Начинаем с голосования позиционных памятей → первый guess
          2. Bind(guess, контекст) → уточнённый запрос
          3. Ищем в памятях снова → новый guess
          4. Если guess стабилизировался → выдаём
          5. До max_steps шагов

        context_ids: list или 1D tensor
        """
        if isinstance(context_ids, list):
            context_ids = torch.tensor(context_ids, device=self.device)

        T = len(context_ids)
        D = self.D

        # === Шаг 1: начальная оценка из позиционных памятей ===
        # Собираем взвешенные голоса от каждой позиции
        vote = torch.zeros(D, device=self.device)
        pos_vectors = []  # сохраним для итераций

        for pos in range(min(self.n_pos, T)):
            tok_id = context_ids[-(pos + 1)].item()
            mem_vec = self.query_memory(pos, tok_id)
            pos_vectors.append((tok_id, mem_vec, 1.0 / (pos + 1)))

            weight = 1.0 / (pos + 1)
            vote += weight * mem_vec

        # Первый guess
        top_ids, top_sims = self.cleanup(vote, top_k=5)
        current_guess = top_ids[0].item()
        current_conf = top_sims[0].item()

        if max_steps <= 1:
            return top_ids[:1], top_sims[:1], 1

        # === Шаги 2+: итеративное уточнение ===
        visited = {current_guess}
        guess_code = self.codebook[current_guess]  # (D,)

        for step in range(1, max_steps):
            # Уточнённый запрос: bind текущего guess с контекстом
            # Идея: "если следующий токен X, что обычно идёт ЕЩЁ дальше?"
            # Это цепочечное рассуждение — как в Cogito task 2

            refined_vote = torch.zeros(D, device=self.device)
            n_refine = 0

            # a) Перепроверяем: для guess, что стояло ПЕРЕД ним (обратная проверка)
            #    Если guess часто идёт после контекста — доверие растёт
            for pos in range(min(self.n_pos, T)):
                tok_id = context_ids[-(pos + 1)].item()
                mem_vec = self.query_memory(pos, tok_id)

                if mem_vec.abs().sum() < 1e-6:
                    continue

                # Насколько guess совпадает с тем, что предсказывает эта память?
                guess_sim = F.cosine_similarity(
                    guess_code.float().unsqueeze(0),
                    mem_vec.float().unsqueeze(0)
                ).item()

                # Если хорошо совпадает — усиливаем этот голос
                weight = (1.0 / (pos + 1)) * max(0, guess_sim)
                refined_vote += weight * mem_vec
                n_refine += 1

            # b) Контекстная связка: bind(last_token, guess) → запрос "что идёт в таком контексте?"
            last_code = self.codebook[context_ids[-1]]
            context_bind = last_code * guess_code  # bind в bipolar = elementwise multiply
            # Ищем: какой токен похож на bind(last, guess)?
            bind_ids, bind_sims = self.cleanup(context_bind, top_k=3)

            # c) Голосование: refined + bind результат
            if n_refine > 0:
                combined = refined_vote
                # Добавляем bind-кандидата с весом
                if bind_sims[0] > 0.05:
                    combined += 0.3 * self.codebook[bind_ids[0]].float()
            else:
                combined = vote  # откатываемся к начальном�� голосу

            new_ids, new_sims = self.cleanup(combined, top_k=5)
            new_guess = new_ids[0].item()
            new_conf = new_sims[0].item()

            # Если guess стабилизировался — хватит итерировать
            if new_guess == current_guess:
                return new_ids[:1], new_sims[:1], step + 1

            # Если мы уже были здесь — цикл, выходим с лучшим
            if new_guess in visited:
                return torch.tensor([current_guess], device=self.device), \
                       torch.tensor([current_conf], device=self.device), step + 1

            # Обновляем
            visited.add(new_guess)
            current_guess = new_guess
            current_conf = new_conf
            guess_code = self.codebook[current_guess]

        return torch.tensor([current_guess], device=self.device), \
               torch.tensor([current_conf], device=self.device), max_steps

    @torch.no_grad()
    def generate(self, sp, prompt, max_len=50, max_steps=5, temperature=0.8, top_k=40):
        """Генерация текста с Cogito-рассуждением и repetition penalty."""
        ids = sp.encode(prompt)

        for _ in range(max_len):
            context = ids[-self.n_pos:]
            ctx_t = torch.tensor(context, device=self.device)

            # Получаем голоса от памятей
            vote = torch.zeros(self.D, device=self.device)
            T = len(context)
            for pos in range(min(self.n_pos, T)):
                tok_id = context[-(pos + 1)]
                mem_vec = self.query_memory(pos, tok_id)
                vote += (1.0 / (pos + 1)) * mem_vec

            # Top-K из кодбука
            top_ids, top_sims = self.cleanup(vote, top_k=max(top_k, 50))

            # Repetition penalty: понижаем скор для недавних токенов
            recent = set(ids[-20:])
            scores = top_sims.clone()
            for j in range(len(top_ids)):
                if top_ids[j].item() in recent:
                    scores[j] *= 0.3  # штраф за повтор

            # Temperature + sampling
            scores = scores / max(temperature, 0.01)
            probs = F.softmax(scores, dim=-1)
            choice = torch.multinomial(probs, 1).item()
            next_id = top_ids[choice].item()

            if next_id == 2:
                break

            ids.append(next_id)

        return sp.decode(ids)


def load_model(device="cpu"):
    """Загрузить обученную модель (нужен только код��ук)."""
    print(f"Loading model from {WEIGHTS}...")
    model, _ = create_model(vocab_size=32000)
    checkpoint = torch.load(WEIGHTS, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate(cogito_model, data, n_seqs=300, seq_len=64, max_steps=5):
    """Оценка Cogito-LM на реальных данных."""
    rng = np.random.RandomState(42)
    max_start = len(data) - seq_len - 1
    starts = rng.randint(0, max_start, size=n_seqs)

    total_correct = 0
    total_top10 = 0
    total = 0
    total_steps = 0

    for i, s in enumerate(starts):
        seq = data[s:s + seq_len]

        # Для каждой позиции в последовательности
        for t in range(max(1, seq_len - 10), seq_len - 1):  # последние 10 позиций
            context = torch.tensor(seq[:t].astype(np.int64), device=cogito_model.device)
            target = int(seq[t])

            if max_steps <= 1:
                pred_ids, pred_sims = cogito_model.predict_simple(context)
                steps = 1
            else:
                pred_ids, pred_sims, steps = cogito_model.predict_cogito(
                    context, max_steps=max_steps
                )

            pred = pred_ids[0].item()
            total_correct += int(pred == target)

            # Top-10: получим 10 кандидатов от simple prediction
            vote = torch.zeros(cogito_model.D, device=cogito_model.device)
            T = len(context)
            for pos in range(min(cogito_model.n_pos, T)):
                tok_id = context[-(pos + 1)].item()
                mem_vec = cogito_model.query_memory(pos, tok_id)
                vote += (1.0 / (pos + 1)) * mem_vec
            top10_ids, _ = cogito_model.cleanup(vote, top_k=10)
            total_top10 += int(target in top10_ids.tolist())

            total += 1
            total_steps += steps

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_seqs} seqs, acc={total_correct/total:.4f}, "
                  f"top10={total_top10/total:.4f}, avg_steps={total_steps/total:.1f}")

    return {
        "acc": total_correct / total,
        "top10": total_top10 / total,
        "avg_steps": total_steps / total,
        "total": total,
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Загружаем кодбук
    model = load_model(device)
    codebook = model.codebook.data.to(device)
    del model  # освобождаем память от блоков
    torch.mps.empty_cache() if device == "mps" else None

    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    print(f"Tokenizer: {sp.get_piece_size()} vocab")
    print(f"Codebook: {codebook.shape}")

    # 2. Загружаем данные
    data = np.fromfile(str(VAL_BIN), dtype=np.uint16)
    print(f"Val data: {len(data):,} tokens")

    # 3. Создаём Cogito-LM и строим памяти
    cogito = CognitoLM(codebook, n_pos_memories=5, device=device)
    cogito.build_memories(data, max_tokens=500_000)

    # 4. Размер модели
    codebook_binary_mb = cogito.V * cogito.D / 8 / 1024 / 1024
    memories_mb = cogito.n_pos * cogito.V * cogito.D * 4 / 1024 / 1024  # float32
    memories_binary_mb = cogito.n_pos * cogito.V * cogito.D / 8 / 1024 / 1024  # если бинаризовать
    print(f"\nModel size:")
    print(f"  Codebook (binary):  {codebook_binary_mb:.1f} MB")
    print(f"  Memories (float32): {memories_mb:.1f} MB")
    print(f"  Memories (binary):  {memories_binary_mb:.1f} MB")
    print(f"  Total (binary):     {codebook_binary_mb + memories_binary_mb:.1f} MB")

    # 5. Генерация текста
    prompts = ["The meaning of life is", "In the beginning", "Science has shown that"]
    print(f"\n{'='*60}")
    print(f"TEXT GENERATION (Cogito reasoning)")
    print(f"{'='*60}")

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        for steps in [1, 3, 5]:
            text = cogito.generate(sp, prompt, max_len=25, max_steps=steps)
            print(f"  steps={steps}: {text[:120]}")

    # 6. Оценка
    print(f"\n{'='*60}")
    print(f"EVALUATION")
    print(f"{'='*60}")

    for steps in [1, 3, 5]:
        print(f"\n--- max_steps={steps} ---")
        t0 = time.time()
        results = evaluate(cogito, data, n_seqs=100, seq_len=64, max_steps=steps)
        elapsed = time.time() - t0
        print(f"  acc={results['acc']:.4f}  top10={results['top10']:.4f}  "
              f"avg_steps={results['avg_steps']:.1f}  time={elapsed:.1f}s")

    # 7. Сравнение с baseline
    print(f"\n{'='*60}")
    print(f"REFERENCE (from previous experiment)")
    print(f"  Full model (blocks): acc=0.2943, ppl=59")
    print(f"  Codebook only (weighted bundle): acc=0.0017, ppl=36143")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
