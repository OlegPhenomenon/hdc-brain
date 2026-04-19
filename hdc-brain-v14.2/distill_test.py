"""
Эксперимент: можно ли выбросить блоки после обучения?

Загружаем обученную v14.1, сравниваем:
  A) Полная модель (кодбук + 8 блоков + thought loop)
  B) Только кодбук + простое HDC суммирование контекста

Если B даёт хоть какое-то качество — значит кодбук несёт знание,
и блоки можно заменить на простые HDC операции.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import sentencepiece as spm

# Добавим путь к v14.1 для импорта
V14_1_DIR = Path(__file__).parent.parent / "hdc-brain-v14.1"
sys.path.insert(0, str(V14_1_DIR))

from hdc_brain_v14_1 import HDCBrainV14_1, create_model

WEIGHTS = V14_1_DIR / "weights" / "best_hdc_brain_v14_1.pt"
SP_MODEL = V14_1_DIR / "bpe_en_32k.model"
VAL_BIN = V14_1_DIR / "val.bin"


class CodebookOnlyPredictor:
    """Предсказание БЕЗ блоков. Только кодбук + HDC операции.

    Алгоритм:
      1. Для каждого токена контекста — взять его код из кодбука
      2. Применить позиционный сдвиг (cyclic permutation)
      3. Сложить все коды с весами (ближний = больший вес)
      4. Сравнить сумму с каждым кодом в кодбуке → logits
    """
    def __init__(self, codebook, hdc_dim, decay=0.85):
        """
        codebook: (V, D) — обученный кодбук (float, STE binarized)
        decay: вес позиции = decay^i (0 = ближайший)
        """
        self.codebook = codebook  # (V, D)
        self.D = hdc_dim
        self.decay = decay

    @torch.no_grad()
    def predict(self, idx):
        """
        idx: (B, T) — токены контекста
        return: logits (B, T, V) — предсказание для каждой позиции
        """
        B, T = idx.shape
        D = self.D

        # STE binarization (как в модели)
        real = self.codebook[idx]  # (B, T, D)
        alpha = torch.mean(torch.abs(real), dim=-1, keepdim=True)
        codes = alpha * torch.sign(real)
        codes = torch.where(codes == 0, alpha, codes)

        # Cyclic permutation для позиций
        positions = torch.arange(T, device=idx.device)
        dim_idx = torch.arange(D, device=idx.device)
        perm_indices = (dim_idx.unsqueeze(0) - positions.unsqueeze(1)) % D
        codes = codes.gather(2, perm_indices.unsqueeze(0).expand(B, -1, -1))

        # Для каждой позиции t: объединить коды 0..t с весами
        logits_all = []
        for t in range(T):
            if t == 0:
                # Только один токен — просто его код
                h = codes[:, 0, :]  # (B, D)
            else:
                # Weighted bundle: сумма codes[0..t] с decay весами
                # Ближний (t) имеет вес 1.0, дальний (0) имеет вес decay^t
                weights = torch.tensor(
                    [self.decay ** (t - i) for i in range(t + 1)],
                    device=idx.device
                ).unsqueeze(0).unsqueeze(-1)  # (1, t+1, 1)
                h = (codes[:, :t+1, :] * weights).sum(dim=1)  # (B, D)

            # Cosine similarity с кодбуком
            h_norm = F.normalize(h, dim=-1)
            cb_norm = F.normalize(self.codebook, dim=-1)
            logits = h_norm @ cb_norm.T  # (B, V)
            logits_all.append(logits)

        return torch.stack(logits_all, dim=1)  # (B, T, V)


def load_model(device="cpu"):
    """Загрузить обученную модель."""
    print(f"Loading model from {WEIGHTS}...")
    model, config = create_model(vocab_size=32000)
    checkpoint = torch.load(WEIGHTS, map_location=device, weights_only=False)

    # Checkpoint может содержать model_state_dict или быть стейтом напрямую
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        print(f"  Checkpoint info: iter={checkpoint.get('iter', '?')}, "
              f"best_bpb={checkpoint.get('best_bpb', '?')}")
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params:,} params")
    return model


def load_real_data(n_seqs=500, seq_len=64):
    """Загрузить реальные токенизированные данные из val.bin."""
    data = np.fromfile(str(VAL_BIN), dtype=np.uint16)
    print(f"  Val data: {len(data):,} tokens")

    # Нарезаем на последовательности
    seqs = []
    max_start = len(data) - seq_len - 1
    rng = np.random.RandomState(42)
    starts = rng.randint(0, max_start, size=n_seqs)

    for s in starts:
        seqs.append(data[s:s + seq_len])

    return torch.tensor(np.stack(seqs), dtype=torch.long)


@torch.no_grad()
def evaluate_full_model(model, sequences, device, n_thoughts=3):
    """Оценка полной модели (с блоками)."""
    model.eval()
    sequences = sequences.to(device)
    B, T = sequences.shape

    x = sequences[:, :-1]  # контекст
    y = sequences[:, 1:]   # цели

    # Batched forward
    batch_size = 32
    total_correct = 0
    total_top10 = 0
    total_top50 = 0
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, B, batch_size):
        bx = x[i:i+batch_size]
        by = y[i:i+batch_size]
        logits, _ = model(bx, n_thoughts=n_thoughts)

        # Accuracy
        pred = logits.argmax(-1)
        total_correct += (pred == by).sum().item()

        # Top-10
        top10 = logits.topk(10, dim=-1).indices
        total_top10 += (top10 == by.unsqueeze(-1)).any(dim=-1).sum().item()

        # Top-50
        top50 = logits.topk(50, dim=-1).indices
        total_top50 += (top50 == by.unsqueeze(-1)).any(dim=-1).sum().item()

        # Loss (cross-entropy)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), by.reshape(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += by.numel()

    acc = total_correct / total_tokens
    top10 = total_top10 / total_tokens
    top50 = total_top50 / total_tokens
    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

    return {"acc": acc, "top10": top10, "top50": top50, "ppl": ppl}


@torch.no_grad()
def evaluate_codebook_only(model, sequences, device, decay=0.85):
    """Оценка без блоков — только кодбук + HDC."""
    sequences = sequences.to(device)
    B, T = sequences.shape

    predictor = CodebookOnlyPredictor(
        codebook=model.codebook.data,
        hdc_dim=model.hdc_dim,
        decay=decay,
    )

    x = sequences[:, :-1]
    y = sequences[:, 1:]

    batch_size = 32
    total_correct = 0
    total_top10 = 0
    total_top50 = 0
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, B, batch_size):
        bx = x[i:i+batch_size]
        by = y[i:i+batch_size]
        logits = predictor.predict(bx)

        # Temperature scaling для cosine similarity
        logits = logits / 0.1

        pred = logits.argmax(-1)
        total_correct += (pred == by).sum().item()

        top10 = logits.topk(10, dim=-1).indices
        total_top10 += (top10 == by.unsqueeze(-1)).any(dim=-1).sum().item()

        top50 = logits.topk(50, dim=-1).indices
        total_top50 += (top50 == by.unsqueeze(-1)).any(dim=-1).sum().item()

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), by.reshape(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += by.numel()

    acc = total_correct / total_tokens
    top10 = total_top10 / total_tokens
    top50 = total_top50 / total_tokens
    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

    return {"acc": acc, "top10": top10, "top50": top50, "ppl": ppl}


@torch.no_grad()
def analyze_codebook(model):
    """Анализ структуры кодбука: есть ли выученные паттерны?"""
    cb = model.codebook.data  # (V, D)

    # Бинаризуем как модель это делает
    alpha = torch.mean(torch.abs(cb), dim=-1, keepdim=True)
    cb_bin = torch.sign(cb)
    cb_bin = torch.where(cb_bin == 0, torch.ones_like(cb_bin), cb_bin)

    V, D = cb_bin.shape

    # Средний cosine similarity между случайными парами
    n_pairs = 1000
    idx1 = torch.randint(0, V, (n_pairs,))
    idx2 = torch.randint(0, V, (n_pairs,))
    sims = F.cosine_similarity(cb[idx1], cb[idx2], dim=-1)

    # Средний alpha (масштаб) по токенам
    alphas = alpha.squeeze()

    # Сколько уникальных бинарных кодов (из выборки)
    sample_size = min(1000, V)
    sample_idx = torch.randperm(V)[:sample_size]
    sample_bin = cb_bin[sample_idx]
    # Хэши бинарных кодов
    hashes = set()
    for i in range(sample_size):
        h = tuple(sample_bin[i, :64].int().tolist())  # первые 64 бита как хэш
        hashes.add(h)

    print(f"\n{'='*60}")
    print(f"CODEBOOK ANALYSIS")
    print(f"  Shape: {V} x {D}")
    print(f"  Alpha (scale): mean={alphas.mean():.4f}, std={alphas.std():.4f}")
    print(f"  Alpha range: [{alphas.min():.4f}, {alphas.max():.4f}]")
    print(f"  Random pair cosine sim: mean={sims.mean():.4f}, std={sims.std():.4f}")
    print(f"  Unique binary prefixes (64-bit, sample {sample_size}): {len(hashes)}")
    print(f"  Codebook is {'diverse' if len(hashes) > sample_size * 0.95 else 'COLLAPSED'}")
    print(f"{'='*60}")

    # Проверим: самые похожие пары
    print(f"\n  Top-5 most similar token pairs (из 1000 случайных):")
    top_sims, top_idx = sims.topk(5)
    for i in range(5):
        a, b = idx1[top_idx[i]].item(), idx2[top_idx[i]].item()
        print(f"    token {a} <-> token {b}: cosine={top_sims[i]:.4f}")


@torch.no_grad()
def generate_text(model, sp, prompt, device, max_len=50, n_thoughts=3):
    """Генерация текста полной моделью."""
    ids = sp.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_len=max_len, temperature=0.8, top_k=40,
                         rep_penalty=1.3, n_thoughts=n_thoughts)
    return sp.decode(out[0].tolist())


@torch.no_grad()
def generate_codebook_only(codebook, hdc_dim, sp, prompt, device,
                           max_len=50, decay=0.85, temperature=0.8, top_k=40):
    """Генерация текста только кодбуком."""
    ids = sp.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    D = hdc_dim

    predictor = CodebookOnlyPredictor(codebook, hdc_dim, decay)

    for _ in range(max_len):
        logits = predictor.predict(idx)  # (1, T, V)
        logits = logits[:, -1, :] / temperature  # последняя позиция

        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)

        if next_id.item() == 2:  # EOS
            break

    return sp.decode(idx[0].tolist())


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Загружаем модель и токенизатор
    model = load_model(device)
    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    print(f"Tokenizer: {sp.get_piece_size()} vocab")

    # 2. Анализ кодбука
    analyze_codebook(model)

    # 3. Загружаем реальные данные
    print(f"\nLoading real validation data...")
    sequences = load_real_data(n_seqs=300, seq_len=64)
    print(f"  {sequences.shape[0]} sequences x {sequences.shape[1]} tokens")

    # 4. Оценка полной модели
    print(f"\nEvaluating FULL model (blocks + codebook)...")
    t0 = time.time()
    full_results = evaluate_full_model(model, sequences, device, n_thoughts=3)
    t_full = time.time() - t0
    print(f"  Time: {t_full:.1f}s")
    for k, v in full_results.items():
        print(f"  {k}: {v:.4f}")

    # 5. Оценка без блоков
    print(f"\nEvaluating CODEBOOK-ONLY (no blocks, pure HDC)...")
    t0 = time.time()
    cb_results = evaluate_codebook_only(model, sequences, device, decay=0.85)
    t_cb = time.time() - t0
    print(f"  Time: {t_cb:.1f}s")
    for k, v in cb_results.items():
        print(f"  {k}: {v:.4f}")

    # 6. Генерация текста
    prompts = ["The meaning of life is", "In the beginning", "Science has shown that"]
    print(f"\n{'='*60}")
    print(f"TEXT GENERATION")
    print(f"{'='*60}")
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        text_full = generate_text(model, sp, prompt, device, max_len=30, n_thoughts=3)
        print(f"  Full model:    {text_full[:120]}")
        text_cb = generate_codebook_only(
            model.codebook.data, model.hdc_dim, sp, prompt, device,
            max_len=30, decay=0.85
        )
        print(f"  Codebook only: {text_cb[:120]}")

    # 7. Сравнение
    print(f"\n{'='*60}")
    print(f"COMPARISON: Full Model vs Codebook Only")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'Full model':>12} {'Codebook only':>14} {'Ratio':>10}")
    print(f"{'-'*50}")
    for k in ["acc", "top10", "top50", "ppl"]:
        fv = full_results[k]
        cv = cb_results[k]
        if k == "ppl":
            ratio = f"{cv/fv:.1f}x worse" if fv > 0 else "inf"
        else:
            ratio = f"{cv/fv:.2f}x" if fv > 0 else "inf"
        print(f"{k:<12} {fv:>12.4f} {cv:>14.4f} {ratio:>10}")
    print(f"{'Speed':<12} {t_full:>11.1f}s {t_cb:>13.1f}s {t_full/t_cb:>9.1f}x")

    # Размер модели
    full_size_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
    cb_size_mb = model.codebook.numel() * 4 / 1024 / 1024
    cb_binary_mb = model.codebook.numel() / 8 / 1024 / 1024
    print(f"\n{'Size (float32)':<20} {full_size_mb:>8.1f} MB {cb_size_mb:>12.1f} MB")
    print(f"{'Size (binary ±1)':<20} {'N/A':>8} {cb_binary_mb:>12.1f} MB")
    print(f"{'='*60}")

    # Verdict
    if cb_results["acc"] > full_results["acc"] * 0.1:
        print(f"\n>>> CODEBOOK CARRIES KNOWLEDGE!")
        print(f">>> Codebook-only retains {cb_results['acc']/max(full_results['acc'],1e-6)*100:.0f}% of full model accuracy")
        print(f">>> Distillation path is viable — blocks can potentially be removed.")
    else:
        print(f"\n>>> Blocks carry most of the knowledge.")
        print(f">>> Simple removal doesn't work — need smarter distillation.")


if __name__ == "__main__":
    main()
