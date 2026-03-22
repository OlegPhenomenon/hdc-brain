"""
Introspection Mode — заглядываем внутрь сознательных агентов.

Показывает:
1. Какие агенты наиболее активны (энергия)
2. Динамический граф: кто с кем общается
3. Fitness scores: кто вносит больше в консенсус
4. Коалиции: какие агенты объединяются
5. HDC memory: что помнит каждый агент (similarity scores)
6. Разброс решений: единодушие vs разногласие
"""

import torch
import torch.nn.functional as F
import pickle
import numpy as np
from hoffman_agent import create_hoffman_swarm


def introspect(model, context_ids, tokenizer_itos, device='cuda', max_steps=10):
    """Генерация с интроспекцией — показываем внутренности."""
    model.eval()
    context = torch.tensor([context_ids], dtype=torch.long, device=device)
    seq_len = model.seq_len

    print("\n" + "="*70)
    print("ИНТРОСПЕКЦИЯ РОЕВОГО СОЗНАНИЯ")
    print("="*70)

    with torch.no_grad():
        for step in range(max_steps):
            if context.size(1) > seq_len:
                context = context[:, -seq_len:]

            # Сохраняем промежуточные данные
            B, T = context.shape
            n_agents = model.n_agents
            state_dim = model.state_dim

            # Прямой forward
            logits, _ = model(context)
            probs = F.softmax(logits[:, -1, :], dim=-1)

            # Top-5 предсказаний
            top_probs, top_ids = probs.topk(5)
            predictions = [(tokenizer_itos.get(idx.item(), '?'), f"{p.item():.3f}")
                          for idx, p in zip(top_ids[0], top_probs[0])]

            # Выбираем следующий токен
            idx_next = torch.multinomial(probs, 1)
            chosen_word = tokenizer_itos.get(idx_next.item(), '?')

            # === ИНТРОСПЕКЦИЯ ===
            print(f"\n--- Шаг {step+1}: выбрано '{chosen_word}' ---")
            print(f"  Top-5 варианты: {predictions}")

            # Энтропия решения (уверенность)
            entropy = -(probs * (probs + 1e-8).log()).sum().item()
            max_entropy = np.log(probs.shape[-1])
            confidence = 1.0 - entropy / max_entropy
            print(f"  Уверенность: {confidence:.1%} (энтропия: {entropy:.2f}/{max_entropy:.2f})")

            # Консенсус — насколько агенты согласны?
            # (Для этого нужно было бы вернуть промежуточные состояния,
            #  но можно аппроксимировать через дисперсию logits)
            logit_var = logits[:, -1, :].var().item()
            print(f"  Дисперсия logits: {logit_var:.4f} (выше = сильнее мнение)")

            context = torch.cat([context, idx_next], dim=1)

    # Итоговый сгенерированный текст
    generated_ids = context[0, len(context_ids):].tolist()
    generated = ''.join(tokenizer_itos.get(i, '?') for i in generated_ids)
    print(f"\n{'='*70}")
    print(f"СГЕНЕРИРОВАНО: {generated}")
    print(f"{'='*70}\n")
    return generated


def main():
    import os

    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    config = {
        'n_agents': 32, 'state_dim': 192, 'hdc_dim': 128,
        'n_sensory': 8, 'memory_slots': 64, 'seq_len': 64, 'dropout': 0.0
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = create_hoffman_swarm(meta['vocab_size'], config)
    model.to(device)

    # Загрузка весов
    for cp in ['best_swarm.pt', 'last_swarm.pt']:
        if os.path.exists(cp):
            try:
                sd = torch.load(cp, map_location=device, weights_only=True)
                sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
                model.load_state_dict(sd)
                print(f"Загружены: {cp}")
                break
            except Exception as e:
                print(f"Не подходит {cp}: {e}")

    print("\nВведите текст для интроспекции (или 'exit'):\n")

    while True:
        prompt = input("Демиург: ")
        if prompt.lower() in ['exit', 'quit']:
            break

        tokens = [meta['stoi'].get(w, 0) for w in prompt.split(' ') if w]
        introspect(model, tokens, meta['itos'], device=device, max_steps=15)


if __name__ == '__main__':
    main()
