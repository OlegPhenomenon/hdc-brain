"""
Подготовка расширенного датасета для Hoffman Swarm.
Цель: 5-10M символов чистого английского текста эпохи Шекспира.

Источники:
1. Tiny Shakespeare (уже есть) — 1.1M
2. Полное собрание Шекспира (Project Gutenberg) — ~5.5M
3. Сонеты Шекспира — ~100K
"""
import os
import requests
import numpy as np
import pickle

SOURCES = {
    # Полное собрание Шекспира (Project Gutenberg)
    'shakespeare_complete': 'https://www.gutenberg.org/cache/epub/100/pg100.txt',
    # Сонеты
    'sonnets': 'https://www.gutenberg.org/cache/epub/1041/pg1041.txt',
}

def download(name, url):
    filepath = f'data_{name}.txt'
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"  {name}: {len(text):,} (кэш)")
        return text
    print(f"  Скачиваю {name}...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        text = r.text
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"  {name}: {len(text):,} символов")
        return text
    except Exception as e:
        print(f"  Ошибка {name}: {e}")
        return ""

def clean_gutenberg(text, start_marker=None, end_marker="*** END OF THE PROJECT"):
    """Убираем Gutenberg заголовки и футеры"""
    # Ищем начало текста
    if start_marker:
        idx = text.find(start_marker)
        if idx >= 0:
            text = text[idx:]
    else:
        # Стандартный маркер Gutenberg
        for marker in ["*** START OF THE PROJECT", "***START OF THE PROJECT"]:
            idx = text.find(marker)
            if idx >= 0:
                # Пропускаем строку с маркером
                text = text[idx:]
                newline = text.find('\n')
                if newline >= 0:
                    text = text[newline+1:]
                break

    # Ищем конец
    if end_marker:
        idx = text.find(end_marker)
        if idx >= 0:
            text = text[:idx]

    return text.strip()

def prepare():
    all_texts = []

    # 1. Оригинальный Tiny Shakespeare (если есть)
    if os.path.exists('input.txt'):
        with open('input.txt', 'r', encoding='utf-8') as f:
            tiny = f.read()
        print(f"  input.txt (Tiny Shakespeare): {len(tiny):,}")
        all_texts.append(tiny)

    # 2. Полное собрание Шекспира
    complete = download('shakespeare_complete', SOURCES['shakespeare_complete'])
    if complete:
        complete = clean_gutenberg(complete)
        # Дополнительная очистка: убираем строки с номерами актов/сцен формата "  ACT I."
        # Оставляем — это часть структуры
        print(f"  shakespeare_complete (очищен): {len(complete):,}")
        all_texts.append(complete)

    # 3. Сонеты
    sonnets = download('sonnets', SOURCES['sonnets'])
    if sonnets:
        sonnets = clean_gutenberg(sonnets, start_marker="From fairest creatures")
        print(f"  sonnets (очищен): {len(sonnets):,}")
        all_texts.append(sonnets)

    # Объединяем
    data = "\n\n".join(all_texts)

    # Фильтруем: оставляем только ASCII printable + newline/tab
    # Это держит vocab маленьким
    allowed = set('\n\t ' + ''.join(chr(i) for i in range(32, 127)))
    data = ''.join(c for c in data if c in allowed)

    print(f"\n  Итого: {len(data):,} символов")

    # Символьный токенизатор
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"  Vocab: {vocab_size}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = np.array([stoi[c] for c in data], dtype=np.uint16)

    # Train/Val 90/10
    n = len(ids)
    train_ids = ids[:int(n * 0.9)]
    val_ids = ids[int(n * 0.9):]

    train_ids.tofile('train.bin')
    val_ids.tofile('val.bin')
    print(f"  train.bin: {len(train_ids):,} токенов")
    print(f"  val.bin: {len(val_ids):,} токенов")

    meta = {
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': {i: ch for i, ch in enumerate(chars)}
    }
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    print(f"  meta.pkl сохранён")

if __name__ == '__main__':
    print("Подготовка расширенного датасета...")
    prepare()
    print("Готово!")
