import os
import requests
import numpy as np

# Датасеты: Шекспир + дополнительные тексты для разнообразия
DATASETS = {
    'shakespeare': "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    # Библия Короля Якова — классический английский, близок к Шекспиру по эпохе
    'bible': "https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt",
    # Сонеты Шекспира (Project Gutenberg, plain text)
    'sonnets': "https://www.gutenberg.org/cache/epub/1041/pg1041.txt",
}

TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"

def download_text(name, url):
    """Скачивает текст если ещё не скачан"""
    filepath = f"data_{name}.txt"
    if not os.path.exists(filepath):
        print(f"Скачивание {name} из {url}...")
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(r.text)
            print(f"  {name}: {len(r.text):,} символов")
        except Exception as e:
            print(f"  Ошибка загрузки {name}: {e} — пропускаем")
            return ""
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"  {name}: {len(text):,} символов (из кэша)")
        return text

    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def prepare_data():
    # Скачиваем все датасеты
    all_text = []
    for name, url in DATASETS.items():
        text = download_text(name, url)
        if text:
            all_text.append(text)

    # Также подхватываем оригинальный input.txt если есть
    if os.path.exists('input.txt'):
        with open('input.txt', 'r', encoding='utf-8') as f:
            orig = f.read()
        if orig not in all_text:
            all_text.insert(0, orig)
            print(f"  input.txt: {len(orig):,} символов (оригинал)")

    data = "\n\n".join(all_text)
    print(f"\nОбщий датасет: {len(data):,} символов")

    # Символьный токенизатор
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Размер словаря (символов): {vocab_size}")

    stoi = {ch: i for i, ch in enumerate(chars)}

    # Кодируем
    ids = np.array([stoi[c] for c in data], dtype=np.uint16)

    # Train/Val (90/10)
    n = len(ids)
    train_ids = ids[:int(n * 0.9)]
    val_ids = ids[int(n * 0.9):]

    train_ids.tofile(TRAIN_BIN)
    val_ids.tofile(VAL_BIN)
    print(f"Сохранено: {TRAIN_BIN} ({len(train_ids):,} токенов)")
    print(f"Сохранено: {VAL_BIN} ({len(val_ids):,} токенов)")

    # Метаданные
    import pickle
    meta = {
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': {i: ch for i, ch in enumerate(chars)}
    }
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    print("Метаданные сохранены в meta.pkl")

if __name__ == "__main__":
    prepare_data()
