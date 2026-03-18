import os
import requests
import numpy as np

# Скачиваем небольшой датасет (Tiny Shakespeare)
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "input.txt"
TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"

def prepare_data():
    if not os.path.exists(DATA_FILE):
        print(f"Скачивание данных из {DATA_URL}...")
        r = requests.get(DATA_URL)
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            f.write(r.text)

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = f.read()
        
    print(f"Длина датасета: {len(data):,} символов")

    # Создаем символьный токенизатор
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Размер словаря (символов): {vocab_size}")
    
    stoi = { ch:i for i,ch in enumerate(chars) }
    # int to char нам не нужен для бинарников, но сохраним vocab_size
    
    # Кодируем весь текст в целые числа
    ids = np.array([stoi[c] for c in data], dtype=np.uint16)
    
    # Бьем на train/val (90/10)
    n = len(ids)
    train_ids = ids[:int(n*0.9)]
    val_ids = ids[int(n*0.9):]
    
    # Сохраняем в бинарные файлы
    train_ids.tofile(TRAIN_BIN)
    val_ids.tofile(VAL_BIN)
    print(f"Сохранено: {TRAIN_BIN} ({len(train_ids):,} токенов)")
    print(f"Сохранено: {VAL_BIN} ({len(val_ids):,} токенов)")
    
    # Сохраняем метаданные для train.py
    import pickle
    meta = {
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': { i:ch for i,ch in enumerate(chars) }
    }
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    print("Метаданные сохранены в meta.pkl")

if __name__ == "__main__":
    prepare_data()