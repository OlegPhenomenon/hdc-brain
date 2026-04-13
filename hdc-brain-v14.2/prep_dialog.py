"""Add Dolly + Alpaca-GPT4 to existing dialog data."""
import os, random, numpy as np
import sentencepiece as spm
from datasets import load_dataset

sp = spm.SentencePieceProcessor(model_file="bpe_en_32k.model")

def fmt(q, a):
    return f"### Instruction: {q.strip()}\n### Response: {a.strip()}\n"

# Load existing
existing = np.fromfile("dialog_train.bin", dtype=np.uint16)
existing_val = np.fromfile("dialog_val.bin", dtype=np.uint16)
print(f"Existing: {len(existing):,} train + {len(existing_val):,} val tokens")

extra_pairs = []

# Dolly 15K
print("Loading Dolly...")
try:
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    for ex in dolly:
        q = ex.get("instruction", "")
        ctx = ex.get("context", "")
        a = ex.get("response", "")
        if ctx:
            q = f"{q}\n{ctx}"
        if q and a and len(a) > 10:
            extra_pairs.append(fmt(q, a))
    print(f"  Dolly: {len(extra_pairs)} pairs")
except Exception as e:
    print(f"  Dolly error: {e}")

# Alpaca-GPT4
n_before = len(extra_pairs)
print("Loading Alpaca-GPT4...")
try:
    agpt4 = load_dataset("vicgalle/alpaca-gpt4", split="train")
    for ex in agpt4:
        q = ex.get("instruction", "")
        inp = ex.get("input", "")
        a = ex.get("output", "")
        if inp:
            q = f"{q}\n{inp}"
        if q and a and len(a) > 10:
            extra_pairs.append(fmt(q, a))
    print(f"  Alpaca-GPT4: {len(extra_pairs) - n_before} pairs")
except Exception as e:
    print(f"  Alpaca-GPT4 error: {e}")

print(f"Extra total: {len(extra_pairs)} pairs")

random.seed(123)
random.shuffle(extra_pairs)

# Tokenize
print("Tokenizing extra...")
extra_ids = []
for i, text in enumerate(extra_pairs):
    extra_ids.extend(sp.encode(text))
    if (i + 1) % 20000 == 0:
        print(f"  {i+1}/{len(extra_pairs)} ({len(extra_ids):,} tokens)")

print(f"Extra: {len(extra_ids):,} tokens")

# Combine
split_extra = int(len(extra_ids) * 0.95)
all_train = np.concatenate([existing, np.array(extra_ids[:split_extra], dtype=np.uint16)])
all_val = np.concatenate([existing_val, np.array(extra_ids[split_extra:], dtype=np.uint16)])

all_train.tofile("dialog_train.bin")
all_val.tofile("dialog_val.bin")

print(f"\nFinal: train={len(all_train):,} tokens ({os.path.getsize('dialog_train.bin')/1e6:.1f}MB)")
print(f"       val={len(all_val):,} tokens ({os.path.getsize('dialog_val.bin')/1e6:.1f}MB)")
print("DONE")
