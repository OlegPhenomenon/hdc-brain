"""
Prepare HIGH QUALITY instruction dataset for v14.1 finetune.

Strategy: quality > quantity. For 299M model:
1. LIMA (1K hand-curated — proven to match 50K+ datasets)
2. Alpaca-GPT4 (52K — GPT4 generated, cleaner than original Alpaca)
3. SlimOrca-Dedup (363K GPT4 — take only short, clear pairs < 500 chars)

Filter: remove code, math-heavy, non-English, too long/short responses.
Target: ~80-100K high quality pairs.
"""
import os
import random
import numpy as np
import sentencepiece as spm
from datasets import load_dataset

sp = spm.SentencePieceProcessor(model_file="bpe_en_32k.model")


def fmt(q, a):
    return f"### Instruction: {q.strip()}\n### Response: {a.strip()}\n"


def is_quality(question, answer):
    """Filter for quality instruction/response pairs."""
    if not question or not answer:
        return False
    # Too short or too long
    if len(answer) < 30 or len(answer) > 1500:
        return False
    if len(question) < 10 or len(question) > 500:
        return False
    # Skip code-heavy responses (our model struggles with code)
    code_markers = ["```", "def ", "class ", "import ", "function ", "var ", "const "]
    code_count = sum(1 for m in code_markers if m in answer)
    if code_count >= 2:
        return False
    # Skip math-heavy
    if answer.count("=") > 5 or answer.count("$$") > 0:
        return False
    # Skip non-English (rough check)
    ascii_ratio = sum(1 for c in answer if ord(c) < 128) / max(len(answer), 1)
    if ascii_ratio < 0.85:
        return False
    return True


all_pairs = []

# 1. LIMA — hand curated, highest quality
print("1. Loading LIMA...")
try:
    lima = load_dataset("GAIR/lima", split="train")
    count = 0
    for ex in lima:
        convs = ex.get("conversations", [])
        if len(convs) >= 2:
            q = convs[0] if isinstance(convs[0], str) else str(convs[0])
            a = convs[1] if isinstance(convs[1], str) else str(convs[1])
            if is_quality(q, a):
                all_pairs.append(fmt(q, a))
                count += 1
    print(f"   LIMA: {count} pairs (hand-curated)")
except Exception as e:
    print(f"   LIMA error: {e}")

# 2. Alpaca-GPT4 — GPT4 generated, clean
print("2. Loading Alpaca-GPT4...")
try:
    agpt4 = load_dataset("vicgalle/alpaca-gpt4", split="train")
    count = 0
    for ex in agpt4:
        q = ex.get("instruction", "")
        inp = ex.get("input", "")
        a = ex.get("output", "")
        if inp:
            q = f"{q}\n{inp}"
        if is_quality(q, a):
            all_pairs.append(fmt(q, a))
            count += 1
    print(f"   Alpaca-GPT4: {count} pairs")
    del agpt4
except Exception as e:
    print(f"   Alpaca-GPT4 error: {e}")

# 3. SlimOrca-Dedup — GPT4 completions, filtered
print("3. Loading SlimOrca-Dedup (this may take a minute)...")
try:
    orca = load_dataset("Open-Orca/SlimOrca-Dedup", split="train")
    print(f"   Raw: {len(orca)} examples")
    count = 0
    for ex in orca:
        convs = ex.get("conversations", [])
        q, a = "", ""
        for msg in convs:
            role = msg.get("from", msg.get("role", ""))
            text = msg.get("value", msg.get("content", ""))
            if role in ("human", "user"):
                q = text
            elif role in ("gpt", "assistant"):
                a = text
        if is_quality(q, a):
            all_pairs.append(fmt(q, a))
            count += 1
            if count >= 50000:  # cap at 50K to keep dataset balanced
                break
    print(f"   SlimOrca: {count} pairs (capped at 50K)")
    del orca
except Exception as e:
    print(f"   SlimOrca error: {e}")

print(f"\nTotal: {len(all_pairs)} quality pairs")

# Shuffle
random.seed(42)
random.shuffle(all_pairs)

# Tokenize per-item
print("Tokenizing...")
all_ids = []
for i, text in enumerate(all_pairs):
    all_ids.extend(sp.encode(text))
    if (i + 1) % 20000 == 0:
        print(f"  {i+1}/{len(all_pairs)} ({len(all_ids):,} tokens)")

print(f"Total: {len(all_ids):,} tokens from {len(all_pairs)} pairs")

# Split 95/5
split = int(len(all_ids) * 0.95)
np.array(all_ids[:split], dtype=np.uint16).tofile("quality_train.bin")
np.array(all_ids[split:], dtype=np.uint16).tofile("quality_val.bin")

print(f"\nSaved:")
print(f"  quality_train.bin: {os.path.getsize('quality_train.bin') / 1e6:.1f} MB")
print(f"  quality_val.bin: {os.path.getsize('quality_val.bin') / 1e6:.1f} MB")
print("DONE")
