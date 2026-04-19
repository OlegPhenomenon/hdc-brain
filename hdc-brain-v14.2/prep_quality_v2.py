"""
Prepare high-quality instruction dataset v2 for HDC-Brain v14.1 finetune.

Strategy for 299M-param model: quality-filtered SFT mix, ~4x bigger than v1.

Sources:
1. OpenHermes 2.5 (Teknium, 1M pairs) — gold standard 2024 SFT mix.
   Filtered to short, ASCII-dominant, no code/math responses.
2. Alpaca-GPT4 (vicgalle, 52K pairs) — keep from v1, GPT-4 generated.
3. Alpaca x3 (Taori, 52K × 3 = 156K) — short factual, helps with "capital of X" style.
4. Factual seed (~5K pairs) — OPTIONAL, generated offline via LLM.
   Skipped here; add manually if needed.

Target: ~400K-500K filtered pairs, ~60-80M tokens.
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
    """Filter for quality instruction/response pairs — strict."""
    if not question or not answer:
        return False
    # Length bounds
    if len(answer) < 30 or len(answer) > 1000:  # tighter than v1 (was 1500)
        return False
    if len(question) < 10 or len(question) > 400:  # tighter
        return False
    # Skip code-heavy
    code_markers = ["```", "def ", "class ", "import ", "function ",
                    " var ", " const ", "public static", "=>", "->",
                    "<?php", "<html", "#include"]
    code_count = sum(1 for m in code_markers if m in answer)
    if code_count >= 1:  # was 2
        return False
    # Skip math-heavy
    if answer.count("=") > 3 or answer.count("$") > 0 or answer.count("\\") > 2:
        return False
    # Skip non-English (stricter)
    ascii_ratio = sum(1 for c in answer if ord(c) < 128) / max(len(answer), 1)
    if ascii_ratio < 0.92:  # was 0.85
        return False
    # Skip multi-turn artifacts / role markers leaking in
    bad_markers = ["user:", "assistant:", "### Instruction", "### Response",
                   "[INST]", "[/INST]", "<|im_start|>", "<|im_end|>"]
    low_q = question.lower()
    low_a = answer.lower()
    for m in bad_markers:
        if m.lower() in low_q or m.lower() in low_a:
            return False
    return True


all_pairs = []

# 1. OpenHermes 2.5 — main source
print("1. Loading OpenHermes-2.5...")
try:
    oh = load_dataset("teknium/OpenHermes-2.5", split="train")
    print(f"   Raw: {len(oh):,} examples")
    count = 0
    for ex in oh:
        convs = ex.get("conversations", [])
        # Extract first user/assistant pair only
        q, a = "", ""
        for msg in convs:
            role = msg.get("from", msg.get("role", ""))
            text = msg.get("value", msg.get("content", ""))
            if role in ("human", "user") and not q:
                q = text
            elif role in ("gpt", "assistant") and q and not a:
                a = text
                break
        if is_quality(q, a):
            all_pairs.append(fmt(q, a))
            count += 1
    print(f"   OpenHermes: {count:,} pairs kept")
    del oh
except Exception as e:
    print(f"   OpenHermes error: {e}")

# 2. Alpaca-GPT4 — keep from v1
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
    print(f"   Alpaca-GPT4: {count:,} pairs kept")
    del agpt4
except Exception as e:
    print(f"   Alpaca-GPT4 error: {e}")

# 3. Alpaca (original, Taori) x3 — factual short answers
print("3. Loading Alpaca x3...")
try:
    al = load_dataset("tatsu-lab/alpaca", split="train")
    count = 0
    kept_once = []
    for ex in al:
        q = ex.get("instruction", "")
        inp = ex.get("input", "")
        a = ex.get("output", "")
        if inp:
            q = f"{q}\n{inp}"
        if is_quality(q, a):
            kept_once.append(fmt(q, a))
    # x3 for factual emphasis
    for _ in range(3):
        all_pairs.extend(kept_once)
        count += len(kept_once)
    print(f"   Alpaca x3: {count:,} pairs ({len(kept_once):,} unique)")
    del al
except Exception as e:
    print(f"   Alpaca error: {e}")

print(f"\nTotal: {len(all_pairs):,} quality pairs")

# Shuffle
random.seed(42)
random.shuffle(all_pairs)

# Tokenize per-item (memory-safe)
print("Tokenizing...")
all_ids = []
for i, text in enumerate(all_pairs):
    all_ids.extend(sp.encode(text))
    if (i + 1) % 50000 == 0:
        print(f"  {i + 1:,}/{len(all_pairs):,} ({len(all_ids):,} tokens)")

print(f"Total: {len(all_ids):,} tokens from {len(all_pairs):,} pairs")

# Split 95/5
split = int(len(all_ids) * 0.95)
np.array(all_ids[:split], dtype=np.uint16).tofile("quality_v2_train.bin")
np.array(all_ids[split:], dtype=np.uint16).tofile("quality_v2_val.bin")

print("\nSaved:")
print(f"  quality_v2_train.bin: {os.path.getsize('quality_v2_train.bin') / 1e6:.1f} MB")
print(f"  quality_v2_val.bin: {os.path.getsize('quality_v2_val.bin') / 1e6:.1f} MB")
print("DONE")
