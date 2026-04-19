"""
Fallback prep for quality_v2 — if OpenHermes 2.5 is unreachable from HF.

Uses alternate high-quality sources:
- Alpaca-GPT4 (vicgalle, 52K) — already worked
- Alpaca x3 (tatsu-lab, 52K × 3 = 156K)
- Dolly-15K (databricks) — high quality human-written
- LIMA (GAIR, 1K) — golden curated
- WizardLM evol-instruct (if available)

Target: ~300K filtered pairs, ~45-60M tokens (less than primary plan but still 3x original).
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
    if not question or not answer:
        return False
    if len(answer) < 30 or len(answer) > 1000:
        return False
    if len(question) < 10 or len(question) > 400:
        return False
    code_markers = ["```", "def ", "class ", "import ", "function ",
                    " var ", " const ", "public static", "=>", "->",
                    "<?php", "<html", "#include"]
    if sum(1 for m in code_markers if m in answer) >= 1:
        return False
    if answer.count("=") > 3 or answer.count("$") > 0 or answer.count("\\") > 2:
        return False
    ascii_ratio = sum(1 for c in answer if ord(c) < 128) / max(len(answer), 1)
    if ascii_ratio < 0.92:
        return False
    bad = ["user:", "assistant:", "### Instruction", "### Response",
           "[INST]", "[/INST]", "<|im_start|>"]
    la, lq = answer.lower(), question.lower()
    for m in bad:
        if m.lower() in la or m.lower() in lq:
            return False
    return True


all_pairs = []

# 1. Alpaca-GPT4
print("1. Alpaca-GPT4...")
try:
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    count = 0
    for ex in ds:
        q = ex.get("instruction", "")
        inp = ex.get("input", "")
        a = ex.get("output", "")
        if inp:
            q = f"{q}\n{inp}"
        if is_quality(q, a):
            all_pairs.append(fmt(q, a))
            count += 1
    print(f"   Alpaca-GPT4: {count:,}")
    del ds
except Exception as e:
    print(f"   err: {e}")

# 2. Alpaca x3 — factual emphasis
print("2. Alpaca x3...")
try:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    once = []
    for ex in ds:
        q = ex.get("instruction", "")
        inp = ex.get("input", "")
        a = ex.get("output", "")
        if inp:
            q = f"{q}\n{inp}"
        if is_quality(q, a):
            once.append(fmt(q, a))
    for _ in range(3):
        all_pairs.extend(once)
    print(f"   Alpaca x3: {3*len(once):,} ({len(once):,} unique)")
    del ds
except Exception as e:
    print(f"   err: {e}")

# 3. Dolly-15K
print("3. Dolly-15K...")
try:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    count = 0
    for ex in ds:
        q = ex.get("instruction", "")
        ctx = ex.get("context", "")
        a = ex.get("response", "")
        if ctx:
            q = f"{q}\n{ctx}"
        if is_quality(q, a):
            all_pairs.append(fmt(q, a))
            count += 1
    print(f"   Dolly: {count:,}")
    del ds
except Exception as e:
    print(f"   err: {e}")

# 4. LIMA
print("4. LIMA...")
try:
    ds = load_dataset("GAIR/lima", split="train")
    count = 0
    for ex in ds:
        convs = ex.get("conversations", [])
        if len(convs) >= 2:
            q = convs[0] if isinstance(convs[0], str) else str(convs[0])
            a = convs[1] if isinstance(convs[1], str) else str(convs[1])
            if is_quality(q, a):
                all_pairs.append(fmt(q, a))
                count += 1
    # repeat 10x to give LIMA weight despite small size
    lima_only = all_pairs[-count:]
    for _ in range(9):
        all_pairs.extend(lima_only)
    print(f"   LIMA: {count:,} x10 = {count*10:,}")
    del ds
except Exception as e:
    print(f"   err: {e}")

# 5. WizardLM Evol Instruct (70K)
print("5. WizardLM Evol Instruct...")
try:
    ds = load_dataset("WizardLMTeam/WizardLM_evol_instruct_70k", split="train")
    count = 0
    for ex in ds:
        q = ex.get("instruction", ex.get("input", ""))
        a = ex.get("output", ex.get("response", ""))
        if is_quality(q, a):
            all_pairs.append(fmt(q, a))
            count += 1
    print(f"   WizardLM: {count:,}")
    del ds
except Exception as e:
    print(f"   err: {e}")

print(f"\nTotal: {len(all_pairs):,} quality pairs")

random.seed(42)
random.shuffle(all_pairs)

print("Tokenizing...")
all_ids = []
for i, text in enumerate(all_pairs):
    all_ids.extend(sp.encode(text))
    if (i + 1) % 30000 == 0:
        print(f"  {i+1:,}/{len(all_pairs):,} ({len(all_ids):,} tokens)")

print(f"Total tokens: {len(all_ids):,}")

split = int(len(all_ids) * 0.95)
np.array(all_ids[:split], dtype=np.uint16).tofile("quality_v2_train.bin")
np.array(all_ids[split:], dtype=np.uint16).tofile("quality_v2_val.bin")

print(f"\nquality_v2_train.bin: {os.path.getsize('quality_v2_train.bin')/1e6:.1f} MB")
print(f"quality_v2_val.bin:   {os.path.getsize('quality_v2_val.bin')/1e6:.1f} MB")
print("DONE")
