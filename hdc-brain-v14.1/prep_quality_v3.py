"""
Quality SFT dataset v3 — biggest available.

Adds to v2 fallback:
- OpenHermes 2.5 (Teknium, 1M pairs) — retry, HF should be back
- TULU 3 SFT mixture (AllenAI, ~1M pairs) — newest 2024 SFT mix
- Capybara (LDJnr, 16K) — high-curated dialog
- Plus: Alpaca-GPT4, Alpaca x3, Dolly, WizardLM Evol from v2

Target: 80-120M tokens of strict-filtered quality.
"""
import os
import random
import shutil
import numpy as np
import sentencepiece as spm
from datasets import load_dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def clear_hf_cache():
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    if os.path.isdir(hf_home):
        shutil.rmtree(hf_home, ignore_errors=True)

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


def extract_first_pair(convs):
    """Extract first user->assistant pair from a conversation list."""
    q, a = "", ""
    for msg in convs:
        role = msg.get("from", msg.get("role", ""))
        text = msg.get("value", msg.get("content", ""))
        if role in ("human", "user") and not q:
            q = text
        elif role in ("gpt", "assistant", "model") and q and not a:
            a = text
            break
    return q, a


all_pairs = []


def add_dataset(name, loader_fn):
    print(f"=== {name} ===", flush=True)
    try:
        before = len(all_pairs)
        loader_fn()
        added = len(all_pairs) - before
        print(f"   {name}: +{added:,} pairs (total: {len(all_pairs):,})", flush=True)
    except Exception as e:
        print(f"   {name} ERROR: {e}", flush=True)
    # Free disk after each dataset — HF caches raw files
    clear_hf_cache()


# 1. OpenHermes 2.5 (retry)
def load_oh():
    ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    print(f"   raw: {len(ds):,}")
    for ex in ds:
        q, a = extract_first_pair(ex.get("conversations", []))
        if is_quality(q, a):
            all_pairs.append(fmt(q, a))


add_dataset("OpenHermes-2.5", load_oh)


# 2. TULU 3 SFT mix
def load_tulu():
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    print(f"   raw: {len(ds):,}")
    count = 0
    for ex in ds:
        q, a = extract_first_pair(ex.get("messages", []))
        if is_quality(q, a):
            all_pairs.append(fmt(q, a))
            count += 1
            if count >= 500_000:  # cap
                break


add_dataset("TULU-3-SFT-mixture", load_tulu)


# 3. Capybara
def load_cap():
    ds = load_dataset("LDJnr/Capybara", split="train")
    print(f"   raw: {len(ds):,}")
    for ex in ds:
        convs = ex.get("conversation", ex.get("conversations", []))
        q, a = extract_first_pair(convs)
        if is_quality(q, a):
            all_pairs.append(fmt(q, a))


add_dataset("Capybara", load_cap)


# 4. Alpaca-GPT4
def load_agpt4():
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    for ex in ds:
        q = ex.get("instruction", "")
        inp = ex.get("input", "")
        a = ex.get("output", "")
        if inp:
            q = f"{q}\n{inp}"
        if is_quality(q, a):
            all_pairs.append(fmt(q, a))


add_dataset("Alpaca-GPT4", load_agpt4)


# 5. Alpaca x3 — factual emphasis
def load_alp():
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


add_dataset("Alpaca x3", load_alp)


# 6. Dolly-15K
def load_dol():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    for ex in ds:
        q = ex.get("instruction", "")
        ctx = ex.get("context", "")
        a = ex.get("response", "")
        if ctx:
            q = f"{q}\n{ctx}"
        if is_quality(q, a):
            all_pairs.append(fmt(q, a))


add_dataset("Dolly-15K", load_dol)


# 7. WizardLM Evol Instruct
def load_wiz():
    ds = load_dataset("WizardLMTeam/WizardLM_evol_instruct_70k", split="train")
    for ex in ds:
        q = ex.get("instruction", ex.get("input", ""))
        a = ex.get("output", ex.get("response", ""))
        if is_quality(q, a):
            all_pairs.append(fmt(q, a))


add_dataset("WizardLM-Evol-70K", load_wiz)


print(f"\n=== TOTAL: {len(all_pairs):,} pairs ===")

random.seed(42)
random.shuffle(all_pairs)

print("\nTokenizing...")
all_ids = []
for i, text in enumerate(all_pairs):
    all_ids.extend(sp.encode(text))
    if (i + 1) % 50000 == 0:
        print(f"  {i+1:,}/{len(all_pairs):,} ({len(all_ids):,} tokens)")

print(f"\nTotal tokens: {len(all_ids):,}")

split = int(len(all_ids) * 0.95)
np.array(all_ids[:split], dtype=np.uint16).tofile("quality_v3_train.bin")
np.array(all_ids[split:], dtype=np.uint16).tofile("quality_v3_val.bin")

print(f"\nquality_v3_train.bin: {os.path.getsize('quality_v3_train.bin')/1e6:.1f} MB")
print(f"quality_v3_val.bin:   {os.path.getsize('quality_v3_val.bin')/1e6:.1f} MB")
print("DONE")
