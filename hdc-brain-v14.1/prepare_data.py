"""
Prepare English training data for HDC-Brain v14.1.

Uses FineWeb-Edu (same dataset as SmolLM2) with BPE 32K tokenizer.

Steps:
1. Train BPE 32K tokenizer on a sample of FineWeb-Edu
2. Download and tokenize FineWeb-Edu data
3. Save as binary files for fast training

Usage:
  # Step 1: Train tokenizer (needs ~1GB sample)
  python3 prepare_data.py --step tokenizer --sample-size 1000000

  # Step 2: Download and tokenize data
  python3 prepare_data.py --step tokenize --num-tokens 2000000000

  # Or do everything:
  python3 prepare_data.py --step all --num-tokens 2000000000
"""
import os
import argparse
import struct
import numpy as np
from pathlib import Path


def train_tokenizer(data_dir, vocab_size=32000, sample_size=1_000_000):
    """Train BPE tokenizer on FineWeb-Edu sample."""
    import sentencepiece as spm
    from datasets import load_dataset

    sample_file = os.path.join(data_dir, 'tokenizer_sample.txt')

    if not os.path.exists(sample_file):
        print(f"Downloading FineWeb-Edu sample ({sample_size:,} docs)...")
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        with open(sample_file, 'w', encoding='utf-8') as f:
            for i, example in enumerate(ds):
                if i >= sample_size:
                    break
                text = example['text'].strip()
                if len(text) > 50:
                    f.write(text + '\n')
                if (i + 1) % 100_000 == 0:
                    print(f"  {i+1:,} docs...")

        print(f"  Saved {sample_file} ({os.path.getsize(sample_file) / 1e9:.1f} GB)")

    model_prefix = os.path.join(data_dir, f'bpe_en_{vocab_size // 1000}k')
    if os.path.exists(model_prefix + '.model'):
        print(f"Tokenizer already exists: {model_prefix}.model")
        return model_prefix + '.model'

    print(f"Training BPE tokenizer (vocab={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=sample_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        byte_fallback=True,
        split_digits=True,
        max_sentence_length=16384,
        shuffle_input_sentence=True,
        input_sentence_size=5_000_000,
        # Special tokens
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )
    print(f"  Tokenizer saved: {model_prefix}.model")
    return model_prefix + '.model'


def tokenize_data(data_dir, tokenizer_path, num_tokens=2_000_000_000, val_ratio=0.001):
    """Download and tokenize FineWeb-Edu data."""
    import sentencepiece as spm
    from datasets import load_dataset

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    print(f"Tokenizer loaded: {sp.get_piece_size()} vocab")

    train_file = os.path.join(data_dir, 'train.bin')
    val_file = os.path.join(data_dir, 'val.bin')

    if os.path.exists(train_file):
        existing_size = os.path.getsize(train_file) // 2  # uint16
        print(f"train.bin already exists: {existing_size:,} tokens ({existing_size/1e9:.1f}B)")
        if existing_size >= num_tokens * 0.95:
            print("  Already have enough data, skipping.")
            return

    print(f"Target: {num_tokens:,} tokens from FineWeb-Edu...")

    # Use sample-10BT subset (10B tokens worth of text)
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Tokenize in chunks and write to binary
    train_tokens = []
    val_tokens = []
    total_tokens = 0
    doc_count = 0

    # Write as uint16 (max token id = 32000 < 65535)
    train_f = open(train_file, 'wb')
    val_f = open(val_file, 'wb')

    CHUNK_SIZE = 10_000_000  # Flush every 10M tokens
    train_buffer = []
    val_buffer = []

    for example in ds:
        text = example['text'].strip()
        if len(text) < 100:
            continue

        ids = sp.encode(text)
        ids.append(sp.eos_id())  # Add EOS between documents

        # Split: ~0.1% for validation
        if doc_count % 1000 == 0:
            val_buffer.extend(ids)
        else:
            train_buffer.extend(ids)

        total_tokens += len(ids)
        doc_count += 1

        # Flush train buffer
        if len(train_buffer) >= CHUNK_SIZE:
            arr = np.array(train_buffer, dtype=np.uint16)
            train_f.write(arr.tobytes())
            train_buffer = []

        # Flush val buffer
        if len(val_buffer) >= CHUNK_SIZE:
            arr = np.array(val_buffer, dtype=np.uint16)
            val_f.write(arr.tobytes())
            val_buffer = []

        if doc_count % 50_000 == 0:
            print(f"  {doc_count:,} docs, {total_tokens:,} tokens ({total_tokens/num_tokens*100:.1f}%)")

        if total_tokens >= num_tokens:
            break

    # Flush remaining
    if train_buffer:
        arr = np.array(train_buffer, dtype=np.uint16)
        train_f.write(arr.tobytes())
    if val_buffer:
        arr = np.array(val_buffer, dtype=np.uint16)
        val_f.write(arr.tobytes())

    train_f.close()
    val_f.close()

    train_size = os.path.getsize(train_file) // 2
    val_size = os.path.getsize(val_file) // 2

    print(f"\nDone!")
    print(f"  Documents: {doc_count:,}")
    print(f"  Train tokens: {train_size:,} ({train_size/1e9:.2f}B)")
    print(f"  Val tokens:   {val_size:,} ({val_size/1e6:.1f}M)")
    print(f"  Files: {train_file}, {val_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for HDC-Brain v14.1')
    parser.add_argument('--step', choices=['tokenizer', 'tokenize', 'all'], default='all')
    parser.add_argument('--data-dir', default='.', help='Directory for data files')
    parser.add_argument('--vocab-size', type=int, default=32000)
    parser.add_argument('--sample-size', type=int, default=1_000_000,
                        help='Number of docs for tokenizer training')
    parser.add_argument('--num-tokens', type=int, default=2_000_000_000,
                        help='Target number of training tokens (default: 2B)')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    tokenizer_path = os.path.join(args.data_dir, f'bpe_en_{args.vocab_size // 1000}k.model')

    if args.step in ('tokenizer', 'all'):
        tokenizer_path = train_tokenizer(args.data_dir, args.vocab_size, args.sample_size)

    if args.step in ('tokenize', 'all'):
        if not os.path.exists(tokenizer_path):
            print(f"Tokenizer not found: {tokenizer_path}")
            print("Run with --step tokenizer first")
            return
        tokenize_data(args.data_dir, tokenizer_path, args.num_tokens)


if __name__ == '__main__':
    main()
