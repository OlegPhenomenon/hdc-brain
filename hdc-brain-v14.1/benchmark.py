"""
Benchmark HDC-Brain v14.1 vs SmolLM2-360M.

Tests:
1. Perplexity on WikiText-103
2. Model size comparison
3. Inference speed
4. Attention parameter efficiency

Usage:
  python3 benchmark.py --checkpoint best_hdc_brain_v14_1.pt
  python3 benchmark.py --checkpoint best_hdc_brain_v14_1.pt --compare-smollm
"""
import torch
import time
import os
import random
import argparse
import numpy as np
import sentencepiece as spm
from hdc_brain_v14_1 import create_model, HDCBrainV14_1

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def count_params(model):
    """Detailed parameter count."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_bytes(model):
    """Estimate model size in bytes."""
    # Float32 size
    fp32 = sum(p.numel() * 4 for p in model.parameters())
    # Float16 size
    fp16 = sum(p.numel() * 2 for p in model.parameters())
    # Binary codebook: 1 bit per dim + alpha per token
    codebook = model.codebook
    binary_bits = codebook.shape[0] * codebook.shape[1]  # 1 bit per element
    alpha_bytes = codebook.shape[0] * 4  # 1 float per token
    binary_size = binary_bits // 8 + alpha_bytes
    # Rest in fp16
    rest_fp16 = sum(p.numel() * 2 for n, p in model.named_parameters() if n != 'codebook')
    binary_total = binary_size + rest_fp16
    return fp32, fp16, binary_total


def benchmark_inference_speed(model, device, seq_lengths=[64, 128, 256, 512],
                               n_thoughts_list=[1, 2, 3], batch_size=1):
    """Benchmark inference speed."""
    model.eval()
    results = []

    for seq_len in seq_lengths:
        for n_thoughts in n_thoughts_list:
            x = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

            # Warmup
            with torch.no_grad():
                model(x, n_thoughts=n_thoughts)

            # Benchmark
            if device == 'cuda':
                torch.cuda.synchronize()

            n_runs = 10
            t0 = time.time()
            with torch.no_grad():
                for _ in range(n_runs):
                    model(x, n_thoughts=n_thoughts)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.time() - t0) / n_runs

            tokens_per_sec = batch_size * seq_len / elapsed
            results.append({
                'seq_len': seq_len,
                'n_thoughts': n_thoughts,
                'time_ms': elapsed * 1000,
                'tokens_per_sec': tokens_per_sec,
            })

    return results


def eval_perplexity_wikitext(model, sp, device, n_thoughts=3, max_tokens=100_000):
    """Evaluate perplexity on WikiText-103 test set."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  Install datasets: pip install datasets")
        return None

    print("  Loading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join([t for t in ds['text'] if len(t.strip()) > 0])

    # Tokenize
    ids = sp.encode(text)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    print(f"  Tokens: {len(ids):,}")

    # Evaluate perplexity in chunks
    model.eval()
    seq_len = model.max_seq_len
    total_loss = 0.0
    total_tokens = 0

    data = torch.tensor(ids, dtype=torch.long)

    with torch.no_grad():
        for i in range(0, len(data) - seq_len - 1, seq_len):
            x = data[i:i + seq_len].unsqueeze(0).to(device)
            y = data[i + 1:i + seq_len + 1].unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                _, loss = model(x, y, n_thoughts=n_thoughts)

            total_loss += loss.item() * seq_len
            total_tokens += seq_len

            if total_tokens % 10_000 < seq_len:
                ppl_so_far = np.exp(total_loss / total_tokens)
                print(f"    {total_tokens:,} tokens, PPL so far: {ppl_so_far:.2f}")

    avg_loss = total_loss / total_tokens
    ppl = np.exp(avg_loss)
    bpb = avg_loss / 0.6931
    return ppl, bpb


def attention_param_comparison():
    """Compare attention parameters vs standard transformer."""
    print("\n=== Attention Parameter Efficiency ===")
    hdc_dim = 4096
    n_heads = 4
    head_dim = hdc_dim // n_heads

    # HDC binding attention: 3 binding vectors per head
    hdc_params = n_heads * 3 * head_dim
    # Standard multi-head attention: Q, K, V, O projections
    std_params = 4 * hdc_dim * hdc_dim  # Q,K,V,O
    # GQA (like SmolLM2): Q full, K/V grouped
    gqa_kv_heads = 5
    gqa_head_dim = 64
    gqa_dim = 960
    gqa_params = gqa_dim * gqa_dim + 2 * gqa_dim * (gqa_kv_heads * gqa_head_dim) + gqa_dim * gqa_dim

    print(f"  HDC Binding (4 heads, D={hdc_dim}): {hdc_params:,} params")
    print(f"  Standard MHA (D={hdc_dim}): {std_params:,} params")
    print(f"  SmolLM2 GQA (D={gqa_dim}, 15H/5KV): {gqa_params:,} params")
    print(f"  Ratio vs Standard: {std_params / hdc_params:.0f}x fewer")
    print(f"  Ratio vs SmolLM2 GQA: {gqa_params / hdc_params:.0f}x fewer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='best_hdc_brain_v14_1.pt')
    parser.add_argument('--tokenizer', default='bpe_en_32k.model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wikitext', action='store_true', help='Run WikiText-103 perplexity')
    parser.add_argument('--speed', action='store_true', help='Run speed benchmarks')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    args = parser.parse_args()

    if args.all:
        args.wikitext = True
        args.speed = True

    print("=" * 60)
    print("HDC-Brain v14.1 Benchmark Suite")
    print("=" * 60)

    # Load model
    if os.path.exists(args.checkpoint):
        print(f"\nLoading model from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        config = ckpt['config']
        vocab_size = ckpt['vocab_size']
        model = HDCBrainV14_1(vocab_size=vocab_size, **config)
        model.load_state_dict(ckpt['model'])
        model = model.to(args.device)
        model.eval()
        print(f"  Loaded iter {ckpt.get('iter', '?')}, val_loss {ckpt.get('val_loss', '?')}")
    else:
        print(f"\nNo checkpoint found, using random weights for size/speed benchmarks")
        model, config = create_model()
        vocab_size = 32000
        model = model.to(args.device)
        model.eval()

    # === Parameter count ===
    print("\n=== Model Size ===")
    total, trainable = count_params(model)
    fp32, fp16, binary = model_size_bytes(model)
    print(f"  Parameters: {total:,} ({total/1e6:.1f}M)")
    print(f"  FP32 size: {fp32/1e6:.1f} MB")
    print(f"  FP16 size: {fp16/1e6:.1f} MB")
    print(f"  Binary codebook + FP16 rest: {binary/1e6:.1f} MB")
    print(f"  SmolLM2-360M FP16: ~724 MB")
    print(f"  Size ratio: {binary / (724*1e6) * 100:.0f}% of SmolLM2")

    # === Attention comparison ===
    attention_param_comparison()

    # === Speed ===
    if args.speed:
        print("\n=== Inference Speed ===")
        results = benchmark_inference_speed(model, args.device)
        print(f"  {'SeqLen':>6} | {'Thoughts':>8} | {'Time (ms)':>10} | {'Tok/s':>10}")
        print(f"  {'-'*6} | {'-'*8} | {'-'*10} | {'-'*10}")
        for r in results:
            print(f"  {r['seq_len']:>6} | {r['n_thoughts']:>8} | "
                  f"{r['time_ms']:>10.1f} | {r['tokens_per_sec']:>10.0f}")

    # === WikiText perplexity ===
    if args.wikitext:
        print("\n=== WikiText-103 Perplexity ===")
        if os.path.exists(args.tokenizer):
            sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
            ppl, bpb = eval_perplexity_wikitext(model, sp, args.device)
            if ppl:
                print(f"\n  WikiText-103 Perplexity: {ppl:.2f}")
                print(f"  WikiText-103 BPB: {bpb:.3f}")
                print(f"\n  Reference (SmolLM2-360M): ~TBD")
                print(f"  Reference (GPT-2 124M): PPL ~29.4")
        else:
            print(f"  Tokenizer not found: {args.tokenizer}")

    print("\n" + "=" * 60)
    print("Benchmark complete.")


if __name__ == '__main__':
    main()
