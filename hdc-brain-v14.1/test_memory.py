"""Test GPU memory usage for different batch sizes."""
import torch
import torch.nn.functional as F
import sys

# Simulate CUDA memory tracking on CPU by estimating
# Run this ON THE SERVER with actual GPU

from hdc_brain_v14_1 import create_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab_size = 32000
config = {
    'hdc_dim': 4096,
    'max_seq_len': 512,
    'n_blocks': 8,
    'controller_dim': 2560,
    'n_heads': 4,
    'dropout': 0.1,
    'max_thoughts': 4,
    'use_checkpoint': True,
}

model, _ = create_model(vocab_size, config)
model = model.to(device)
model.train()
n_params = sum(p.numel() for p in model.parameters())

print(f"Model: {n_params:,} params")
print(f"Device: {device}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {total_mem:.1f} GB")

# Test different batch sizes
for batch_size in [1, 2, 4, 8]:
    for seq_len in [256, 512]:
        for n_thoughts in [1, 2]:
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            try:
                with torch.amp.autocast(device, enabled=True, dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float16):
                    _, loss = model(x, y, n_thoughts=n_thoughts)
                    loss.backward()

                if device == 'cuda':
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  B={batch_size} T={seq_len} thoughts={n_thoughts}: {peak:.2f} GB peak {'OK' if peak < total_mem * 0.9 else 'TIGHT!'}")
                else:
                    print(f"  B={batch_size} T={seq_len} thoughts={n_thoughts}: OK (CPU, no memory tracking)")

                model.zero_grad(set_to_none=True)

            except torch.cuda.OutOfMemoryError:
                print(f"  B={batch_size} T={seq_len} thoughts={n_thoughts}: OOM!")
                torch.cuda.empty_cache()
                model.zero_grad(set_to_none=True)
                break
            except Exception as e:
                print(f"  B={batch_size} T={seq_len} thoughts={n_thoughts}: {e}")
                model.zero_grad(set_to_none=True)

print("\nDone. Use the largest batch that fits under 90% VRAM.")
