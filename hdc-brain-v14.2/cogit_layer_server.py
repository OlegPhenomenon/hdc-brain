"""
CogitLayer for full v14.1 (4096d, 32K vocab).
GPU-native: all memory and operations on CUDA.

Memory budget: pos_mem (5, 32000, 4096) float32 = ~2.4 GB on GPU.
Total VRAM: 18 GB model + 2.5 GB CogitLayer = ~20.5 GB / 24 GB available.
"""
import torch
import numpy as np


class CogitLayer:
    """HDC associative memory on GPU. No backprop, pure index ops."""

    def __init__(self, vocab_size, dim, n_pos=5, device="cuda"):
        self.V = vocab_size
        self.D = dim
        self.n_pos = n_pos
        self.device = device

        # All on GPU for speed. float16 to save ~1.25 GB VRAM
        self.pos_mem = torch.zeros(n_pos, vocab_size, dim, device=device, dtype=torch.float16)
        self.pos_count = torch.zeros(n_pos, vocab_size, device=device)
        self.cb_binary = None  # (V, D) on GPU

    def update_codebook(self, codebook_param):
        """Sync binary codebook from model weights."""
        with torch.no_grad():
            self.cb_binary = torch.sign(codebook_param.data).to(self.device).float()

    @torch.no_grad()
    def observe_batch(self, token_ids):
        """Record observations from batch. token_ids: (B, T) on GPU."""
        if self.cb_binary is None:
            return

        B, T = token_ids.shape
        for pos in range(min(self.n_pos, T - 1)):
            offset = pos + 1
            ctx_ids = token_ids[:, :T - offset].reshape(-1)
            tgt_ids = token_ids[:, offset:T].reshape(-1)
            tgt_codes = self.cb_binary[tgt_ids].half()  # (N, D) float16
            self.pos_mem[pos].index_add_(0, ctx_ids, tgt_codes)
            ones = torch.ones(len(ctx_ids), device=self.device)
            self.pos_count[pos].index_add_(0, ctx_ids, ones)

    @torch.no_grad()
    def get_hint_fast(self, token_ids, target_device=None):
        """Fully GPU-vectorized hint computation.

        token_ids: (B, T) on GPU
        Returns: (B, T, D) on GPU, L2-normalized per token
        """
        B, T = token_ids.shape
        D = self.D
        hint = torch.zeros(B, T, D, device=self.device)

        for pos in range(min(self.n_pos, T)):
            weight = 1.0 / (pos + 1)
            if pos == 0:
                flat_ids = token_ids.reshape(-1)  # (B*T,)
                counts = self.pos_count[pos][flat_ids].unsqueeze(1).clamp(min=1)
                mem_vecs = self.pos_mem[pos][flat_ids].float() / counts
                hint += weight * mem_vecs.view(B, T, D)
            else:
                tok_ids = token_ids[:, :T - pos]  # (B, T-pos)
                flat_ids = tok_ids.reshape(-1)
                counts = self.pos_count[pos][flat_ids].unsqueeze(1).clamp(min=1)
                mem_vecs = self.pos_mem[pos][flat_ids].float() / counts
                hint[:, pos:] += weight * mem_vecs.view(B, T - pos, D)

        # Normalize by number of contributing positions
        n_votes = torch.arange(1, T + 1, device=self.device).float().clamp(max=self.n_pos)
        hint /= n_votes.view(1, T, 1)

        # L2 normalize per token
        hint_norm = hint.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        hint = hint / hint_norm

        return hint

    @torch.no_grad()
    def decay_memories(self, factor=0.995):
        self.pos_mem *= factor
        self.pos_count *= factor

    @torch.no_grad()
    def preseed(self, data_np, max_tokens=500_000, chunk_size=5000):
        """Pre-seed memories from numpy data. Processes on GPU in chunks."""
        if self.cb_binary is None:
            raise ValueError("Call update_codebook() first")

        n = min(len(data_np), max_tokens)
        print(f"  Pre-seeding CogitLayer from {n:,} tokens on GPU...")

        for pos in range(self.n_pos):
            offset = pos + 1
            if offset >= n:
                break
            for start in range(0, n - offset, chunk_size):
                end = min(start + chunk_size, n - offset)
                ctx = torch.from_numpy(data_np[start:end].astype(np.int64)).to(self.device)
                tgt = torch.from_numpy(data_np[start+offset:end+offset].astype(np.int64)).to(self.device)
                tgt_codes = self.cb_binary[tgt].half()
                self.pos_mem[pos].index_add_(0, ctx, tgt_codes)
                self.pos_count[pos].index_add_(0, ctx, torch.ones(len(ctx), device=self.device))

        for pos in range(self.n_pos):
            active = (self.pos_count[pos] > 0).sum().item()
            print(f"    pos_mem[{pos}]: {active}/{self.V} tokens have data")
        print(f"  Pre-seed done.")

    def memory_stats(self):
        mem_bytes = self.pos_mem.nelement() * 4
        cnt_bytes = self.pos_count.nelement() * 4
        total_mb = (mem_bytes + cnt_bytes) / 1024 / 1024
        vram_mb = total_mb  # all on GPU now
        return {
            "pos_mem_shape": list(self.pos_mem.shape),
            "total_mb": round(total_mb, 1),
            "vram_mb": round(vram_mb, 1),
            "active_tokens": [
                int((self.pos_count[p] > 0).sum()) for p in range(self.n_pos)
            ],
        }
