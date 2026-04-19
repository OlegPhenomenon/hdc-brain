"""
HDC-Brain v14.1: English 300M

Evolution from v14 (103M Russian):
- Multi-Head Binding Attention (4 heads instead of 1)
- Scaled controller (2560 instead of 768)
- 8 blocks instead of 6
- 32K BPE vocab for English
- Target: ~300M params, competitive with SmolLM2-360M

Architecture:
1. BPE Token -> STE Bipolar Codebook -> Cyclic Permutation
2. 8 x HDCBlock (Memory + MultiHead Binding Attention + Controller)
3. Thought Loops: K passes through the same blocks
4. Output @ Codebook^T -> logits (weight-tied)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import math


class ControllerBlock(nn.Module):
    """Residual block: LN -> down-project -> GELU -> up-project."""
    def __init__(self, hdc_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hdc_dim)
        self.down = nn.Linear(hdc_dim, inner_dim)
        self.up = nn.Linear(inner_dim, hdc_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = F.gelu(self.down(h))
        h = self.dropout(self.up(h))
        return x + h


class MultiHeadBindingAttention(nn.Module):
    """Multi-Head Binding Attention.

    Instead of standard QKV projections (3 x D x D params),
    we use HDC binding: each head has 3 binding vectors (3 x head_dim params).

    For 4 heads with D=4096:
    - Standard attention: 3 x 4096 x 4096 = 50M params per layer
    - Our binding attention: 4 x 3 x 1024 = 12K params per layer
    - That's 4096x fewer parameters!

    Each head operates on its slice of the HDC dimension,
    computing Q=x*bv_q, K=x*bv_k, V=x*bv_v via element-wise binding.
    """
    def __init__(self, hdc_dim, n_heads=4):
        super().__init__()
        assert hdc_dim % n_heads == 0, f"hdc_dim {hdc_dim} must be divisible by n_heads {n_heads}"
        self.hdc_dim = hdc_dim
        self.n_heads = n_heads
        self.head_dim = hdc_dim // n_heads
        self.scale = self.head_dim ** -0.5

        # Each head has its own binding vectors
        # Shape: (n_heads, head_dim) for each of Q, K, V
        self.bv_q = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.bv_k = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.bv_v = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)

        # No output projection needed: in HDC each dimension is independent,
        # so heads just write to their own slice and concatenate naturally.

    def _ste_sign(self, w):
        """STE binarization: forward uses sign, backward uses real gradient."""
        alpha = torch.mean(torch.abs(w), dim=-1, keepdim=True)
        hard = alpha * torch.sign(w)
        hard = torch.where(hard == 0, alpha * torch.ones_like(hard), hard)
        return (hard - w).detach() + w

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        HD = self.head_dim

        # Binarize binding vectors via STE
        bv_q = self._ste_sign(self.bv_q)  # (H, HD)
        bv_k = self._ste_sign(self.bv_k)
        bv_v = self._ste_sign(self.bv_v)

        # Reshape x to (B, T, H, HD)
        x_heads = x.view(B, T, H, HD)

        # Bind: element-wise multiply (HDC operation)
        Q = x_heads * bv_q  # (B, T, H, HD)
        K = x_heads * bv_k
        V = x_heads * bv_v

        # Transpose for attention: (B, H, T, HD)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Causal mask
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float('-inf'))

        # Sigmoid attention (HDC-style, not softmax)
        attn = torch.sigmoid(scores * 4.0)
        attn = attn.masked_fill(~causal, 0.0)

        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, H, T, HD)

        # Concatenate heads (no projection needed in HDC)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)  # (B, T, D)

        return out


class HDCMemory(nn.Module):
    """HDC Working Memory: parallel scan recurrence with context mass.

    Each token has:
    - mass: how important this token is (learned projection)
    - decay: how fast previous context fades (learned projection)

    This creates a weighted context where recent important tokens
    have the most influence, similar to a learned exponential average.
    """
    def __init__(self, hdc_dim):
        super().__init__()
        self.hdc_dim = hdc_dim
        self.mass_proj = nn.Linear(hdc_dim, 1, bias=False)
        self.decay_proj = nn.Linear(hdc_dim, 1, bias=False)

    def forward(self, tokens_bound):
        B, T, D = tokens_bound.shape
        mass = torch.sigmoid(self.mass_proj(tokens_bound))    # (B, T, 1)
        decay = torch.sigmoid(self.decay_proj(tokens_bound))  # (B, T, 1)
        weighted = mass * tokens_bound

        # Parallel scan via log-space cumsum
        log_decay = torch.log(decay.clamp(min=1e-6))
        cum_log = torch.cumsum(log_decay, dim=1)
        diff = cum_log - cum_log.transpose(1, 2)
        diff = diff.clamp(max=0)

        causal = torch.tril(torch.ones(T, T, device=tokens_bound.device))
        diff = diff.masked_fill(causal == 0, -1e9)
        W = torch.exp(diff)

        context = torch.matmul(W, weighted)
        return context


class HDCBlock(nn.Module):
    """Memory + Multi-Head Binding Attention + Controller.

    Flow: x -> Memory -> LN -> Attention -> LN -> Controller -> out
    Each sub-block has a residual connection.
    """
    def __init__(self, hdc_dim, controller_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.memory = HDCMemory(hdc_dim)
        self.attention = MultiHeadBindingAttention(hdc_dim, n_heads)
        self.controller = ControllerBlock(hdc_dim, controller_dim, dropout)
        self.ln_mem = nn.LayerNorm(hdc_dim)
        self.ln_attn = nn.LayerNorm(hdc_dim)

    def forward(self, x):
        mem = self.memory(x)
        x = self.ln_mem(x + mem)
        attn = self.attention(x)
        x = self.ln_attn(x + attn)
        x = self.controller(x)
        return x


class ThoughtLoop(nn.Module):
    """Thought Loop: multi-pass reasoning through the same blocks.

    The model makes K passes through the same blocks.
    Each pass is a "thought" that refines understanding.

    - thought_gate: learned scalar [0,1] for how much each thought contributes
    - Thought position embeddings: each thought knows its number
    - Residual connection between thoughts prevents degradation

    Key finding from v14: 1 thought = garbage, 3 thoughts = quality.
    Thought Loops are architecturally essential, not optional.
    """
    def __init__(self, hdc_dim, max_thoughts=4):
        super().__init__()
        self.max_thoughts = max_thoughts
        self.thought_gates = nn.Parameter(torch.zeros(max_thoughts))
        self.thought_pos = nn.Parameter(torch.randn(max_thoughts, hdc_dim) * 0.01)
        self.ln = nn.LayerNorm(hdc_dim)

    def forward(self, h, blocks, n_thoughts=None, use_checkpoint=False):
        if n_thoughts is None:
            n_thoughts = self.max_thoughts if self.training else 2

        n_thoughts = min(n_thoughts, self.max_thoughts)

        # First pass — mandatory, full
        for block in blocks:
            if use_checkpoint and self.training:
                h = torch_checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)

        if n_thoughts <= 1:
            return h

        # Additional thoughts
        for t in range(1, n_thoughts):
            gate = torch.sigmoid(self.thought_gates[t])
            thought_input = self.ln(h) + self.thought_pos[t]

            thought = thought_input
            for block in blocks:
                if use_checkpoint and self.training:
                    thought = torch_checkpoint(block, thought, use_reentrant=False)
                else:
                    thought = block(thought)

            # Gated residual: h = (1-gate)*h + gate*thought
            h = h + gate * (thought - h)

        return h


class HDCBrainV14_1(nn.Module):
    """HDC-Brain v14.1: English 300M.

    Encoder:
      BPE Token -> STE Bipolar Codebook -> Cyclic Permutation

    Processing:
      ThoughtLoop(8 x HDCBlock) — K passes through same blocks

    Decoder:
      Output @ Codebook^T -> logits (weight-tied)

    Key differences from v14:
    - Multi-Head Binding Attention (4 heads)
    - Larger controller (2560 vs 768)
    - 8 blocks (vs 6)
    - 32K English vocab (vs 16K Russian)
    """
    def __init__(self, vocab_size, hdc_dim=4096, max_seq_len=512,
                 n_blocks=8, controller_dim=2560, n_heads=4,
                 dropout=0.1, max_thoughts=4, use_checkpoint=False):
        super().__init__()
        self.hdc_dim = hdc_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint

        # === STE Bipolar Codebook ===
        self.codebook = nn.Parameter(torch.randn(vocab_size, hdc_dim) * 0.02)

        # === HDC Blocks ===
        self.blocks = nn.ModuleList([
            HDCBlock(hdc_dim, controller_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        # === Thought Loop ===
        self.thought_loop = ThoughtLoop(hdc_dim, max_thoughts)

        # === Output ===
        self.output_ln = nn.LayerNorm(hdc_dim)
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def _ste_encode(self, idx):
        """STE bipolar encoding: binary in forward, continuous in backward."""
        real = self.codebook[idx]
        alpha = torch.mean(torch.abs(real), dim=-1, keepdim=True)
        hard = alpha * torch.sign(real)
        hard = torch.where(hard == 0, alpha, hard)
        return (hard - real).detach() + real

    def _cyclic_position(self, x):
        """Cyclic permutation for position encoding (HDC-native)."""
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        indices = (torch.arange(D, device=x.device).unsqueeze(0) - positions.unsqueeze(1)) % D
        return x.gather(2, indices.unsqueeze(0).expand(B, -1, -1))

    def forward(self, idx, targets=None, n_thoughts=None):
        B, T = idx.shape

        # === Encode ===
        tokens = self._ste_encode(idx)
        tokens = self._cyclic_position(tokens)

        # === Process with Thought Loops ===
        h = self.thought_loop(tokens, self.blocks, n_thoughts, self.use_checkpoint)

        # === Output ===
        h = self.output_ln(h)
        logits = F.linear(h, self.codebook) * self.output_scale

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, start_ids, max_len=200, temperature=0.8, top_k=40,
                 rep_penalty=1.3, n_thoughts=None):
        """Autoregressive generation with Thought Loops and repetition penalty."""
        idx = start_ids.clone()
        generated = []

        for _ in range(max_len):
            context = idx[:, -self.max_seq_len:]
            logits, _ = self(context, n_thoughts=n_thoughts)
            logits = logits[:, -1, :] / temperature

            # Repetition penalty
            if rep_penalty > 1.0 and generated:
                for token_id in set(generated[-50:]):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= rep_penalty
                    else:
                        logits[0, token_id] *= rep_penalty

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            generated.append(next_id.item())
            idx = torch.cat([idx, next_id], dim=1)

            if next_id.item() == 2:  # </s> / EOS
                break

        return idx


def create_model(vocab_size=32000, config=None):
    """Factory for HDC-Brain v14.1."""
    if config is None:
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
    model = HDCBrainV14_1(
        vocab_size=vocab_size,
        **config,
    )
    return model, config


if __name__ == '__main__':
    import time

    vocab_size = 32000  # English BPE
    config = {
        'hdc_dim': 4096,
        'max_seq_len': 512,
        'n_blocks': 8,
        'controller_dim': 2560,
        'n_heads': 4,
        'dropout': 0.1,
        'max_thoughts': 4,
        'use_checkpoint': False,
    }
    model, config = create_model(vocab_size, config)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"HDC-Brain v14.1: English 300M")
    print(f"  HDC dim: {config['hdc_dim']}")
    print(f"  Blocks: {config['n_blocks']} (Memory + {config['n_heads']}-Head Binding Attention + Controller)")
    print(f"  Controller: {config['controller_dim']}d inner")
    print(f"  Attention heads: {config['n_heads']}")
    print(f"  Thought Loops: {config['max_thoughts']} max")
    print(f"  Vocab: {vocab_size} (BPE)")
    print(f"  Params: {n_params:,}")
    print()

    # Params breakdown
    codebook_params = model.codebook.numel()
    block_params = sum(p.numel() for block in model.blocks for p in block.parameters())
    thought_params = sum(p.numel() for p in model.thought_loop.parameters())
    other_params = n_params - codebook_params - block_params - thought_params

    print(f"  Params breakdown:")
    print(f"    Codebook:     {codebook_params:>12,} ({codebook_params/n_params*100:.1f}%)")
    print(f"    Blocks (x{config['n_blocks']}):  {block_params:>12,} ({block_params/n_params*100:.1f}%)")
    print(f"    Thought Loop: {thought_params:>12,} ({thought_params/n_params*100:.1f}%)")
    print(f"    Output/Other: {other_params:>12,} ({other_params/n_params*100:.1f}%)")
    print(f"    TOTAL:        {n_params:>12,}")

    # Per-block breakdown
    print(f"\n  Per-block breakdown:")
    block = model.blocks[0]
    mem_p = sum(p.numel() for p in block.memory.parameters())
    attn_p = sum(p.numel() for p in block.attention.parameters())
    ctrl_p = sum(p.numel() for p in block.controller.parameters())
    ln_p = block.ln_mem.weight.numel() + block.ln_mem.bias.numel() + \
           block.ln_attn.weight.numel() + block.ln_attn.bias.numel()
    print(f"    Memory:    {mem_p:>10,} ({mem_p/block_params*config['n_blocks']*100:.1f}%)")
    print(f"    Attention: {attn_p:>10,} ({attn_p/block_params*config['n_blocks']*100:.1f}%)")
    print(f"    Controller:{ctrl_p:>10,} ({ctrl_p/block_params*config['n_blocks']*100:.1f}%)")
    print(f"    LayerNorms:{ln_p:>10,} ({ln_p/block_params*config['n_blocks']*100:.1f}%)")

    # Comparison with SmolLM2-360M
    print(f"\n  vs SmolLM2-360M:")
    smollm_attn_per_layer = 960 * 960 * 3  # Q,K,V projections + output
    our_attn_per_layer = attn_p
    print(f"    SmolLM2 attention params/layer: ~{smollm_attn_per_layer:,}")
    print(f"    Our binding attention params/layer: {our_attn_per_layer:,}")
    print(f"    Ratio: {smollm_attn_per_layer / our_attn_per_layer:.0f}x fewer attention params")

    # Test forward + backward
    device = 'cpu'
    model = model.to(device)
    model.train()

    B, T = 2, 64  # Small batch for CPU test
    x = torch.randint(0, vocab_size, (B, T), device=device)
    y = torch.randint(0, vocab_size, (B, T), device=device)

    print(f"\n  Testing forward + backward (B={B}, T={T})...")
    t0 = time.time()
    logits, loss = model(x, y, n_thoughts=2)
    loss.backward()
    elapsed = time.time() - t0
    print(f"    Forward+backward: {elapsed*1000:.0f}ms")
    print(f"    logits shape: {logits.shape}")
    print(f"    loss: {loss.item():.4f}")

    # Grad check
    with_grad = sum(1 for _, p in model.named_parameters()
                    if p.grad is not None and p.grad.norm() > 0)
    total = sum(1 for _ in model.parameters() if _.requires_grad)
    print(f"    Gradients: {with_grad}/{total} params have gradients")

    # Thought gate values
    gates = torch.sigmoid(model.thought_loop.thought_gates).tolist()
    print(f"    Thought gates (init): {[round(g, 3) for g in gates]}")

    print(f"\n  ALL OK")
