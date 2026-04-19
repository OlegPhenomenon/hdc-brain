# BiHDTrans: Binary Hyperdimensional Transformer (arXiv:2509.24425)

**Authors**: Jingtao Zhang, Yi Liu, Qi Shen, Changhong Wang (Sun Yat-sen University / Northeastern University)
**Task**: Multivariate Time Series (MTS) Classification
**Code**: https://anonymous.4open.science/r/BiHDTrans-4E55

---

## 1. Core Idea: HDC + Self-Attention Fusion

BiHDTrans performs self-attention **entirely in binary hyperdimensional space**. Instead of
real-valued Q/K/V projections and softmax, it uses:
- Binary hypervectors {-1, +1}^D (typically D=10000)
- Element-wise binding (Hadamard product) as the linear projection
- Integer dot-products for attention scores
- Boolean masking instead of softmax

**Key theoretical insight**: Binarizing in holographic high-dimensional space incurs
significantly *less* information distortion than directly binarizing neural network weights
(Theorem 1 & 2 in the paper, with proofs).

---

## 2. Architecture: Three Modules

### 2.1 HD Encoder (Input -> Binary Hypervectors)

For multivariate time series with N features and L time steps:

1. **Position hypervectors** F_i in {-1,+1}^D: randomly generated, quasi-orthogonal
   (Hamming distance ~0.5 between any pair)
2. **Value hypervectors** V_i^t: quantized feature values mapped to correlated HVs where
   Hamming distance is proportional to value difference
3. **Spatial encoding** per timestep: bind position with value, bundle across features
4. **Temporal encoding**: cyclic permutation rho^t embeds temporal order
5. **Binarize** with sign()

```
H_e^t = sign(rho^t(sum_{i=1}^{N} F_i * V_i^t))
```

Output: L binary hypervectors H_e = [H_e^1, H_e^2, ..., H_e^L], each in {-1,+1}^D

### 2.2 HD Transformer (Self-Attention in Binary Space)

**Q/K/V projection via binding** (replaces FC linear layers):
```
H_q = H_e * BV_q    (element-wise multiply with trainable binary BV_q)
H_k = H_e * BV_k
H_v = H_e * BV_v
```
Where BV_q, BV_k, BV_v are in {-1,+1}^D and are trainable.

This works because binding = multiplication by a **diagonal weight matrix** (each
dimension is independently scaled by +1 or -1). This is a restricted form of linear
projection but valid in HD space because dimensions are i.i.d.

**Attention scores via integer dot-product + boolean masking**:
```
B_a = bool(H_q @ H_k^T)     # B_a in {0,1}^{L x L}
```
- The dot product H_q @ H_k^T is computed in **integer arithmetic** (for binary {-1,+1}
  vectors, dot product = D - 2*HammingDist, which is integer)
- `bool()` maps positive values to 1, non-positive to 0
- **NO softmax, NO 1/sqrt(d) scaling** -- the binarization replaces both

**Attention output via selective bundling**:
```
H_a^t = sign(sum_{i=1}^{L} H_v^i * b_{t,i})    # b_{t,i} in {0,1}
```
- Only value HVs where the boolean mask is 1 are bundled (summed)
- Result is binarized via sign()

**Output projection**:
```
H_c = H_a * BV_a    # another trainable binary binding vector
```

**NO feed-forward block**. The paper argues that since HD space distributes information
uniformly (i.i.d.) across all dimensions, the FF block provides no additional expressiveness.

**Multi-head attention**: D is split across N_h heads, each head gets D/N_h dimensions.

### 2.3 HD Classifier (Associative Memory)

- Class prototypes C_k in {-1,+1}^D stored in associative memory (AM)
- Inference: argmax_k Hamming_similarity(H_output, C_k)
- Equivalent to a BNN linear layer with binary weights

**Only the last token** is used for classification (like [CLS] in BERT).

---

## 3. Training Procedure

### Learnable Parameters:
- **BV_q, BV_k, BV_v, BV_a**: binding hypervectors (one per attention head) -- each D-dimensional
- **C**: class prototypes in AM -- D x K matrix (K = number of classes)

### STE-Based Training:
1. Maintain **real-valued latent parameters** C_d and BV_d^(j) in [-1, +1]
2. Forward pass: binarize via sign(): C = sign(C_d), BV = sign(BV_d)
3. Backward pass: use **Straight-Through Estimator (STE)** -- gradient passes through sign() unchanged
4. Clip real-valued params to [-1, +1] after each update
5. Loss: **cross-entropy**
6. Optimizer: standard (Adam presumably)

### Gradient formulas:
```
dL/dC_d = dL/dO * H^(j) * sign(BV_d^(j))
dL/dBV_d^(j) = dL/dO * sign(C_d) * H^(j)
```

### What is NOT trainable:
- Position hypervectors F_i (random, fixed)
- Value hypervectors V_i (constructed from quantization codebook, fixed)
- The sign() binarization thresholds

### Architecture choices:
- Single Transformer encoder block
- Single attention layer (no stacking)
- Final token output for classification

---

## 4. Boolean Masking Mechanism (Key Innovation)

Traditional attention:
```
A = softmax(Q @ K^T / sqrt(d))    # A in [0,1]^{L x L}, rows sum to 1
output = A @ V                     # weighted average of values
```

BiHDTrans attention:
```
B_a = bool(H_q @ H_k^T)           # B_a in {0,1}^{L x L}, binary mask
output = sign(B_a @ H_v)           # selective bundling + binarize
```

Key differences:
- **No normalization**: no 1/sqrt(d), no softmax
- **Hard selection**: each position either contributes (1) or does not (0)
- **Integer arithmetic only**: dot products of binary vectors are simple popcount operations
- **sign() re-binarizes** the output back to {-1,+1}^D

This is called "selective bundling" -- only the value HVs that pass the boolean gate
are summed together. The sign() after summation re-binarizes the result.

---

## 5. Sequence Modeling Details

- Each timestep = one "token" (multivariate observation encoded to single HV)
- Temporal order encoded via **cyclic permutation** rho^t (shift by t positions)
- Self-attention captures cross-timestep dependencies
- Only the **last token** is used for final classification
- Sequence length L varies by dataset (from 2 to 405 in experiments)

---

## 6. Performance Numbers

### Accuracy (Table 1, average across 7 MTS datasets):

| Model | Type | Avg Accuracy |
|-------|------|-------------|
| Full Precision Transformer | FP32 | 86.22% |
| **BiHDTrans** | **Binary HD** | **84.36%** |
| BiT | Binary NN | 77.69% |
| BiBERT | Binary NN | 77.00% |
| BiViT | Binary NN | 71.77% |
| DistHD | HD computing | 69.89% |
| LeHDC | HD computing | 43.59% |
| QuantHD | HD computing | 36.56% |
| Vanilla HDC | HD computing | 33.13% |

BiHDTrans vs SOTA binary Transformers: **+6.67% average accuracy**
BiHDTrans vs SOTA HD computing: **+14.47% minimum improvement**

### Latency (Table 2, FPGA Artix-7 at 100MHz, microseconds):

| Model | Avg Latency |
|-------|------------|
| BiViT | 577.67 us |
| BiT | 573.46 us |
| BiBERT | 569.92 us |
| **BiHDTrans** | **20.60 us** |

BiHDTrans: **39.4x lower latency** than SOTA binary Transformers

### Dimensionality Reduction (D from 10000 -> 3600, 64% reduction):
- Still 1-2% better accuracy than SOTA binary Transformers
- 4.4x less model size
- Additional 49.8% latency reduction vs full-D baseline

### Hardware (FPGA Artix-7 xc7a200t):
- LUT utilization: 83-99% (varies by dataset/config)
- BRAM utilization: 9-56%
- FF utilization: 7-39%

---

## 7. Key Tricks for Making HDC + Attention Work

1. **Binding replaces linear projection**: Element-wise multiply with trainable binary
   vector = diagonal weight matrix. Valid because HD dimensions are i.i.d.

2. **Boolean masking replaces softmax**: The dot product of binary HVs naturally produces
   integer scores centered around 0. Thresholding at 0 (bool function) gives a natural
   binary attention mask. No need for temperature scaling.

3. **No feed-forward block**: In holographic HD space, information is already uniformly
   distributed across dimensions. FF block adds cost with no expressiveness gain.

4. **STE training**: Real-valued latents + sign() binarization + STE gradients. Same
   technique as BinaryConnect / BNN literature.

5. **Cyclic permutation for position encoding**: Instead of learned positional embeddings,
   use rho^t (circular bit-shift by t). This is free in hardware (just wiring).

6. **Quantized value codebook**: Feature values are quantized to q levels (q >= 3 required
   by Theorem 1). Each level maps to a binary HV where Hamming distances are proportional
   to value differences. This preserves metric structure in binary space.

7. **i.i.d. property enables parallelism**: Since each dimension of an HD vector is
   independent, you can pipeline computation across dimensions (process d dims at a time
   on resource-constrained hardware).

8. **Single block, last token**: Simple architecture -- one encoder block, classify from
   the last token only. Keeps things lightweight.

---

## 8. Theoretical Justification (Theorems)

**Theorem 1**: When D -> infinity and quantization q >= 3, the information distortion from
binarizing in hyperspace is LOWER than directly binarizing real-valued data. This explains
why BiHDTrans achieves better accuracy than binary NNs despite being fully binary.

**Theorem 2**: When binary HVs are bundled with binarized weights (boolean mask), the
distortion vs real-valued weights converges to zero as D -> infinity. This justifies
replacing softmax attention with boolean masking.

---

## 9. Implications for Your Project

For your hoffman_swarm HDC language model:

- **Binding as projection**: Instead of W_q, W_k, W_v matrices, use element-wise multiply
  with trainable binary vectors. Massively reduces parameters (D params vs D^2).

- **Boolean attention mask**: Replace softmax(QK^T/sqrt(d)) with bool(H_q @ H_k^T).
  Eliminates the most expensive nonlinearity in attention.

- **STE training works**: The paper confirms that STE + sign() binarization is viable for
  learning both the binding vectors and the associative memory weights.

- **No FF needed in HD space**: If your representations are truly holographic/distributed,
  skip the feed-forward block.

- **Dimensionality is tunable**: You can reduce D significantly (64% reduction still
  competitive) to trade accuracy for speed/size.

- **Caveat**: This paper targets classification (fixed-length MTS), not autoregressive
  generation. Adapting boolean attention for causal LM would require a causal mask on
  top of the boolean mask (lower-triangular AND with B_a).
