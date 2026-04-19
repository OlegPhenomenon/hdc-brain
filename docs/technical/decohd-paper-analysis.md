# DecoHD Paper Analysis (arXiv:2511.03911v2)

**Title:** DecoHD: Decomposed Hyperdimensional Classification under Extreme Memory Budgets
**Authors:** Sanggeon Yun, Hyunwoo Oh, Ryozo Masukawa, Mohsen Imani
**Venue:** DATE 2026 (accepted)
**Code:** No official repository found

## Core Idea

Instead of storing C full class prototypes of dimension D (C x D parameters),
decompose them into a small shared set of "channel" vectors across N layers,
then combine them via binding (element-wise multiply) and bundling (weighted sum).

## Mathematical Formulation

### Setup
- D = hypervector dimension (tested: 1K, 10K)
- C = number of classes
- N = number of decomposition layers (tested: 1, 2, 3)
- L_i = channels per layer i
- M = total paths = product(L_i for i=1..N)
- d = latent dimension (tested: 256, 1024, 4096)

### Step 1: Input Encoding (frozen)
```
h = phi(x) = x @ W_enc    where W_enc in R^(d_in x D), frozen Gaussian random
```

### Step 2: Channel Generation from Latents
```
A_l^(i) = a_l^(i) @ R^(i)    for each layer i, channel l
```
Where:
- a_l^(i) in R^d  -- LEARNABLE low-dimensional latent vector
- R^(i) in R^(d x D) -- FROZEN random projector (Gaussian)
- A_l^(i) in R^D  -- the actual channel hypervector

### Step 3: Path Composition via Binding
Each path m = (m_1, ..., m_N) selects one channel per layer:
```
Z_m(h) = h * A_{m_1}^(1) * A_{m_2}^(2) * ... * A_{m_N}^(N)
```
Where * = element-wise multiplication (binding).

### Step 4: Class Prototype via Bundling
```
Y_c(h) = sum_{m=1..M} W_{c,m} * Z_m(h)
```
Where W in R^(C x M) are learnable bundling weights.

### Step 5: Classification Score
```
s_c(x) = dot(Y_c(h), h)
```

### Step 6: Loss
```
L = CrossEntropy(softmax(s_1..s_C), y)
```

## Training Algorithm (Algorithm 1)

```
INIT:
  a_l^(i) ~ N(0, sigma^2) for all layers i, channels l
  W_{c,m} = 1/M for all classes c, paths m
  FREEZE: phi (encoder), R^(i) (projectors)

FOR each epoch:
  FOR each mini-batch {(x_b, y_b)}:
    1. h_b = phi(x_b)
    2. A_l^(i) = a_l^(i) @ R^(i)  for all i, l
    3. Z_m(h_b) = h_b * prod_i A_{m_i}^(i)  for all paths m
    4. Y_c(h_b) = sum_m W_{c,m} * Z_m(h_b)  for all classes c
    5. s_c = dot(Y_c(h_b), h_b)
    6. loss = CrossEntropy(s, y)
    7. AdamW update on {a_l^(i)} and W only

Optimizer: AdamW, lr=1e-3, wd=5e-5, batch=1024, epochs=1000
```

## Inference (Streaming, Memory-Efficient)

```
h = phi(x)
Materialize channels: A_l^(i) = a_l^(i) @ R^(i)
s_c = 0 for all c

FOR m = 1..M:
  Z_m = h * prod_i A_{m_i}^(i)
  t_m = dot(Z_m, h)           // scalar
  FOR all c:
    s_c += W_{c,m} * t_m
  DISCARD Z_m

RETURN argmax_c s_c
```

Key insight: at inference, Z_m is used only for its dot product with h,
so you only need one path in memory at a time.

## Parameter Count

### Trainable (DecoHD):
```
P_train = sum(L_i) * d + C * M
         = L_tot * d + C * prod(L_i)
```

### Full baseline:
```
P_baseline = C * D
```

### Inference storage (deployed):
```
Channels: L_tot * D  (materialized from latents)
Bundling head: C * M
Total: L_tot * D + C * M
```

### Normalized memory footprint:
```
m = (C*M + L_tot*D) / (C*D)
```

## Results

### Datasets (all classification, NO language/NLP):
| Dataset | Features | Classes | Train | Test |
|---------|----------|---------|-------|------|
| ISOLET  | 617      | 26      | 6238  | 1559 |
| UCIHAR  | 261      | 12      | 6213  | 1554 |
| PAMAP2  | 75       | 5       | 611K  | 102K |
| PAGE    | 10       | 5       | 4925  | 548  |

### Accuracy vs Baseline:
- D=10K, m<=0.7: average gap ~0.15%
- D=10K, m<=0.5: average gap ~0.1%
- Worst case at m<=0.5: ~5.7% gap
- D=1K, 16-bit, m<=0.5: ~0.7% gap

### Parameter Reduction:
- Up to 97.44% fewer trainable parameters at saturation
- Memory footprint: 0.38x baseline at equivalent accuracy

### Hardware (1-layer DecoHD ASIC vs baselines):
| vs Platform        | Energy(x) | Speed(x) | Memory(x) |
|--------------------|-----------|----------|-----------|
| CPU (Ryzen 9 9950X)| 277x     | 35x      | 0.38x     |
| GPU (RTX 4090)     | 13.5x    | 3.7x     | 0.38x     |
| HDC ASIC baseline  | 2.0x     | 2.4x     | 0.38x     |

### Depth trade-offs (multi-layer on ASIC):
| Layers | Speedup | Memory | Accuracy drop |
|--------|---------|--------|---------------|
| 1      | 2.42x   | 0.38x  | 1.01%         |
| 2      | 0.94x   | 0.12x  | 9.81%         |
| 3      | 0.89x   | 0.08x  | 28.7%         |

## Binary Vector Considerations

The paper uses **float32** (with fp16/bf16/fp8/fp4 ablations).
- Binding = element-wise multiplication (NOT XOR)
- No binary/bipolar variant discussed
- No integer quantization applied
- For binary: would need to replace multiply-binding with XOR-binding

## Applicability to 16K Token x 8192D Binary Codebook

### What DecoHD does NOT cover:
- Language models / token embeddings
- Large vocabularies (max 26 classes tested)
- Binary/bipolar vectors
- Sequence processing

### Adaptation needed for your use case:

**Your setup:** 16K tokens, 8192 binary dimensions
- C = 16384 (token vocabulary = "classes")
- D = 8192 (binary dimensions)
- Full codebook: 16384 * 8192 = 134M bits = 16MB

**DecoHD-style decomposition (hypothetical):**
With N=2 layers, L_1=L_2=128 channels:
- M = 128 * 128 = 16384 paths (matches vocab!)
- Channel storage: 256 * 8192 = 2M bits = 256KB
- Bundling head: 16384 * 16384 = 268M entries (WORSE!)

**Problem:** When C is large (16K), the bundling head W in R^(C x M)
becomes the bottleneck. DecoHD was designed for small C (5-26 classes).

**Alternative for large vocab:**
With N=2, L_1=128, L_2=128:
- Assign each token a fixed path: token_id -> (m_1, m_2) = (id/128, id%128)
- W becomes unnecessary (fixed 1-hot assignment)
- codebook[token] = A_{id/128}^(1) * A_{id%128}^(2)
- Storage: 256 * 8192 = 2M bits vs 134M bits = 67x compression
- But: each token is a FIXED product of two basis vectors, no flexibility

**Better approach for binary:**
- Use XOR instead of multiply for binding
- Basis vectors: B_1[0..127] in {0,1}^8192, B_2[0..127] in {0,1}^8192
- codebook[token] = B_1[token / 128] XOR B_2[token % 128]
- 256 basis vectors instead of 16384 full vectors
- XOR preserves binary, POPCNT-friendly
- But: representational capacity is limited to L_1 * L_2 distinct vectors
