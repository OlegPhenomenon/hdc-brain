# Poduval et al. (2026) - Optimal Hyperdimensional Representation for Learning and Cognitive Computation

**Paper**: Frontiers in Artificial Intelligence, 10 Feb 2026, DOI: 10.3389/frai.2026.1690492
**Authors**: Poduval, Errahmouni Barkam, Liu, Yun, Ni, Zou, Bastian, Imani (UC Irvine, Purdue, West Point)

---

## 1. The Opposing Geometric Constraints

The central insight: **learning and cognition impose fundamentally different geometric constraints on the hyperspace representation**.

### Learning tasks (classification, pattern extraction):
- Require **correlative encodings** that cluster similar inputs together
- Correlated representations increase **memorization and generalization capacity**
- Intra-class similarity must be HIGH while inter-class similarity stays LOW
- Benefit from shared structure among samples in the bundled class hypervector
- Capacity scales as sqrt(p) where p = number of training samples (Eq. 5)

### Cognitive tasks (decoding, retrieval, symbolic reasoning, factorization):
- Require **exclusive encodings** that maximize separation between data points
- Near-orthogonal hypervectors minimize cross-talk during retrieval
- Essential for accurate information retrieval and interpretable decoding
- Factorization (resonator networks) demands even HIGHER exclusivity than simple decoding

**The single parameter `w` (kernel width) controls this trade-off.**

---

## 2. Optimal Separation Ranges

### Separation Metric Definition (Eq. 4):

```
s = (mu_2 - mu_1) / (sigma_1 + sigma_2)
```

where mu_i and sigma_i are means and standard deviations of the two score distributions (signal vs noise). This is a variance-normalized metric related to d' from signal detection theory.

### For Decoding/Cognition tasks:
- **Optimal separation: ~2-3** for robust performance under noise
- Achieved with **small w** (e.g., w = 0.1 to 0.5)
- Small w produces near-orthogonal positional codes, minimizing interference
- At D=1500, w=0.1 gives separation ~4.0; at w=0.5 gives ~4.5 (peak), declining after
- Decoding accuracy ranges 85% to 100% as separation is maximized
- Degradation boundary emerges around w ~ 1.5 (separation drops below ~2)

### For Learning/Classification tasks:
- **Optimal separation: ~0.8-1.2** for peak classification accuracy
- Achieved with **intermediate w** (e.g., w ~ 0.5 to 1.0)
- Peak classification accuracy at w ~ 0.5 for D >= 1500
- Too small w (too orthogonal): intra-class similarity suppressed, classifier cannot pool evidence
- Too large w (too correlated): inter-class codes overlap, decision boundary collapses
- Classification accuracy: 65% (random/orthogonal) to 95% (tuned correlative)
- Separation between class-similarity distributions scales as sqrt(p) * (mu1 - mu12)/(sigma1 + sigma12)

### For Factorization (Resonator Networks):
- Requires **highest exclusivity** (smallest w)
- Each factor must be independently decodable
- All kernel widths w_i must be set to ~1 (small) for correct factorization
- Large w_i causes factors to be ambiguous (cross-terms in iterative updates)

---

## 3. The Universal Kernel-Based Encoding

### Base Hypervectors (Eq. 1):

```
B_x = exp(i * theta_x / w_x),    B_y = exp(i * theta_y / w_y)
```

where:
- theta_x, theta_y ~ N(0, 1)^D  (random Gaussian phases, sampled once)
- w_x, w_y are kernel length scales (the tunable parameters)
- D is hypervector dimensionality
- i is the imaginary unit

These are **complex-valued phasor vectors** (FHRR-style VSA).

### Positional Encoding (Eq. 2):

```
P_{X,Y} = B_x^X (circle) B_y^Y
```

where (circle) denotes component-wise complex multiplication. This encodes integer coordinate (X, Y) by binding powers of the base vectors.

### Image Encoding:

For an image f with pixel values f_{X,Y}:

```
V_f = sum_{X,Y} f_{X,Y} * P_{X,Y}
```

This is a superposition (bundling) of all position-weighted basis hypervectors.

### Mathematical Equivalence:

This construction is mathematically equivalent to:
1. **Fractional Power Encoding (FPE)** (Plate, 1995)
2. **Random Fourier Feature constructions** (Rahimi & Recht, 2007)
3. **Vector Function Architecture** (Frady et al., 2022)

By Bochner's theorem, exponentiating Gaussian-distributed phases yields a **shift-invariant Gaussian kernel** whose width is controlled by the scaling parameter w.

### Similarity Structure:

The expected similarity between two positional codes:

```
E[delta(P_{X,Y}, P_{X',Y'})] = k((X-X')/w_x) * k((Y-Y')/w_y)
```

where k() is the Gaussian kernel function. This means:
- **Large w**: nearby positions have HIGH correlation (correlative encoding)
- **Small w**: nearby positions have LOW correlation (exclusive encoding)

---

## 4. Kernel Width Tuning

### Heuristic for Setting w (Section 3.2):

Compute empirical spatial co-activation probabilities:

```
p^1(x, X, Y) = P(f_{x,Y} = 1 AND f_{X,Y} = 1)
p^2(y, X, Y) = P(f_{X,y} = 1 AND f_{X,Y} = 1)
```

Then compute average correlation lengths:

```
l_x = <|x - X|>_{p^1}    (expected horizontal co-activation distance)
l_y = <|y - Y|>_{p^2}    (expected vertical co-activation distance)
```

Set kernel scales to match:

```
w_x ~ l_x
w_y ~ l_y
```

This ensures the positional code respects the dominant spatial correlations in the data.

### Empirical Guidelines from MNIST experiments (D=500, 13x13 binarized):

| Task | Optimal w_x | Optimal w_y | Notes |
|------|-------------|-------------|-------|
| Decoding | 0.1 - 0.5 | 0.1 - 0.5 | Small w, near-orthogonal codes |
| Classification | ~1.0 | ~0.5 | Intermediate, captures horizontal structure |
| Factorization | 1.0 | 1.0 | All factors need w_i = 1 (small/exclusive) |

### Anisotropy Insight:
- MNIST exhibits anisotropic structure: vertical strokes are more regular
- Increasing w_y is less harmful than increasing w_x for decoding
- For classification, w_x matters more (horizontal variation distinguishes digits)
- Decoding boundaries: w_y ~ 1.0, w_x ~ 1.5 (beyond which accuracy degrades)

---

## 5. Complete Mathematical Framework

### Decoding Pipeline (Section 2.2.2):

1. Encode image: V_f = sum_{X,Y} f_{X,Y} * P_{X,Y}
2. Compute similarity score: s_{X,Y} = Re(delta(P_{X,Y}, V_f))
3. Binarize: f_hat_{X,Y} = 1 if s_{X,Y} >= tau (threshold = mean score)
4. Iterative refinement: subtract current estimate, repeat

### Initial Decoding Estimate (Eq. 3):

```
f^0_{X,Y} = f_{X,Y} + sum_{X'!=X, Y'!=Y} f_{X',Y'} * delta(B_x^X (circle) B_y^Y, B_x^{X'} (circle) B_y^{Y'})
                                              |___________________________________|
                                                        Noise ~ N(mu, sigma)
```

where the noise statistics are:

```
mu = (1/2) * sum_{X'!=X,Y'!=Y} k((X-X')/w_x) * k((Y-Y')/w_y)

sigma^2 = (1/4) * sum_{X'!=X,Y'!=Y} [k((X-X')/w_x) * k((Y-Y')/w_y)]^2
```

### Iterative Refinement:

```
f^n_{X,Y} = Binarize(delta(B_x^X (circle) B_y^Y, V_f - V_{f^{n-1}}) + f^{n-1}_{X,Y})
```

Repeated until convergence: progressively cancels noise contributions.

### Learning Capacity (Section 2.5, Eq. 5):

For p training samples encoded as {H_1, ..., H_p}, bundled into class hypervector C = sum H_i:

Similarity of query Q with correct class:
```
delta(Q, C) ~ N(p * mu_1, sqrt(p) * sigma_1)
```

Similarity with incorrect class:
```
delta(Q, C') ~ N(p * mu_12, sqrt(p) * sigma_12)
```

**Class separation for learning:**
```
s = sqrt(p) * (mu_1 - mu_12) / (sigma_1 + sigma_12)
```

Key insight: separation grows as **sqrt(p)**, meaning more training samples enhance learning capacity when correlations are maintained.

### Factorization / Resonator Network (Section 2.6, Eq. 6):

Object encoded as: H = O_i (circle) B_X^x (circle) B_T^t (circle) B_S^s

Resonator update at iteration n:
```
G_{A,n} = [M]_A ( H (circle) prod_{i!=A} G_{i,n-1} )
```

where [M]_A is projection onto the codebook subspace for factor A.

- Near-orthogonal codes (small w_i) ensure cross-terms cancel in iterative updates
- Large w_i causes overlapping representations, spurious convergence

---

## 6. Quantitative Results Summary

### Decoding Accuracy (Table 1, Figure 7):

| D \ w | 0.1 | 0.5 | 1.0 | 1.5 | 2.0 |
|-------|-----|-----|-----|-----|-----|
| 1500  | 0.98| 0.99| 0.94| 0.92| 0.90|
| 1000  | 0.98| 0.98| 0.94| 0.92| 0.90|
| 500   | 0.96| 0.97| 0.94| 0.92| 0.90|
| 100   | 0.90| 0.91| 0.92| 0.92| 0.87|

**Best decoding: small w (0.1-0.5), large D (>=500)**

### Decoding Separation (Table 1c):

| D \ w | 0.1 | 0.5 | 1.0 | 1.5 | 2.0 |
|-------|-----|-----|-----|-----|-----|
| 1500  | 4.02| 4.51| 2.78| 2.28| 1.89|
| 1000  | 3.23| 3.80| 2.62| 2.24| 1.90|
| 500   | 2.26| 2.69| 2.33| 2.16| 1.80|
| 100   | 0.87| 1.07| 1.52| 1.86| 1.43|

### Classification Accuracy (Table 2, Figure 8):

| D \ w | 0.1 | 0.5 | 1.0 | 1.5 | 2.0 |
|-------|-----|-----|-----|-----|-----|
| 1500  | 0.86| 0.89| 0.91| 0.93| 0.88|
| 1000  | 0.84| 0.88| 0.92| 0.92| 0.86|
| 500   | 0.81| 0.88| 0.91| 0.94| 0.87|
| 100   | 0.66| 0.81| 0.85| 0.90| 0.87|

**Best classification: intermediate w (1.0-1.5), large D (>=500)**

### Classification Separation (Table 2c):

| D \ w | 0.1 | 0.5 | 1.0 | 1.5 | 2.0 |
|-------|-----|-----|-----|-----|-----|
| 1500  | 0.69| 0.81| 1.19| 1.22| 1.01|
| 1000  | 0.74| 0.82| 1.18| 1.23| 0.99|
| 500   | 0.63| 0.89| 1.12| 1.19| 1.02|
| 100   | 0.29| 0.68| 0.89| 1.25| 1.02|

---

## 7. Practical Recommendations for Hybrid Systems (Learning + Memory)

This is the critical section for our hoffman_swarm architecture.

### The Core Problem:
A hybrid system needs BOTH:
- **Correlative encoding** for neural learning (classification, pattern extraction)
- **Exclusive encoding** for HDC memory (associative retrieval, factorization)

These are geometrically OPPOSED requirements on the same representation space.

### Recommendation 1: Task-Adaptive Kernel Width
From the conclusion: "Rather than treating the encoder as a fixed, task-agnostic module, kernel-based HDC systems can expose w (and related parameters) as explicit knobs that are tuned according to the dominant role of the representation."

**Implementation**: Use different w values for different operational modes:
- **Memory write/retrieval mode**: w_small (0.1-0.5) for near-orthogonal codes
- **Learning/classification mode**: w_medium (0.5-1.5) for correlative structure
- Switch based on whether the system is currently learning or retrieving

### Recommendation 2: Dual Codebook Architecture
Run two parallel encoding paths:
- **Learning codebook**: w ~ 1.0, optimized for class separation and generalization
- **Memory codebook**: w ~ 0.1-0.3, optimized for precise retrieval and factorization
- Cross-reference between them as needed

### Recommendation 3: Dimension as a Buffer
- Higher D compensates for suboptimal w choice
- At D=1500, even w=1.0 gives 94% decoding accuracy
- At D=1500, w=1.0 gives 91% classification accuracy
- **If you can afford large D, you can use a SINGLE intermediate w (~1.0) that works reasonably for BOTH tasks**
- The "sweet spot" for a single encoder: w ~ 0.5-1.0 at D >= 1000

### Recommendation 4: Data-Driven Kernel Selection
- Compute empirical co-activation statistics from data
- Set w_x ~ l_x, w_y ~ l_y (match data correlation lengths)
- This provides a principled initialization, then sweep locally

### Recommendation 5: Scaling Behavior
- Decoding separation increases with D (more dimensions = more orthogonality budget)
- Learning separation increases with sqrt(p) (more samples = better class centroids)
- For hybrid systems: **increase D** to simultaneously serve both tasks
- At D=1500: decoding and classification both achieve >90% at w~1.0

### Recommendation 6: For Resonator/Factorization Components
- If your architecture includes a factorization component (resonator network), that subsystem MUST use exclusive encoding (small w)
- Cannot compromise on this -- large w causes incorrect factorization
- Keep factorization encoding separate from learning encoding

---

## 8. Key Implications for Our Architecture

### Direct Mapping to hoffman_swarm:

1. **HDC Associative Memory (hdc_am.py)**: Should use **small w (0.1-0.5)** for the codebook. This maximizes retrieval accuracy and separation. The current approach likely uses random/orthogonal codes, which is already close to optimal for memory.

2. **Neural Learning Component (train_hdc.py)**: If using HDC representations as input features for learning, the encoding should use **intermediate w (0.5-1.0)** to preserve intra-class correlations while maintaining inter-class separation.

3. **Bridging Strategy**: The paper suggests that at sufficiently high D (>=1000), a single encoding with w~1.0 can serve both purposes with acceptable performance (~90%+ for both tasks). This is the simplest hybrid approach.

4. **Trainable w**: The kernel width w could itself be a **learnable parameter** (as we already do with trainable codebooks). Gradient descent on w would naturally find the task-optimal balance.

5. **Separation Metric as Loss**: The separation metric s = (mu2 - mu1)/(sigma1 + sigma2) could serve as an auxiliary training signal to monitor encoding quality.

---

## 9. Mathematical Summary Card

```
ENCODING:
  B_i = exp(i * theta_i / w_i),  theta_i ~ N(0,1)^D
  P_{X,Y} = B_x^X (circle) B_y^Y
  V_f = sum_{X,Y} f_{X,Y} * P_{X,Y}

SIMILARITY:
  E[delta(P_{X,Y}, P_{X',Y'})] = k((X-X')/w_x) * k((Y-Y')/w_y)
  k() = Gaussian kernel

SEPARATION (decoding):
  s(w) = (mu_2 - mu_1) / (sigma_1 + sigma_2)
  Optimal: s >= 2-3 (requires small w)

SEPARATION (learning):
  s(p) = sqrt(p) * (mu_1 - mu_12) / (sigma_1 + sigma_12)
  Optimal: s ~ 0.8-1.2 (requires intermediate w)

CAPACITY:
  Learning capacity grows as sqrt(p)
  Decoding capacity grows with D

KERNEL TUNING:
  w_x ~ l_x = <|x - X|>_{co-activation}
  w_y ~ l_y = <|y - Y|>_{co-activation}

RESONATOR UPDATE:
  G_{A,n} = [M]_A ( H (circle) prod_{i!=A} G_{i,n-1} )
  Requires small w for all factors
```
