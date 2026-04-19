# HDC Advances 2024-2026: Practical New Operations & Learning Algorithms

Research survey completed 2026-03-30. Focus: practical techniques applicable to binary HDC language modeling.

---

## 1. New HDC Operations Beyond Bind/Bundle/Permute

### 1.1 Self-Attention Based Semantic Decomposition (Yeung et al., March 2024)
- **arXiv**: 2403.13218
- **Core idea**: Replaces the iterative resonator network update rule with a self-attention mechanism for factorizing composite hypervectors. The attention-based FHRR resonator outperforms the original resonator network by a large margin as number of factors and bundled terms increases. Requires significantly fewer iterations to converge.
- **Relevance to 16K binary codebook**: HIGH. Attention-based factorization could replace Hebbian codebook updates. The attention mechanism operates directly in HD space, making it a potential drop-in for codebook search/update.

### 1.2 Generalized Holographic Reduced Representations (Yeung et al., May 2024)
- **arXiv**: 2405.09689
- **Core idea**: Extends FHRR with a flexible, non-commutative binding operation. Improved memorization capacity and better decoding accuracy for compositional structures. Non-commutativity is key for sequence representation (order matters).
- **Relevance to 16K binary codebook**: MEDIUM. Non-commutative binding is more natural for language (word order matters), but requires complex-valued vectors, not binary.

### 1.3 Modular Composite Representation (MCR) (Angioli et al., November 2025)
- **arXiv**: 2511.09708
- **Core idea**: Represents information with high-dimensional INTEGER vectors using modular arithmetic. Outperforms binary and integer vectors in capacity while approaching complex-valued representations at a fraction of memory. 3.08x faster execution and 2.68x lower energy vs binary spatter codes.
- **Relevance to 16K binary codebook**: HIGH. MCR with small modulus (e.g., mod 4 or mod 8) could replace binary vectors, giving much better capacity per dimension while staying CPU-friendly (integer ops only). Could allow shrinking from 16K binary to ~4K integer vectors.

### 1.4 Linear Codes for HDC (March 2024)
- **arXiv**: 2403.03278
- **Core idea**: Uses random linear codes (subspaces over Boolean field) to solve the recovery/factorization problem. Strictly faster than resonator networks, often by an order of magnitude.
- **Relevance to 16K binary codebook**: HIGH. Direct binary operations. Could provide a faster alternative to resonator-based codebook lookup.

---

## 2. HDC Learning Algorithms That Are NOT Hebbian

### 2.1 THDC: Training HDC with Backpropagation (January 2026)
- **arXiv**: 2602.00116
- **Core idea**: End-to-end HDC training via backpropagation. Replaces random item memory with TRAINABLE embeddings + one-layer binary neural network for class representations. Achieves same accuracy as SOTA HDC with dimension reduced from 10,000 to 64. Uses STE (Straight-Through Estimator) for binary gradients.
- **Relevance to 16K binary codebook**: CRITICAL. This is exactly what you tried in v10.1. The paper validates the approach and provides the architecture details. Key insight: train BOTH the item memory (codebook) AND the associative memory jointly.

### 2.2 DecoHD: Decomposed HDC (Yun et al., November 2025)
- **arXiv**: 2511.03911
- **Core idea**: Learns directly in a decomposed HD parameterization: small shared per-layer channels with multiplicative binding across layers and bundling at the end. Reduces trainable parameters by up to 97% with negligible accuracy loss. Preserves holographic properties.
- **Relevance to 16K binary codebook**: HIGH. Decomposed codebook = instead of storing 256 x 16K binary vectors, store ~16 x 16K basis vectors and compose tokens as products. Massive memory savings. Could enable much larger effective codebooks.

### 2.3 LeHDC: Learning-Based HDC Classifier (2022, still referenced as baseline)
- **arXiv**: 2203.09680
- **Core idea**: Enhanced representational precision via learning-based encoder optimization. Foundation for later THDC work.
- **Relevance to 16K binary codebook**: MEDIUM. Predecessor to THDC, less powerful.

### 2.4 LDC: Low-Dimensional Computing (Duan et al., February 2025)
- **arXiv**: 2502.14075
- **Core idea**: Reduces vector dimension by ~100x while maintaining accuracy through gradient-based optimization. Key finding: batch normalization and knowledge distillation are critical (BN free at inference, KD boosts confidence). Accepted at CPAL 2025.
- **Relevance to 16K binary codebook**: HIGH. If gradient optimization can maintain quality at 160 dimensions instead of 16K, the entire approach becomes vastly more efficient. BN insight is directly applicable.

### 2.5 Clo-HDnn: Continual On-Device Learning (July 2025)
- **arXiv**: 2507.17953
- **Core idea**: Kronecker HD Encoder + gradient-free continual learning to update class hypervectors. Progressive search for optimal architecture.
- **Relevance to 16K binary codebook**: MEDIUM. Gradient-free continual learning could be useful for online codebook adaptation, but primarily targets classification.

---

## 3. Multi-Level / Hierarchical Bundling

### 3.1 DecoHD (see 2.2 above)
- Multi-layer decomposition with binding across layers and bundling at end = hierarchical composition.

### 3.2 HyperGraphX: Graph Learning with HDC (October 2025)
- **arXiv**: 2510.23980
- **Core idea**: Combines graph convolution with binding and bundling operations. Multi-hop message passing = hierarchical bundling across graph neighborhoods. 9561x faster than GCNII on GPU. 15.5pp higher accuracy than GCN.
- **Relevance to 16K binary codebook**: LOW-MEDIUM. Graph structure is different from sequential language, but the multi-hop bundling pattern could inspire hierarchical context encoding (bundle local context, then bind with position, then bundle broader context).

---

## 4. HDC for Sequence Modeling / Language

### 4.1 Hyperdimensional Probe (September 2025)
- **arXiv**: 2509.25045
- **Core idea**: Uses VSA operations to decode human-interpretable information from LLM latent representations. Shows that HD algebra can probe and decompose transformer hidden states. Works with any HuggingFace autoregressive model.
- **Relevance to 16K binary codebook**: MEDIUM. Not a generative model, but proves that HD space can meaningfully represent the same information as transformer hidden states. Could inform how to structure codebook to capture linguistic features.

### 4.2 AR-HDC: Autoregressive HD Time Series (February 2024)
- **arXiv**: 2402.01999
- **Core idea**: Reframes online nonlinear time-series forecasting as linear hyperdimensional forecasting. Maps nonlinear low-D data to high-D space for fast online prediction. One-element-ahead autoregressive prediction.
- **Relevance to 16K binary codebook**: HIGH. This is the closest existing work to what you're building. Autoregressive prediction in HD space = next-token prediction. The mapping from nonlinear to linear HD space is key insight: if you can linearize the prediction problem in HD space, you can use simple operations for next-token prediction.

### 4.3 BiHDTrans: Binary HD Transformer (September 2025)
- **arXiv**: 2509.24425
- **Core idea**: Integrates self-attention into the HD computing paradigm for time series. Binary operations throughout. 14.47% better than SOTA HD models. 64% dimensionality reduction with only 1-2% accuracy loss. 4.4x smaller model, 49.8% less latency.
- **Relevance to 16K binary codebook**: CRITICAL. This is the most directly relevant paper. Binary HD + attention + sequence modeling. The architecture shows how to do attention in binary HD space efficiently. The 64% dimensionality reduction means you might get away with ~6K dimensions instead of 16K.

---

## 5. Resonator Networks: New Developments

### 5.1 Self-Attention Resonator (Yeung et al., March 2024)
- (See 1.1 above) Attention-based update rule dramatically improves convergence, avoids limit cycles.

### 5.2 Noise Injection for Limit Cycles (Late 2024)
- **arXiv**: 2412.00354
- **Core idea**: Injects noise to escape limit cycles in resonator networks, improving convergence. Simple technique, significant practical improvement.
- **Relevance to 16K binary codebook**: MEDIUM. If using resonator-based codebook lookup, noise injection is a free improvement.

### 5.3 Convolutional Sparse Coding + Resonator Networks (April 2024)
- **arXiv**: 2404.19126
- **Core idea**: Combines convolutional sparse coding with resonator networks for visual scene factorization. Sparse codes improve convergence speed.
- **Relevance to 16K binary codebook**: LOW. Vision-specific, but the sparse coding insight is transferable.

---

## 6. HDC + Attention Mechanisms

### 6.1 Recasting Self-Attention with HRR (Chou et al., 2023)
- **arXiv**: 2305.19534
- **Core idea**: Replaces standard self-attention with HRR-based attention. O(TH log H) time, O(TH) space. Converges in 10x fewer epochs. Enables very long sequences.
- **Relevance to 16K binary codebook**: HIGH. If adapted to binary HD space, this could provide an efficient attention mechanism for your model. The log(H) factor means 16K dims would cost ~14 multipliers per step.

### 6.2 Attention as Binding (December 2025)
- **arXiv**: 2512.14709
- **Core idea**: Interprets transformer self-attention as approximate VSA operations. Queries/keys = role spaces, values = fillers, attention weights = soft unbinding, residual = superposition. Proposes VSA-inspired architectural biases: explicit binding/unbinding heads and hyperdimensional memory layers.
- **Relevance to 16K binary codebook**: HIGH (theoretical). Provides a principled bridge between your HD approach and transformer attention. The proposed "binding/unbinding heads" could be implemented in binary HD space as your model's attention mechanism.

### 6.3 BiHDTrans (see 4.3 above)
- Practical implementation of attention in binary HD space for sequence modeling.

---

## 7. Learnable HD Encoders

### 7.1 THDC (see 2.1) - Trainable item memory via backprop + STE
### 7.2 LDC (see 2.4) - Gradient-based optimization reducing dimensions 100x
### 7.3 NeuroHD-RA (July 2025)
- **arXiv**: 2507.14184
- **Core idea**: Learnable RR-block encoder + BinaryLinear HD projection layer. Jointly optimized with cross-entropy + proxy-based metric loss. "Neural distillation" into HD space.
- **Relevance to 16K binary codebook**: HIGH. The dual loss (cross-entropy + metric) approach could help codebook learning. Metric loss encourages similar tokens to have similar HD vectors, which is important for language.

### 7.4 LARS-VSA: Learning Abstract Rules (May 2024)
- **arXiv**: 2405.14436
- **Core idea**: Neuro-symbolic architecture combining symbolic VSA reasoning with connectionist learning. Learns abstract rules from limited samples using HD representations.
- **Relevance to 16K binary codebook**: LOW-MEDIUM. More focused on abstract reasoning than language modeling.

---

## 8. Sparse Binary HDC

### 8.1 Sparse Binary Representation Learning for Knowledge Tracing (January 2025)
- **arXiv**: 2501.09893
- **Core idea**: Learns sparse binary representations via binarization method that is fully trainable with SGD. Each bit indicates presence/absence of a learned feature. Sparsity is a hyperparameter.
- **Relevance to 16K binary codebook**: HIGH. Directly applicable: learn sparse binary codebook entries with gradient descent. Sparsity control = capacity control. The SGD-trainable binarization is exactly what you need.

### 8.2 iEEG Seizure Detection with Sparse HDC (November 2025)
- **arXiv**: 2511.05503
- **Core idea**: Sparse HDC with compressed item memory (CompIM). Reduces energy consumption. Introduces sparsity as explicit hyperparameter affecting capacity vs efficiency tradeoff.
- **Relevance to 16K binary codebook**: MEDIUM. CompIM could reduce codebook storage. Sparsity analysis provides theoretical backing for choosing optimal density.

---

## Top 5 Most Relevant Papers for Your Project

Ranked by direct applicability to learning a 16K-token binary codebook on CPU:

1. **BiHDTrans (2509.24425)** - Binary HD + attention + sequence modeling. The closest architecture to what you're building. Study their attention mechanism.

2. **THDC (2602.00116)** - Validates trainable codebook with STE (what you did in v10.1). Shows it works, gives architecture details. Key: train item memory AND associative memory jointly.

3. **AR-HDC (2402.01999)** - Autoregressive prediction in HD space. Direct precedent for next-token prediction with HDC.

4. **MCR (2511.09708)** - Integer modular vectors instead of binary. 3x faster, better capacity. Could replace your binary vectors with mod-4 integers for massive capacity improvement while staying CPU-friendly.

5. **DecoHD (2511.03911)** - Decomposed codebook: store basis vectors, compose tokens as products. 97% parameter reduction. Could enable 64K+ effective vocabulary.

---

## Practical Recommendations for v15/v16

1. **Replace binary with MCR (mod 4 or mod 8)**: Better capacity per dimension, still integer-only CPU ops
2. **Use STE for codebook training** (validated by THDC): Train the codebook end-to-end, not with Hebbian
3. **Add attention in HD space** (from BiHDTrans): Binary attention is feasible and effective
4. **Decompose the codebook** (from DecoHD): Store 16 basis vectors, compose 256 tokens as bound products
5. **Use self-attention resonator** for codebook lookup (from Yeung 2024): Faster convergence, no limit cycles
6. **Sparse binary with SGD** (from SBRKT): If staying binary, make sparsity a learnable parameter

## Sources

- [Self-Attention Semantic Decomposition in VSA](https://arxiv.org/abs/2403.13218)
- [Generalized Holographic Reduced Representations](https://arxiv.org/abs/2405.09689)
- [Efficient HDC with Modular Composite Representations](https://arxiv.org/abs/2511.09708)
- [Linear Codes for HDC](https://arxiv.org/abs/2403.03278)
- [THDC: Training HDC with Backpropagation](https://arxiv.org/abs/2602.00116)
- [DecoHD: Decomposed HD Classification](https://arxiv.org/abs/2511.03911)
- [LDC: Low-Dimensional Computing](https://arxiv.org/abs/2502.14075)
- [Clo-HDnn: Continual On-Device Learning](https://arxiv.org/abs/2507.17953)
- [HyperGraphX: Graph Learning with HDC](https://arxiv.org/abs/2510.23980)
- [Hyperdimensional Probe for LLMs](https://arxiv.org/abs/2509.25045)
- [AR-HDC: Online Time Series Forecasting](https://arxiv.org/abs/2402.01999)
- [BiHDTrans: Binary HD Transformer](https://arxiv.org/abs/2509.24425)
- [Noise in Resonator Factorizers](https://arxiv.org/abs/2412.00354)
- [Convolutional Sparse Coding + Resonator Networks](https://arxiv.org/abs/2404.19126)
- [Recasting Self-Attention with HRR](https://arxiv.org/abs/2305.19534)
- [Attention as Binding](https://arxiv.org/abs/2512.14709)
- [NeuroHD-RA](https://arxiv.org/abs/2507.14184)
- [LARS-VSA](https://arxiv.org/abs/2405.14436)
- [Sparse Binary Representation Learning](https://arxiv.org/abs/2501.09893)
- [Sparse HDC for Seizure Detection](https://arxiv.org/abs/2511.05503)
- [LeHDC](https://arxiv.org/abs/2203.09680)
