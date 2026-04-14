# HDC-Brain: A 300M Hyperdimensional Language Model with Bipolar Codebook

**Oleg Hasjanov**
*Independent Researcher, Tallinn, Estonia*
*oleg.hasjanov@example.com*

---

## Abstract

We present **HDC-Brain v14.1**, a 299M parameter language model built on hyperdimensional computing (HDC) principles as an alternative to the standard transformer architecture. The model replaces learned embeddings with a **bipolar (±1) codebook** trained via straight-through estimator (STE), and replaces the quadratic-parameter QKV projections with **multi-head binding attention** using only 12K parameters per layer — a 5000× reduction over equivalent transformers. Additional components include **thought loops** (iterative multi-pass reasoning through shared blocks), a **parallel-scan HDC memory** with learned mass/decay (constant-memory alternative to KV-cache), and an optional **CogitLayer** — an HDC associative memory that provides n-gram statistical hints and accelerates training.

The model was pretrained on 3B tokens of FineWeb-Edu for 88 hours on a single RTX 3090, reaching a validation BPB of **5.434**. After instruction finetuning on 140K mixed prompt-response pairs (Alpaca-GPT4, SlimOrca, Alpaca), the model reaches BPB **4.105** and produces coherent factual answers (e.g. correctly identifying Paris as the capital of France) and creative text. The bipolar codebook enables **16 MB** inference storage for a 32K vocabulary (versus 512 MB float32), making HDC-Brain a candidate architecture for edge deployment.

---

## 1. Introduction

Transformer-based language models have become the de-facto standard for natural language processing, but their deployment on resource-constrained devices remains challenging. A 1B parameter transformer requires ~4 GB of memory in float32 and relies on GPU-friendly float matrix multiplication. Edge devices (phones, embedded systems) would benefit from architectures built around binary operations (XNOR + POPCNT) that are dramatically faster on commodity hardware.

**Hyperdimensional Computing (HDC)** [Kanerva, 2009] offers a natural fit: in HDC, information is represented by very high-dimensional vectors (typically thousands of dimensions) whose components are binary (±1). Similarity is cosine similarity, binding is element-wise multiplication (XNOR for bipolar), and bundling is summation. HDC has been successfully applied to classification and reasoning tasks, but — to our knowledge — has not been extended to full generative language modeling at scale.

In this work we bridge HDC and language modeling. Our contributions are:

1. **STE Bipolar Codebook** — token embeddings are sign-constrained ±1 vectors, trained in float32 and binarized at forward pass via a straight-through estimator. This yields 1 bit per parameter at inference.
2. **Multi-Head Binding Attention** — instead of the three learned QKV projections (3D² parameters per layer), we use 3 learned binding vectors per head, applied via element-wise multiply. This reduces attention parameters from 67M to 12K at hidden size 4096.
3. **Thought Loops** — a learned multi-pass mechanism where the same block stack is applied K times with per-pass gating. Empirically, K=3 produces quality output while K=1 yields unusable text; we argue iterative refinement is structurally necessary when per-layer capacity is reduced.
4. **HDC Memory** — a parallel-scan recurrence with learned per-token mass (importance) and decay (forgetting), replacing KV-cache with O(D) memory regardless of sequence length.
5. **CogitLayer** — a non-parametric HDC associative memory of positional n-gram statistics. Added at train time as a free "hint" to token embeddings, it accelerates pretraining convergence by a factor of 2–3× at comparable quality on our 90K-pair mini-experiment.

We demonstrate that this architecture can be pretrained from scratch on 3B tokens and instruction-finetuned to produce a coherent dialogue model. The resulting 299M-parameter model fits into approximately 184 MB of memory at inference (codebook binary + controller int8).

---

## 2. Related Work

**Hyperdimensional Computing.** The framework of vector symbolic architectures (VSA) [Plate 1995, Kanerva 2009] provides the mathematical foundation for our architecture: high-dimensional vectors, binding (⊗), bundling (+), and cleanup (cosine search). Prior work has demonstrated HDC for classification [Neubert et al. 2019], reasoning [Komer et al. 2020], and symbolic reasoning (bAbI) [Rahimi et al. 2019], but generative language modeling has remained unexplored.

**Binary Neural Networks.** BinaryConnect [Courbariaux et al. 2015] and XNOR-Net [Rastegari et al. 2016] established that neural networks can be trained with binary weights using straight-through estimators. Our bipolar codebook extends this principle to token embeddings in a language model, which (to our knowledge) has not been done before at the 300M parameter scale.

**Efficient Language Models.** Recent work has explored architectures for efficient inference: Mamba [Gu & Dao, 2023] uses selective state-space models, RWKV [Peng et al. 2023] combines RNN and transformer features, and LLaMA-family models rely on transformer variants. Our architecture differs in that it *embraces* binary operations from the bottom up rather than quantizing a float network post-hoc.

**Thought Loops and Iterative Reasoning.** Universal Transformer [Dehghani et al. 2018] introduced depth-recurrent transformers. Our thought loops are similar in spirit but are gated by the model rather than halting by a confidence signal, and operate on bipolar representations.

---

## 3. Architecture

HDC-Brain consists of six components: (i) an STE bipolar codebook, (ii) a cyclic positional permutation, (iii) a stack of 8 HDC blocks, (iv) thought loops, (v) an output linear-tied to the codebook, and (vi) an optional CogitLayer. We describe each below.

### 3.1 STE Bipolar Codebook

Token embeddings are stored as a matrix $C \in \mathbb{R}^{V \times D}$ of float parameters (where $V = 32{,}000$, $D = 4096$). At forward pass we apply sign binarization with a scale:

$$
\alpha_i = \frac{1}{D} \sum_{j=1}^{D} |C_{i,j}|, \qquad
\hat{C}_{i,j} = \alpha_i \cdot \text{sign}(C_{i,j}).
$$

The gradient is passed through by the straight-through estimator: $\nabla C = \nabla \hat{C}$. At inference, only the sign bit is needed (and the scalar $\alpha_i$), giving 1 bit per codebook parameter — 16 MB for the full 32K × 4096 codebook.

### 3.2 Multi-Head Binding Attention

A standard multi-head attention layer uses four learned projections $W_Q, W_K, W_V, W_O \in \mathbb{R}^{D \times D}$, for $4D^2 = 67$M parameters per layer at $D = 4096$. We replace these with **binding vectors**: for each of $H = 4$ heads, three learned vectors $b_q^h, b_k^h, b_v^h \in \mathbb{R}^{D/H}$ (binarized via STE). The Q, K, V tensors for head $h$ are computed by element-wise multiplication:

$$
Q^h = x^h \odot \text{sign}(b_q^h), \quad
K^h = x^h \odot \text{sign}(b_k^h), \quad
V^h = x^h \odot \text{sign}(b_v^h).
$$

No output projection is required: in HDC, each dimension is independent, so heads simply write to their own slice of the representation. The total attention parameter count is $3HD/H = 3D = 12{,}288$ per layer, a 5461× reduction. Attention scores use sigmoid (not softmax) to match the HDC-style similarity:

$$
\text{attn}(Q, K) = \sigma(4 \cdot QK^\top / \sqrt{D/H}).
$$

### 3.3 HDC Memory

For each block we include an **HDC Memory** module that applies a parallel-scan recurrence with learned per-token mass $m_t \in (0,1)$ and decay $d_t \in (0,1)$:

$$
s_t = d_t \cdot s_{t-1} + m_t \cdot x_t.
$$

Unlike KV-cache, which grows linearly with sequence length, this memory is a single $D$-dimensional state. Computation is done via cumulative log-decay and a causal mask on the decay matrix for $O(T^2)$ compute but $O(D)$ state memory.

### 3.4 Thought Loops

The input passes through the block stack $K$ times. Each additional pass is gated:

$$
h^{(k+1)} = h^{(k)} + \sigma(g_k) \cdot (B(\text{LN}(h^{(k)}) + p_k) - h^{(k)}),
$$

where $B$ is the block stack, $p_k$ is a learned per-pass position encoding, and $g_k$ a learned gate. We train with $K = 3$ and evaluate with $K \in \{1, 2, 3, 4\}$.

### 3.5 CogitLayer (Optional)

CogitLayer maintains positional associative memories $M_0, M_1, \ldots, M_{n-1} \in \mathbb{R}^{V \times D}$, where $M_p[v]$ is the average code of tokens appearing $p+1$ positions after token $v$ in the training data. At train time, the hint for the $t$-th token is computed as:

$$
h_t = \frac{1}{n_t} \sum_{p=0}^{\min(n-1, t)} \frac{1}{p+1} \frac{M_p[x_{t-p-1}]}{\|M_p[x_{t-p-1}]\|}
$$

(with $n_t$ the number of contributing positions). The hint is added to token embeddings before the blocks, scaled by the current token-vector norm and a hyperparameter $\lambda = 0.1$. CogitLayer has **zero trainable parameters** — it is a data structure, not a module — and is preseeded from a subset of training data.

### 3.6 Parameter Budget

Total: 299,290,629 parameters (see Figure 3):
- Codebook: 131.1M (43.8%) — bipolar at inference
- Controller (8 blocks, FFN): 167.9M (56.1%)
- Other (attention + memory + LN + thought + output): 328K (0.11%)

![Parameter breakdown](figures/fig_params.png)

**Figure 3:** Parameter breakdown of HDC-Brain v14.1. The attention + memory + LN components that constitute the HDC "intelligence" are 328K parameters total — 0.11% of the model.

![Attention comparison](figures/fig_attn_comparison.png)

**Figure 4:** Attention parameters per layer, HDC-Brain vs standard transformer at $D = 4096$. 5461× reduction.

---

## 4. Training

### 4.1 Pretrain

We pretrain on **FineWeb-Edu** [Penedo et al. 2024], a 3B-token curated web dataset. Training runs on a single NVIDIA RTX 3090 (24 GB VRAM) for 88 hours.

**Hyperparameters:**
- Batch size: 16 × 8 (gradient accumulation) = 128 effective
- Sequence length: 512
- Tokens per step: 65,536
- Learning rate: 3e-4 (peak), 3e-5 (min), 500-step warmup, cosine decay
- Optimizer: AdamW, codebook LR = 0.1 × main LR (no weight decay)
- Mixed precision: bfloat16
- Thought loops: K = 3 during training
- Gradient checkpointing enabled

The model reaches **best BPB 5.434** at iter 57,500 (see Figure 1). Training is stable with gradient norm < 1 throughout.

![Pretrain curves](figures/fig_pretrain.png)

**Figure 1:** (Left) Validation BPB during pretraining. (Right) Train and validation cross-entropy loss. Plateau near iter 57,500 indicates dataset saturation at 1.25 epochs over 3B tokens.

### 4.2 Instruction Finetuning

From the best pretrained checkpoint we perform supervised instruction tuning. We construct a 140K-pair instruction dataset by combining:

- **Alpaca** [Taori et al. 2023]: 52K pairs, repeated 3× (factual simple responses)
- **Alpaca-GPT4** [Peng et al. 2023]: 40K pairs (GPT-4 responses)
- **SlimOrca** [Lian et al. 2023]: 50K pairs filtered (short, no code, no math)

We filter SlimOrca to responses 30–1500 characters, ASCII-dominant, with at most one code block. Prompts are formatted as:

```
### Instruction: {question}
### Response: {answer}
```

**Hyperparameters:**
- Batch size: 4 × 2 (accumulation) = 8 effective
- Learning rate: 1e-4 → 1e-5 cosine, 200-step warmup
- 15,000 iterations (~3 hours)
- CogitLayer: 3 positions, preseeded from 2M training tokens; hint weight λ = 0.1
- No gradient checkpointing (batch size 4 already fits)

The model reaches **best BPB 4.105** at iter 14,000, down from the base 5.434.

![Finetune curves](figures/fig_finetune.png)

**Figure 2:** (Left) Instruction finetuning loss. (Right) Validation BPB improves from the pretrained base 5.434 to 4.105 over 15K iterations.

---

## 5. Results

### 5.1 Qualitative Generation

After finetuning, the model produces coherent responses to novel prompts. Selected generations (temperature 0.7, top-k 40, repetition penalty 1.3):

| Prompt | Response |
|--------|----------|
| What is the capital of France? | "The capital city of France is Paris." |
| Explain what is artificial intelligence. | "Artificial Intelligence (AI) and machine learning are two branches of computer science that can help companies make decisions, improve their processes by making data-driven predictions or decisions based on past experiences..." |
| Write a short poem about the ocean. | "A sea of glistens, A warm place that is with joy and grace, Shall we go, The love of life's sight so grand, A moment to embrace new things, a serene evening stroll on the ground." |
| How does a car engine work? | "A vehicle typically has two main components, such as the transmission and storage of power plants within cities. These include an engine's fuel system (e.g., a car), fuel-efficient vehicles such as powered automatic engines or electric cars..." (partially coherent, factually incomplete) |

The model correctly answers the France question — demonstrating factual recall acquired during instruction tuning — while still exhibiting the common small-model failure mode of hallucination on mechanical questions (car engine).

### 5.2 CogitLayer Ablation

In a controlled mini-experiment (37M parameters, Alpaca instructions, 15-minute time budget on Apple M3 MPS):

| Metric | Baseline | +CogitLayer | Ratio |
|--------|----------|-------------|-------|
| Validation loss | 7.92 | 7.16 | 0.90× |
| Perplexity | 2764 | 1292 | 2.1× |
| Top-1 accuracy | 2.5% | 11.9% | **4.7×** |
| Top-10 accuracy | 20.5% | 34.3% | 1.7× |

CogitLayer provides substantial early training speedup by pre-loading positional n-gram statistics. The effect diminishes at scale but remains positive (best BPB 4.105 with CogitLayer vs 4.216 without on full instruction finetuning).

### 5.3 Inference Efficiency

At inference, only sign bits of the codebook and quantized controller are needed:
- Codebook: 32,000 × 4096 / 8 = **16.0 MB** (1 bit/param)
- Controller (8 blocks, int8): **167.9 MB**
- Memory state (HDC parallel scan): **16 KB per context** (constant in sequence length)
- **Total: ≈ 184 MB**

A float16 transformer of equivalent parameter count would require ≈ 600 MB + a KV-cache that grows linearly with context.

---

## 6. Discussion

### 6.1 When HDC wins

HDC-Brain is a deliberate bet on architectures whose primitive operations are cheap on commodity hardware. Binary binding can be implemented as XNOR + POPCNT on any CPU, at 8× the throughput of float multiply-accumulate. The constant-memory HDC state makes long-context inference cheap (no KV-cache explosion). This makes HDC-Brain a candidate for phone-class and embedded deployment at 300M–1B scale.

### 6.2 When transformers win

For maximum quality at the frontier, transformers retain the advantage: full-rank QKV projections are strictly more expressive than binding vectors. Our current results show that HDC-Brain underperforms a size-matched transformer on raw language modeling (estimated BPB gap of ~1.0), but the gap narrows with CogitLayer and instruction tuning. We do not claim parity with the frontier; we claim a different point on the efficiency-quality tradeoff.

### 6.3 Limitations

- **No RLHF.** The model is only instruction-tuned (SFT). Reinforcement learning from human feedback would likely substantially improve response quality.
- **Limited context length.** Trained at 512 tokens. HDC memory is constant-size, so longer contexts are possible but untested.
- **Empirical CogitLayer instability.** Live observation of batches during finetuning caused intermittent CUDA deadlocks on our hardware; we worked around this by using preseed-only mode. Root cause is under investigation.
- **Dataset quality.** A large fraction of our instruction dataset was filtered ShareGPT/SlimOrca, which is known to be noisy. A higher-quality SFT dataset (e.g. OpenHermes 2.5 with stricter filtering) would likely improve downstream metrics.

### 6.4 Future work

- **Fully binary training.** Replace STE with a direct binary update rule (e.g. error-weighted bit voting) to eliminate the float shadow weights.
- **HDC-native adapters.** A learned binding vector per block (4096 bits per block × 8 blocks = 4 KB total) may provide a lightweight alternative to LoRA for task specialization.
- **Larger scale.** Scale to 1–3B parameters on a larger compute budget to test whether HDC-Brain keeps pace with transformers.

---

## 7. Conclusion

We have shown that hyperdimensional computing can be scaled to full-scale generative language modeling. HDC-Brain v14.1, a 299M-parameter model, achieves BPB 5.434 on FineWeb-Edu and, after instruction tuning, produces factually correct responses to simple questions. The architecture achieves 5461× reduction in attention parameters over a standard transformer while remaining trainable with standard tools (PyTorch, AdamW, STE). The bipolar codebook enables 16 MB inference storage for a 32K vocabulary, making the architecture a candidate for edge-device deployment. We release our training code, instruction finetuning pipeline, and the best pretrained and finetuned checkpoints to the research community.

---

## References

*(To be populated — key references: Kanerva 2009, Rahimi 2019, Courbariaux 2015, Rastegari 2016, Vaswani 2017, Dehghani 2018, Touvron 2023, Gu & Dao 2023, Peng 2023, Penedo 2024, Taori 2023, Lian 2023.)*

---

## Appendix A: Full Hyperparameters

### Pretrain

| Parameter | Value |
|-----------|-------|
| HDC dimension | 4096 |
| Vocabulary | 32,000 (BPE, English) |
| Blocks | 8 |
| Heads | 4 |
| Controller inner dim | 2,560 |
| Thought loops (train) | 3 |
| Batch size | 16 |
| Gradient accumulation | 8 |
| Sequence length | 512 |
| Tokens per step | 65,536 |
| Max iterations | 120,000 (stopped at 57,500 best) |
| Learning rate peak | 3e-4 |
| Learning rate min | 3e-5 |
| Warmup steps | 500 |
| LR schedule | Cosine |
| Optimizer | AdamW, β=(0.9, 0.95) |
| Codebook LR multiplier | 0.1 |
| Weight decay (main) | 0.05 |
| Weight decay (codebook) | 0.0 |
| Dropout | 0.1 |
| Gradient clip | 1.0 |
| Precision | bfloat16 mixed |
| Hardware | 1× NVIDIA RTX 3090 |
| Training time | 88 hours |

### Finetune

| Parameter | Value |
|-----------|-------|
| Batch size | 4 |
| Gradient accumulation | 2 |
| Sequence length | 512 |
| Learning rate peak | 1e-4 |
| Learning rate min | 1e-5 |
| Warmup steps | 200 |
| Max iterations | 15,000 |
| CogitLayer positions | 3 |
| CogitLayer hint weight | 0.1 |
| CogitLayer preseed tokens | 2,000,000 |
| Dropout | 0.05 |
| Training time | 3 hours |
