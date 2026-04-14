# HDC-Brain: Paper Outline

## Title options
1. **"HDC-Brain: A 300M Hyperdimensional Language Model with Binary Codebook"**
2. **"Beyond Transformers: Language Modeling with Hyperdimensional Computing"**
3. **"HDC-Brain: Bipolar Codebook and Binding Attention for Efficient Language Models"**

Recommended: #1 — most specific, most scientific.

## Abstract (150-200 words)

We present HDC-Brain, a 299M parameter language model built on hyperdimensional computing (HDC) principles rather than standard transformer architecture. Key innovations:

1. **STE Bipolar Codebook**: Token embeddings are ±1 vectors (1-bit per parameter at inference), trained via Straight-Through Estimator
2. **Multi-Head Binding Attention**: Replaces QKV projections with element-wise HDC binding, using 4000× fewer attention parameters
3. **Thought Loops**: Iterative reasoning through same blocks, 3 passes proven essential for quality
4. **HDC Memory**: Parallel scan recurrence with learned mass/decay, constant memory vs linear KV-cache
5. **CogitLayer**: Optional HDC associative memory for 2-3× training speedup

Trained on 3B tokens of FineWeb-Edu, achieves BPB 5.434 base and BPB 4.105 after instruction tuning on Alpaca-GPT4 + SlimOrca. Model answers factual questions ("The capital city of France is Paris") and generates coherent text. Binary codebook enables 16 MB inference weight for 32K vocabulary versus 512 MB for float32 equivalent.

## Sections

### 1. Introduction
- Motivation: edge deployment needs, transformer limitations
- Why HDC: binary operations, constant memory, XNOR/POPCNT hardware
- Contribution summary

### 2. Related Work
- Hyperdimensional computing (Kanerva, VSA)
- Binary neural networks (XNOR-Net, BinaryConnect)
- Efficient LLMs (LLaMA, Mamba, RWKV)
- Differentiable binary training (STE)

### 3. Architecture

#### 3.1 STE Bipolar Codebook
- Math: forward = sign(w), backward = identity
- Why ±1 instead of ±α: alpha scaling preserves magnitude
- 32K × 4096 bits = 16 MB

#### 3.2 Multi-Head Binding Attention
- Standard attention: 3×D×D = 50M params/layer
- Our method: 4×3×D/H = 12K params/layer
- Sigmoid attention, causal mask
- No output projection needed

#### 3.3 HDC Memory
- Parallel scan recurrence
- Learned mass (importance) + decay (forgetting)
- O(T) memory vs O(T²) KV-cache

#### 3.4 Thought Loops
- Multi-pass through same blocks
- Learned gates per thought
- 3 passes = quality, 1 pass = garbage (empirical)

#### 3.5 CogitLayer (Optional)
- HDC associative memory for n-gram statistics
- Provides hint to token embeddings
- Not part of parameters — pure data structure

### 4. Training

#### 4.1 Pretrain
- Dataset: FineWeb-Edu 3B tokens
- 57,500 iters × 65,536 tok/step = 3.8B tokens seen (1.25 epochs)
- bf16 mixed precision, RTX 3090
- 88 hours wall time

#### 4.2 Instruction Finetune
- Alpaca-GPT4 (40K) + SlimOrca filtered (50K) + Alpaca ×3 (52K)
- 15,000 iterations, 3 hours
- CogitLayer preseed only (no live update)

### 5. Results

#### 5.1 Training Curves
- Plot: BPB vs iters during pretrain
- Plot: Loss vs iters during finetune
- Plot: CogitLayer vs baseline comparison (from mini experiment)

#### 5.2 Qualitative Examples
- Pre-finetune generation (coherent but undirected)
- Post-finetune: Paris question, AI explanation, ocean poem
- Failure modes: factual errors, hallucinations

#### 5.3 Efficiency Metrics
- Parameter breakdown: codebook/attention/controller/other
- Inference memory: 16 MB codebook + 167 MB controller = 183 MB
- vs. equivalent transformer: 1.2 GB float16

### 6. Discussion

- When HDC wins: edge devices, low-power inference
- When transformer wins: maximum quality, long context
- CogitLayer as general acceleration technique
- Limitations: no RLHF, limited context length

### 7. Conclusion
- Summary of contributions
- Future: binary training (replace STE), HDC-native adapters, larger scale

## Figures Needed

1. **Architecture diagram** — block structure with Memory/Attention/Controller
2. **Training curve pretrain** — BPB & loss vs iters over 88h
3. **Training curve finetune** — Loss vs iters with BPB evals
4. **CogitLayer comparison** — from mini experiment (4.7× speedup)
5. **Parameter breakdown** — pie chart: codebook/blocks/other
6. **Generation examples** — table with before/after finetune

## Tables Needed

1. **Architecture parameters** — our vs transformer equivalent
2. **Training hyperparameters** — pretrain and finetune
3. **Benchmark comparison** — BPB vs SmolLM2-360M, GPT-2, Pythia-160M
4. **Generation quality** — manual scoring on test prompts

## Data Sources

- `docs/experiments/v14.1-finetune-mixed/experiment.jsonl` — pretrain curves
- `docs/experiments/v14.1-finetune-mixed/finetune.log` — finetune metrics
- `hdc-brain-v14.1/weights/best_*.pt` — checkpoints
- `hdc-brain-v14.2/` — architecture code

## Format

- LaTeX with arXiv template (NeurIPS 2024 style recommended)
- 8-10 pages main content + appendix
- Bibliography: 30-40 references
