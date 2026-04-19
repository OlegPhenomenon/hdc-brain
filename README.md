# HDC-Brain

A 300M-parameter hyperdimensional language model with a bipolar (±1) codebook and binding-based attention — built as an alternative to the standard transformer stack.

> **Paper:** [`docs/paper/paper.md`](docs/paper/paper.md) — *HDC-Brain: A 300M Hyperdimensional Language Model with Bipolar Codebook* (Oleg Hasjanov, 2026).

## What this is

HDC-Brain replaces three components of a standard transformer with HDC-native primitives:

- **STE bipolar codebook** — token embeddings are sign-constrained ±1 vectors (1 bit per parameter at inference). 32K × 4096 = **16 MB** versus 512 MB in float32.
- **Multi-head binding attention** — 3 learned binding vectors per head instead of QKV projections. **12,288 params/layer** versus 67M in a transformer of equivalent width (5461× reduction).
- **Thought loops** — iterative K-pass reasoning through a shared block stack; K = 3 is best (K = 1 loses ≈ 0.3 bits/token).
- **Parallel-scan HDC memory** — learned mass/decay recurrence, O(D) state regardless of sequence length, replacing KV-cache.

Total: **299M parameters**, pretrained on 3B tokens of FineWeb-Edu, instruction-finetuned on 75M tokens of filtered prompt-response data.

## Key numbers

| Metric | Value |
|---|---|
| Parameters | 299,290,629 |
| Pretrain loss (FineWeb-Edu val) | **5.434 bits/token** ≈ **1.25 bits/byte** |
| Finetune loss (quality_v3 val) | **3.521 bits/token** |
| Gap to SmolLM-360M on raw bytes | +0.44 bits/byte (behind) |
| Gap to GPT-2-medium on raw bytes | −0.13 bits/byte (ahead) |
| Attention params per layer | 12,288 (vs 67M dense) |
| Codebook storage (inference) | 16 MB |

See paper §5 for measured baselines and §5.1 for qualitative strengths and failure modes.

## Repository layout

```
hdc-brain-v14.1/
  hdc_brain_v14_1.py     Model (codebook, binding attention, thought loops, HDC memory)
  train.py               Pretrain on FineWeb-Edu
  finetune_v3.py         Instruction finetune → BPB 3.521
  prep_quality_v3.py     Build quality_v3 SFT dataset (75M tokens)
  prepare_data.py        Tokenize FineWeb-Edu for pretrain
  chat.py                Interactive CLI
  benchmark.py           Inference efficiency (paper §5.2)
  test_memory.py         HDC memory unit tests
  bpe_en_32k.model       SentencePiece tokenizer (shipped)
  server_setup.sh        vast.ai / Lambda / RunPod bootstrap
docs/paper/
  paper.md, figures/, make_plots.py
docs/experiments/
  finetune_v3.log        Source log for numbers cited in paper
```

Pretrained weights are not committed to git — download from Hugging Face (links below).

## Quick start

Download the finetuned checkpoint and chat on CPU or GPU.

Requires **Python 3.10+** and **PyTorch ≥ 2.4** (for safe `weights_only=True` checkpoint loading).

```bash
git clone https://github.com/OlegPhenomenon/hdc-brain.git
cd hdc-brain/hdc-brain-v14.1
pip install -r requirements.txt

# Download best_finetune_v3_v14_1.pt into ./weights/
huggingface-cli download olegphenomenon/hdc-brain-v14.1-finetune-v3 \
    best_finetune_v3_v14_1.pt --local-dir ./weights

python chat.py                  # CPU inference (default)
python chat.py --device mps     # Apple Silicon (9–18 tok/s on M3)
python chat.py --device cuda    # NVIDIA GPU
```

## Reproducing the paper

All numbers in the paper were produced on a single RTX 3090 (24 GB).

**1. Pretrain (≈ 88 h):**

```bash
cd hdc-brain-v14.1
python prepare_data.py          # downloads FineWeb-Edu, tokenizes to train.bin / val.bin
python train.py                 # → best_hdc_brain_v14_1.pt, val loss ≈ 5.434 bits/token
```

**2. Finetune (≈ 5.7 h):**

```bash
python prep_quality_v3.py       # downloads OpenHermes 2.5, TULU-3, Alpaca-GPT4 ×3,
                                # Alpaca, Dolly-15K, WizardLM — filters to 591K pairs
python finetune_v3.py           # → best_finetune_v3_v14_1.pt, best val at step 20K
                                # (val loss ≈ 2.44 → BPB 3.521)
```

**3. Reproduce figures and tables:**

```bash
cd docs/paper
python make_plots.py            # regenerates fig_pretrain, fig_finetune, fig_params,
                                # fig_attn_comparison
```

**4. Thought-loops ablation (Table 1 in §3.4):** runs in a few minutes; snippet provided in paper appendix.

## Weights (Hugging Face)

Two checkpoints are published under **CC BY-NC 4.0** (research / non-commercial use free; commercial use requires a separate license):

- [`olegphenomenon/hdc-brain-v14.1-base`](https://huggingface.co/olegphenomenon/hdc-brain-v14.1-base) — pretrain checkpoint, 3.3 GB (float32)
- [`olegphenomenon/hdc-brain-v14.1-finetune-v3`](https://huggingface.co/olegphenomenon/hdc-brain-v14.1-finetune-v3) — SFT checkpoint used by `chat.py`, 1.1 GB

## Honest limitations

This is a feasibility study, not a frontier model:

- Single run, no hyperparameter sweep, no seed averaging
- Arbitrary factual recall is poor (e.g. "capital of Russia" → "Tokyo"); arithmetic and code generation fail
- Undertrained relative to Chinchilla (3B tokens / 299M params ≈ 10:1 vs 20:1)
- Compute advantage of the bipolar codebook requires custom XNOR/POPCNT kernels — **not implemented here**; only the *storage* advantage is realised
- Codebook is random bipolar with STE; semantic initialisation (e.g. FastText → sign) is deferred to future work

Full discussion in paper §6.

## Citation

```bibtex
@article{hasjanov2026hdcbrain,
  author  = {Oleg Hasjanov},
  title   = {HDC-Brain: A 300M Hyperdimensional Language Model with Bipolar Codebook},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {Preprint}
}
```

*(arXiv ID added after submission.)*

## License

Code: [Apache License 2.0](LICENSE) — unrestricted use.
Weights: CC BY-NC 4.0 (free for research, academic, and personal non-commercial use; commercial use requires a separate license — contact `oleg.phenomenon@gmail.com`).

## Contact

Oleg Hasjanov — oleg.phenomenon@gmail.com — Tallinn, Estonia.
