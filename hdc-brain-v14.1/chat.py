"""
Interactive chat with HDC-Brain v14.1 finetuned model.

Usage:
    python chat.py                    # default: CPU (honest edge-style measurement)
    python chat.py --device mps       # Apple GPU on M-series (faster, NOT edge)
    python chat.py --device cuda      # NVIDIA GPU (dev machine / server)
    python chat.py --base             # use pretrained base (no instruction tune)
    python chat.py --max-tokens 300   # longer responses
    python chat.py --temperature 0.5  # more focused

Note on "edge" benchmarking:
    CPU is the honest reference for edge-device claims — phones and embedded
    systems don't have discrete GPUs or MPS. MPS on Mac uses the Apple GPU,
    which is *not* representative of phone-class hardware. Also note that even
    on CPU, this reference implementation performs float matmul on sign-
    quantized weights; a true XNOR/POPCNT kernel would be faster still but is
    not implemented here.

Measures tokens/sec per response so you can judge inference speed on your hardware.
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm

from hdc_brain_v14_1 import create_model


HERE = Path(__file__).parent
TOKENIZER_PATH = HERE / "bpe_en_32k.model"
# Default: v3 finetune on quality_v3 (OpenHermes+TULU+Alpaca+Dolly+WizardLM, 75M tokens)
# Best BPB 3.521 — CogitLayer-free, standalone.
FINETUNE_CKPT = HERE / "weights" / "best_finetune_v3_v14_1.pt"
FINETUNE_CLEAN_CKPT = HERE / "weights" / "best_finetune_clean_v14_1.pt"
BASE_CKPT = HERE / "weights" / "best_hdc_brain_v14_1.pt"


def pick_device(requested):
    # Honest default: CPU. MPS/CUDA must be requested explicitly,
    # because paper claims about edge deployment should be measured on CPU.
    if requested:
        return requested
    return "cpu"


@torch.no_grad()
def generate(model, sp, prompt, device, max_tokens=150, temperature=0.7,
             top_k=40, rep_penalty=1.3, n_thoughts=3, stream=True):
    """Generate a response token-by-token, streaming to stdout.

    Returns (response_text, num_tokens, seconds).
    """
    model.eval()
    ids = sp.encode(prompt)
    idx = torch.tensor([ids], device=device, dtype=torch.long)

    # Track only newly generated tokens
    start_len = idx.size(1)
    eos_id = 2  # </s> in SentencePiece default
    stop_markers = ["### Instruction", "### Response"]

    t0 = time.time()
    for _ in range(max_tokens):
        ctx = idx[:, -model.max_seq_len:]
        tokens = model._ste_encode(ctx)
        tokens = model._cyclic_position(tokens)
        h = model.thought_loop(tokens, model.blocks, n_thoughts, False)
        h = model.output_ln(h)
        logits = F.linear(h, model.codebook) * model.output_scale
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        # Repetition penalty on recent tokens
        recent = set(idx[0, -50:].tolist())
        for tid in recent:
            if logits[0, tid] > 0:
                logits[0, tid] /= rep_penalty
            else:
                logits[0, tid] *= rep_penalty

        # Top-k filter
        if top_k and top_k < logits.size(-1):
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        nid = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nid], dim=1)

        # Decode current suffix for stop-marker check
        suffix = sp.decode(idx[0, start_len:].tolist())

        # Stop if model starts emitting a new instruction/response turn
        stop_hit = None
        for marker in stop_markers:
            pos = suffix.find(marker)
            if pos >= 0:
                stop_hit = pos
                break

        if stream:
            shown = suffix[:stop_hit] if stop_hit is not None else suffix
            delta = shown[getattr(generate, "_printed_len", 0):]
            print(delta, end="", flush=True)
            generate._printed_len = len(shown)

        if stop_hit is not None:
            # Truncate suffix at stop marker
            suffix = suffix[:stop_hit].rstrip()
            break

        if nid.item() == eos_id:
            break
    dt = time.time() - t0

    full_ids = idx[0, start_len:].tolist()
    text = suffix if isinstance(suffix, str) else sp.decode(full_ids)
    if stream:
        print()
        generate._printed_len = 0
    return text, len(full_ids), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true",
                    help="Use pretrained base (no instruction tuning)")
    ap.add_argument("--clean", action="store_true",
                    help="Use clean finetune (BPB 4.076) instead of v3 (BPB 3.521)")
    ap.add_argument("--device", default=None,
                    choices=["cpu", "cuda", "mps"], help="Force device")
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--thoughts", type=int, default=3)
    args = ap.parse_args()

    device = pick_device(args.device)
    if device == "cpu":
        note = "honest edge-style reference"
    elif device == "mps":
        note = "Apple GPU — faster but NOT edge-representative"
    elif device == "cuda":
        note = "NVIDIA GPU — dev/server, NOT edge-representative"
    else:
        note = ""
    print(f"[device] {device}  ({note})")

    if args.base:
        ckpt_path = BASE_CKPT
    elif args.clean:
        ckpt_path = FINETUNE_CLEAN_CKPT
    else:
        ckpt_path = FINETUNE_CKPT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    print(f"[tokenizer] {TOKENIZER_PATH.name}")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    V = sp.get_piece_size()

    print(f"[checkpoint] {ckpt_path.name}")
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    cfg = ckpt.get("config", {
        "hdc_dim": 4096, "max_seq_len": 512, "n_blocks": 8,
        "controller_dim": 2560, "n_heads": 4, "dropout": 0.05,
        "max_thoughts": 4, "use_checkpoint": False,
    })
    cfg["use_checkpoint"] = False  # never checkpoint at inference
    model, _ = create_model(V, cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.1f}M params loaded in {time.time()-t0:.1f}s")

    val_loss = ckpt.get("val_loss", None)
    if val_loss is not None:
        print(f"[ckpt stats] val_loss={val_loss:.3f}  BPB={val_loss/0.6931:.3f}")

    # Warmup: first forward is slow on MPS/CUDA (kernel compile)
    print("[warmup] ", end="", flush=True)
    t0 = time.time()
    _ = generate(model, sp, "Hello.", device, max_tokens=5,
                 temperature=args.temperature, top_k=args.top_k,
                 rep_penalty=args.rep_penalty, n_thoughts=args.thoughts,
                 stream=False)
    print(f"{time.time()-t0:.1f}s")

    mode = "BASE (raw text completion)" if args.base else "INSTRUCTION"
    print()
    print("=" * 60)
    print(f"HDC-Brain v14.1 chat — {mode}")
    print("Type your question. Commands: /quit, /reset, /temp N, /len N")
    print("=" * 60)
    print()

    temperature = args.temperature
    max_tokens = args.max_tokens

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            break
        if user == "/reset":
            print("[context is stateless — nothing to reset]")
            continue
        if user.startswith("/temp "):
            try:
                temperature = float(user.split()[1])
                print(f"[temperature={temperature}]")
            except Exception:
                print("[usage: /temp 0.7]")
            continue
        if user.startswith("/len "):
            try:
                max_tokens = int(user.split()[1])
                print(f"[max_tokens={max_tokens}]")
            except Exception:
                print("[usage: /len 200]")
            continue

        if args.base:
            prompt = user
        else:
            prompt = f"### Instruction: {user}\n### Response:"

        print("ai> ", end="", flush=True)
        text, n_tok, dt = generate(
            model, sp, prompt, device,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=args.top_k,
            rep_penalty=args.rep_penalty,
            n_thoughts=args.thoughts,
            stream=True,
        )
        tps = n_tok / dt if dt > 0 else 0.0
        print(f"[{n_tok} tok, {dt:.2f}s, {tps:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
