"""Apply JSON logging patches to train.py on server."""

with open("/workspace/train.py") as f:
    code = f.read()

# 1. Startup metadata (top-level, after print "Starting training")
old1 = 'print(f"Starting training from iter {start_iter}...")\nt0 = time.time()'
new1 = '''print(f"Starting training from iter {start_iter}...")
log_json("experiment_start", {
    "model": "HDC-Brain v14.1", "n_params": n_params,
    "config": {k: v for k, v in config.items()},
    "train": {"batch": BATCH, "accum": GRAD_ACCUM, "tps": tokens_per_step,
              "seq": SEQ_LEN, "lr": LR, "lr_min": LR_MIN, "warmup": WARMUP,
              "max_iters": MAX_ITERS, "amp": AMP_DTYPE, "clip": CLIP_GRAD,
              "thoughts": TRAIN_THOUGHTS, "wd": 0.05, "betas": [0.9, 0.95]},
    "data": {"train_tok": int(len(train_data)), "val_tok": int(len(val_data)),
             "vocab": vocab_size, "tokenizer": TOKENIZER_PATH},
    "hw": {"gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
           "vram": round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if device == "cuda" else 0},
    "resume": resume_path, "start_iter": start_iter,
})
t0 = time.time()'''
assert old1 in code, "PATCH 1 ANCHOR NOT FOUND"
code = code.replace(old1, new1)
print("Patch 1: experiment_start OK")

# 2. Step logging (after running_count reset)
old2 = '        running_loss = 0.0\n        running_count = 0'
new2 = '''        log_json("step", {"iter": it, "loss": round(avg_loss, 6), "lr": round(lr, 8),
            "grad": round(float(grad_norm), 4), "ms": round(ms_per_step, 1), "tok": tokens_seen})
        running_loss = 0.0
        running_count = 0'''
assert old2 in code, "PATCH 2 ANCHOR NOT FOUND"
code = code.replace(old2, new2)
print("Patch 2: step logging OK")

# 3. Eval logging (after mass proj norm line)
old3 = '        log(f"  Mass proj norm: {masses.norm():.4f}")'
new3 = '''        log(f"  Mass proj norm: {masses.norm():.4f}")
        log_json("eval", {"iter": it, "train": round(train_loss, 6), "val": round(val_loss, 6),
            "bpb": round(bpb, 4), "best_bpb": round(best_val / 0.6931, 4), "gap": round(gap, 4),
            "best": is_best, "hours": round(elapsed_h, 2),
            "gates": [round(g, 4) for g in gates], "mass": round(float(masses.norm()), 4),
            "vram_mb": torch.cuda.max_memory_allocated() // (1024*1024) if device == "cuda" else 0})'''
assert old3 in code, "PATCH 3 ANCHOR NOT FOUND"
code = code.replace(old3, new3)
print("Patch 3: eval logging OK")

# 4. End logging
old4 = 'print(f"Saved {LAST_CKPT_PATH}")'
new4 = '''print(f"Saved {LAST_CKPT_PATH}")
log_json("end", {"iter": it, "best_val": round(best_val, 6), "best_bpb": round(best_val / 0.6931, 4),
    "hours": round((time.time() - t0) / 3600, 2), "reason": "signal" if stop_training else "done"})'''
assert old4 in code, "PATCH 4 ANCHOR NOT FOUND"
code = code.replace(old4, new4)
print("Patch 4: end logging OK")

with open("/workspace/train.py", "w") as f:
    f.write(code)

print("\nAll patches applied! Verifying syntax...")

import py_compile
try:
    py_compile.compile("/workspace/train.py", doraise=True)
    print("Syntax OK!")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
