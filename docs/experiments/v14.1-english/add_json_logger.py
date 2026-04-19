"""
Patch train.py to add JSON experiment logging for arXiv.
Run on server: python3 add_json_logger.py
"""

with open('/workspace/train.py', 'r') as f:
    code = f.read()

# 1. Add imports
if 'import json' not in code:
    code = code.replace('import time', 'import json\nimport time\nfrom datetime import datetime, timezone', 1)

# 2. Add JSON log function after LOG_FILE
json_block = """
# === JSON experiment log (for arXiv reproducibility) ===
EXPERIMENT_LOG = 'experiment.jsonl'

def log_json(event_type, data):
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'event': event_type,
        **data,
    }
    with open(EXPERIMENT_LOG, 'a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\\n')

"""

if 'EXPERIMENT_LOG' not in code:
    code = code.replace(
        "# === Graceful stop ===",
        json_block + "# === Graceful stop ===",
    )

# 3. Add metadata logging at startup
startup = """
    log_json('experiment_start', {
        'model': 'HDC-Brain v14.1',
        'model_config': {k: v for k, v in config.items()},
        'n_params': n_params,
        'training': {
            'batch': BATCH, 'grad_accum': GRAD_ACCUM,
            'tokens_per_step': tokens_per_step, 'seq_len': SEQ_LEN,
            'lr': LR, 'lr_min': LR_MIN, 'warmup': WARMUP,
            'max_iters': MAX_ITERS, 'amp': AMP_DTYPE,
            'clip': CLIP_GRAD, 'thoughts': TRAIN_THOUGHTS,
            'weight_decay': 0.05, 'betas': [0.9, 0.95],
            'codebook_lr_factor': 0.1,
        },
        'data': {
            'train_tokens': int(len(train_data)),
            'val_tokens': int(len(val_data)),
            'tokenizer': TOKENIZER_PATH,
            'vocab_size': vocab_size,
        },
        'hardware': {
            'device': device,
            'gpu': torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
            'vram_gb': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if device == 'cuda' else 0,
        },
        'resume': resume_path, 'start_iter': start_iter,
    })
"""

if "log_json('experiment_start'" not in code:
    anchor = 'print(f"Starting training from iter {start_iter}...")'
    code = code.replace(anchor, anchor + '\n' + startup)

# 4. Add step logging
step_log = """
        log_json('step', {
            'iter': it, 'loss': round(avg_loss, 6),
            'lr': round(lr, 8), 'grad_norm': round(float(grad_norm), 4),
            'ms_per_step': round(ms_per_step, 1), 'tokens': tokens_seen,
        })
"""

if "log_json('step'" not in code:
    code = code.replace(
        '        running_loss = 0.0\n        running_count = 0',
        step_log + '\n        running_loss = 0.0\n        running_count = 0',
    )

# 5. Add eval logging - insert after mass proj norm log line
eval_log = """
        log_json('eval', {
            'iter': it, 'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6), 'bpb': round(bpb, 4),
            'best_bpb': round(best_val / 0.6931, 4),
            'gap': round(gap, 4), 'is_best': is_best,
            'hours': round(elapsed_h, 2),
            'gates': [round(g, 4) for g in gates],
            'mass_norm': round(float(masses.norm()), 4),
            'vram_mb': torch.cuda.max_memory_allocated() // (1024*1024) if device == 'cuda' else 0,
        })
"""

if "log_json('eval'" not in code:
    # Find the second occurrence of log(f"{'='*60}")  in eval block
    # Insert after the mass_proj log line
    anchor = "        log(f\"  Mass proj norm: {masses.norm():.4f}\")"
    code = code.replace(anchor, anchor + '\n' + eval_log)

# 6. Add experiment end logging
end_log = """
log_json('experiment_end', {
    'final_iter': it,
    'best_val': round(best_val, 6),
    'best_bpb': round(best_val / 0.6931, 4),
    'total_hours': round((time.time() - t0) / 3600, 2),
    'reason': 'signal' if stop_training else 'done',
})
"""

if "log_json('experiment_end'" not in code:
    anchor = 'print(f"Saved {LAST_CKPT_PATH}")'
    code = code.replace(anchor, anchor + '\n' + end_log)

with open('/workspace/train.py', 'w') as f:
    f.write(code)

print('OK: JSON logger patched into train.py')
print('Output: /workspace/experiment.jsonl')
