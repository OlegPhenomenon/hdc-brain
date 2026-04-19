#!/bin/bash
# HDC-Brain v14.1 — Server Setup Script
# Run on vast.ai / Lambda / RunPod after SSH connection
#
# Usage:
#   chmod +x server_setup.sh
#   ./server_setup.sh

set -e

echo "=========================================="
echo "  HDC-Brain v14.1 Server Setup"
echo "=========================================="

# 1. System info
echo ""
echo "--- System Info ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU found"
python3 --version
echo "CPU cores: $(nproc)"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"

# 2. Install dependencies
echo ""
echo "--- Installing dependencies ---"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install sentencepiece datasets numpy tqdm

# 3. Verify GPU
echo ""
echo "--- GPU Check ---"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'BF16: {torch.cuda.is_bf16_supported()}')
"

# 4. Prepare data (tokenizer + tokenize)
echo ""
echo "--- Data Preparation ---"
if [ ! -f "bpe_en_32k.model" ]; then
    echo "Training tokenizer..."
    python3 prepare_data.py --step tokenizer --sample-size 500000
else
    echo "Tokenizer already exists"
fi

if [ ! -f "train.bin" ]; then
    echo "Tokenizing data (target: 2B tokens, ~4GB file)..."
    python3 prepare_data.py --step tokenize --num-tokens 2000000000
    # For budget training: 500M tokens is enough for 1-day run
    # python3 prepare_data.py --step tokenize --num-tokens 500000000
else
    echo "Training data already exists"
    ls -lh train.bin val.bin
fi

# 5. Test model
echo ""
echo "--- Model Test ---"
python3 hdc_brain_v14_1.py

# 6. Start training
echo ""
echo "=========================================="
echo "  Ready to train!"
echo "  Run: nohup python3 train.py > train.log 2>&1 &"
echo "  Monitor: tail -f train.log"
echo "=========================================="
