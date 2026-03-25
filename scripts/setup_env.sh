#!/bin/bash
# ============================================================
# Environment Setup Script for RL-LLM-Scheduling Project
# Run this on the GPU server before starting any experiments
# ============================================================

set -e

echo "=== Phase 1: Create Conda Environment ==="
conda create -n llumnix-rl python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate llumnix-rl

echo "=== Phase 2: Install PyTorch with CUDA ==="
# Detect CUDA version and install appropriate PyTorch
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "Detected CUDA version: ${CUDA_VERSION:-not found}"

# Default to CUDA 11.8; change if your server has different CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo "=== Phase 3: Clone and Install Llumnix ==="
git clone https://github.com/llumnix-project/llumnix-ray.git
cd llumnix-ray

# Install with vLLM backend
pip install -e ".[vllm]"

# Verify
python -c "import llumnix; import vllm; import ray; print('Llumnix + vLLM + Ray: OK')"

echo "=== Phase 4: Install RL Libraries ==="
pip install stable-baselines3 gymnasium tensorboard wandb numpy scipy pandas matplotlib

python -c "from stable_baselines3 import PPO; import gymnasium; print('RL libs: OK')"

echo "=== Phase 5: Create Data Directories ==="
cd ..
mkdir -p data models results logs output/{midterm,final,results,models,code}

echo "=== Phase 6: Download Datasets ==="
# ShareGPT
echo "Downloading ShareGPT dataset..."
wget -q https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O ./data/sharegpt.json && echo "ShareGPT: OK" || echo "ShareGPT: FAILED (download manually)"

echo "=== Phase 7: Download Model ==="
echo "NOTE: LLaMA-2-7B requires HuggingFace access. Run manually:"
echo "  pip install huggingface_hub"
echo "  huggingface-cli login"
echo "  huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./models/llama-7b"
echo ""
echo "If you don't have access, alternatives:"
echo "  - Use a smaller open model (e.g., facebook/opt-6.7b)"
echo "  - Use any 7B model compatible with vLLM"

echo "=== Phase 8: Quick Sanity Test ==="
echo "After downloading the model, run this to verify:"
echo "  python -m llumnix.entrypoints.vllm.api_server \\"
echo "    --initial-instances 1 \\"
echo "    --model ./models/llama-7b \\"
echo "    --host 0.0.0.0 --port 8000 \\"
echo "    --launch-ray-cluster"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "1. Download LLaMA-7B model (see above)"
echo "2. Run sanity test"
echo "3. Start implementing RL components (see docs/IMPLEMENTATION_GUIDE.md)"
