#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-mukv}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Using existing conda environment: ${ENV_NAME}"
else
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"

python -m pip install --no-cache-dir -U pip "setuptools<82" wheel
python -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
python -m pip install --no-cache-dir -U "git+https://github.com/huggingface/transformers.git@66bc4def9505fa7c7fe4aa7a248c34a026bb552b"
python -m pip install --no-cache-dir ninja packaging
python -m pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation

cd "${PROJECT_ROOT}"
python -m pip install --no-cache-dir -e . --no-build-isolation

cd "${PROJECT_ROOT}/model/longva"
python -m pip install --no-cache-dir -e . --no-build-isolation

python - <<'PY'
import torch
import transformers
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
from longva.model import LlavaQwenForCausalLM

print("MuKV environment check passed")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print("llava-onevision: ok")
print("longva: ok")
PY
