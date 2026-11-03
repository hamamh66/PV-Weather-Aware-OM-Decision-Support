#!/bin/bash
# creates the Python environment for pv_finetune.py (Digital Research Alliance of Canada wheels where available)
set -e
module load python/3.11 >/dev/null 2>&1 || true
cd "$(dirname "$0")"
unset PIP_PREFIX
if [ ! -d venv ]; then virtualenv --no-download venv; fi
source venv/bin/activate
pip install --no-index torch torchvision timm transformers scikit-learn statsmodels scipy matplotlib pillow 2>&1 | tail -2
pip install kagglehub 2>&1 | tail -2
python -c "import torch,timm,transformers,kagglehub;print('ENV OK', torch.__version__, torch.cuda.is_available(), timm.__version__, transformers.__version__)"
