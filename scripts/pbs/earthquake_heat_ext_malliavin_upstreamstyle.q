#!/bin/sh
#PBS -N earthquake_heat_upstyle
#PBS -j oe
#PBS -l ncpus=48
#PBS -q hi

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=48

export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/matplotlib-$USER"
mkdir -p "$MPLCONFIGDIR"

cd "$HOME/scoremodel"

source "$HOME/scoremodel-venv/bin/activate"

export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"

echo "=== environment ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "=== start training ==="

python scripts/earthquake_malliavin_upstream_style_training.py
echo "=== finished ==="