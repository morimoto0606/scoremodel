#!/bin/sh
#PBS -N scoremodel_cpu_check
#PBS -j oe
#PBS -l ncpus=24
#PBS -q mid

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export PYTHONPATH="$HOME/scoremodel/src:$HOME/scoremodel"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/matplotlib-$USER"
mkdir -p "$MPLCONFIGDIR"

cd "$HOME/scoremodel" || exit 1
source "$HOME/scoremodel-venv/bin/activate"

python - <<'PY'
import sys
import torch
import scoremodel_ext

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("threads:", torch.get_num_threads())
print("scoremodel_ext:", scoremodel_ext.__file__)
PY

python scripts/experiment_earthquake_teacher_compare_smoke.py --help
