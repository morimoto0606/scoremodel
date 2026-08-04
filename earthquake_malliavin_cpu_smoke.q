#!/bin/sh
#PBS -N mall_cpu_smoke
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

python scripts/experiment_earthquake_teacher_compare_smoke.py \
  --teacher malliavin \
  --output-dir results/earthquake_malliavin_cpu_smoke \
  --data-path upstream/riemannian-score-sde/data/quakes_all.csv \
  --train-size 32 \
  --validation-size 8 \
  --n-steps 4 \
  --minimum-time 0.005 \
  --maximum-time 0.3 \
  --time-sampling curated-lowt \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --weight-decay 0 \
  --hidden 64 \
  --n-blocks 2 \
  --num-frequencies 8 \
  --reverse-steps 16 \
  --n-generated-samples 32 \
  --seed 0 \
  --split-seed 0 \
  --reverse-seed 0 \
  --device cpu \
  --dtype float64 \
  --skip-viz
