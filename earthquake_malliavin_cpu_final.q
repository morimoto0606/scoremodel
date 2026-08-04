#!/bin/sh
#PBS -N mall_cpu_final
#PBS -j oe
#PBS -l ncpus=48
#PBS -q hi

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=48

export PYTHONPATH="$HOME/scoremodel/src:$HOME/scoremodel"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/matplotlib-$USER"
mkdir -p "$MPLCONFIGDIR"

cd "$HOME/scoremodel" || exit 1
source "$HOME/scoremodel-venv/bin/activate"

ROOT="results/earthquake_final_curated_lowt_mims_cpu"
OUTDIR="$ROOT/malliavin"
mkdir -p "$OUTDIR"

python scripts/experiment_earthquake_teacher_compare_smoke.py \
  --teacher malliavin \
  --output-dir "$OUTDIR" \
  --data-path upstream/riemannian-score-sde/data/quakes_all.csv \
  --train-size 4096 \
  --validation-size 1024 \
  --n-steps 32 \
  --minimum-time 0.005 \
  --maximum-time 0.3 \
  --time-sampling curated-lowt \
  --epochs 3000 \
  --batch-size 512 \
  --learning-rate 2e-4 \
  --weight-decay 0 \
  --hidden 1024 \
  --n-blocks 6 \
  --num-frequencies 16 \
  --reverse-steps 256 \
  --n-generated-samples 4096 \
  --seed 0 \
  --split-seed 0 \
  --reverse-seed 0 \
  --device cpu \
  --dtype float64 \
  --skip-viz
