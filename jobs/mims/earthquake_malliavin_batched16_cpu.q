#!/bin/bash
#PBS -N mall_batch16
#PBS -j oe
#PBS -l ncpus=48
#PBS -q hi

set -euo pipefail

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=48

export PYTHONPATH="$HOME/scoremodel/src"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/matplotlib-$USER"
mkdir -p "$MPLCONFIGDIR"

cd "$HOME/scoremodel"
source "$HOME/scoremodel-venv/bin/activate"

OUTDIR="results/earthquake_final_curated_lowt_batched16_mims_cpu/malliavin"
LOGFILE="results/earthquake_final_curated_lowt_batched16_mims_cpu.log"

mkdir -p "$OUTDIR"

echo "========================================" > "$LOGFILE"
echo "MIMS CPU MALLIAVIN BATCHED16 START" >> "$LOGFILE"
echo "DATE=$(date)" >> "$LOGFILE"
echo "HOST=$(hostname)" >> "$LOGFILE"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS" >> "$LOGFILE"
echo "========================================" >> "$LOGFILE"

python -u scripts/experiment_earthquake_teacher_compare_smoke.py \
  --teacher malliavin \
  --teacher-implementation batched \
  --teacher-batch-size 16 \
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
  --covariance-regularization 1e-6 \
  --device cpu \
  --dtype float64 \
  --skip-viz \
  >> "$LOGFILE" 2>&1

echo >> "$LOGFILE"
echo "========================================" >> "$LOGFILE"
echo "MIMS CPU MALLIAVIN BATCHED16 END" >> "$LOGFILE"
echo "DATE=$(date)" >> "$LOGFILE"
echo "========================================" >> "$LOGFILE"
