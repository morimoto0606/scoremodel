#!/bin/bash
#PBS -N eq_linear_beta
#PBS -j oe
#PBS -l ncpus=48
#PBS -q hi

set -euo pipefail

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=48
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="$HOME/scoremodel/src:$HOME/scoremodel"

cd "$HOME/scoremodel"
source "$HOME/scoremodel-venv/bin/activate"

ROOT="results/earthquake_linear_beta_mims"
LOG="$ROOT/run_all.log"
mkdir -p "$ROOT"

COMMON_ARGS=(
  --data-path upstream/riemannian-score-sde/data/quakes_all.csv
  --train-size 4096
  --validation-size 1024
  --n-steps 32
  --minimum-time 0.001
  --maximum-time 1.0
  --time-sampling uniform
  --epochs 3000
  --batch-size 512
  --learning-rate 2e-4
  --weight-decay 0
  --hidden 1024
  --n-blocks 6
  --num-frequencies 16
  --reverse-steps 256
  --n-generated-samples 4096
  --seed 0
  --split-seed 0
  --reverse-seed 0
  --covariance-regularization 1e-6
  --beta-schedule linear
  --beta-0 0.001
  --beta-f 5
  --beta-t0 0
  --beta-tf 1
  --device cpu
  --dtype float64
  --skip-viz
)

{
  echo "========================================"
  echo "EARTHQUAKE LINEAR BETA 3-TEACHER START"
  echo "DATE=$(date)"
  echo "HOST=$(hostname)"
  echo "========================================"
} > "$LOG"

for TEACHER in heat varadhan malliavin
do
  OUTDIR="$ROOT/$TEACHER"
  mkdir -p "$OUTDIR"

  echo "===== $TEACHER START: $(date) =====" | tee -a "$LOG"

  EXTRA_ARGS=()
  if [ "$TEACHER" = "malliavin" ]; then
    EXTRA_ARGS+=(
      --teacher-implementation batched
      --teacher-batch-size 16
    )
  fi

  python -u scripts/experiment_earthquake_teacher_compare_smoke.py \
    --teacher "$TEACHER" \
    --output-dir "$OUTDIR" \
    "${EXTRA_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    >> "$LOG" 2>&1

  echo "===== $TEACHER END: $(date) =====" | tee -a "$LOG"
done

echo "ALL DONE: $(date)" | tee -a "$LOG"
