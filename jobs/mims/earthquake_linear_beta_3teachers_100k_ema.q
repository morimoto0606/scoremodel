#!/bin/bash
#PBS -q hi
#PBS -l select=1:ncpus=48
#PBS -j oe
#PBS -o logs/mims/earthquake_linear_beta_3teachers_100k_ema.log

set -euo pipefail

cd "$HOME/scoremodel"
source "$HOME/scoremodel-venv/bin/activate"

export PYTHONPATH=src:.
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48

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
  --training-unit updates
  --updates 100000
  --warmup-updates 1000
  --lr-scheduler cosine
  --ema-rate 0.999
  --use-ema-for-validation
  --use-ema-for-reverse
  --checkpoint-every-updates 25000
  --device cpu
  --dtype float64
  --skip-viz
)

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

for teacher in heat varadhan malliavin
do
  echo "========================================"
  echo "Starting teacher: ${teacher}"
  echo "Time: $(date)"
  echo "========================================"

  EXTRA_ARGS=()

  if [ "$teacher" = "malliavin" ]; then
    EXTRA_ARGS+=(
      --teacher-implementation batched
      --teacher-batch-size 16
    )
  fi

  python scripts/experiment_earthquake_teacher_compare_smoke.py \
    --teacher "$teacher" \
    --output-dir "results/earthquake_linear_beta_100k_ema_${teacher}"\
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

  echo "Completed teacher: ${teacher}"
  echo "Time: $(date)"
done

echo "All three teachers completed: $(date)"