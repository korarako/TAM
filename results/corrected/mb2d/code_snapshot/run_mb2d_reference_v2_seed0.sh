#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ADTM_ROOT}
PY=${CONDA_ROOT}/envs/ab/bin/python
RUN="${ROOT}/outputs/mb2d/mb2d_reference_v2_anchor_1.00_target_1.20_seed0_run1"
LOG="${ROOT}/logs/mb2d_reference_v2_anchor_1.00_target_1.20_seed0_run1.log"
STATUS="${RUN}/status.txt"
REFERENCE="${ROOT}/data/mb2d_reference_v2"

if [[ -e "${RUN}" ]]; then
  echo "Refusing to overwrite existing run directory: ${RUN}" >&2
  exit 2
fi

mkdir -p "${RUN}" "${ROOT}/logs"
exec > >(tee -a "${LOG}") 2>&1

write_status() {
  local phase="$1"
  {
    printf 'phase=%s\n' "${phase}"
    printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host=%s\n' "$(hostname)"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  } > "${STATUS}"
}

on_error() {
  local rc=$?
  {
    printf 'phase=failed\n'
    printf 'exit_code=%s\n' "${rc}"
    printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${STATUS}"
  exit "${rc}"
}
trap on_error ERR

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

cp "${REFERENCE}/manifest.json" "${RUN}/reference_manifest.json"
cp "${REFERENCE}/SHA256SUMS" "${RUN}/reference_SHA256SUMS"
cp "${ROOT}/scripts/run_mb2d_reference_v2_seed0.sh" "${RUN}/exact_run_script.sh"

write_status reference_validation
"${PY}" scripts/generate_mb2d_reference_v2.py validate \
  --output data/mb2d_reference_v2 \
  > "${RUN}/reference_validation.json"

write_status train_fm
"${PY}" main.py train-fm \
  --problem mb2d \
  --model mlp \
  --data-name mb2d_reference_v2/train \
  --reference-data-name mb2d_reference_v2/eval \
  --beta-grid 0.25 0.50 0.75 1.00 1.50 \
  --beta0 1.00 \
  --beta1 1.20 \
  --run-dir "${RUN}" \
  --seed 0 \
  --n-per-beta 100000 \
  --n-eval 100000 \
  --fm-steps 30000 \
  --batch-size 512 \
  --fm-lr 1.0e-3 \
  --fm-lr-min 1.0e-5 \
  --fm-samples-per-beta 0 \
  --fm-eval-every 0 \
  --hidden-dim 512 \
  --n-layers 3 \
  --t-embed-dim 16 \
  --beta-embed-dim 8 \
  --prior-scale 1.0 \
  --grad-clip 1.0 \
  --ode-steps 300 \
  --ode-method euler \
  --log-every 100
sha256sum "${RUN}/fm_params.pkl" > "${RUN}/fm_params.sha256"

write_status train_am
"${PY}" main.py train-am \
  --problem mb2d \
  --model mlp \
  --data-name mb2d_reference_v2/train \
  --reference-data-name mb2d_reference_v2/eval \
  --beta-grid 0.25 0.50 0.75 1.00 1.50 \
  --beta0 1.00 \
  --beta1 1.20 \
  --run-dir "${RUN}" \
  --seed 0 \
  --n-eval 100000 \
  --am-steps 10000 \
  --am-batch-size 512 \
  --am-lr 2.0e-5 \
  --am-lr-min 2.0e-7 \
  --K 40 \
  --am-num-loss-steps 20 \
  --am-keep-last-steps 10 \
  --max-sigma 50.0 \
  --energy-grad-scale 1.0 \
  --prior-scale 1.0 \
  --grad-clip 1.0 \
  --adam-b1 0.9 \
  --weight-decay 0.0 \
  --hidden-dim 512 \
  --n-layers 3 \
  --t-embed-dim 16 \
  --beta-embed-dim 8 \
  --log-every 100
sha256sum "${RUN}/am_params.pkl" > "${RUN}/am_params.sha256"

write_status sample_eval
"${PY}" main.py sample-eval \
  --problem mb2d \
  --model mlp \
  --data-name mb2d_reference_v2/train \
  --reference-data-name mb2d_reference_v2/eval \
  --beta-grid 0.25 0.50 0.75 1.00 1.50 \
  --beta0 1.00 \
  --beta1 1.20 \
  --run-dir "${RUN}" \
  --seed 0 \
  --n-eval 100000 \
  --ode-steps 300 \
  --ode-method euler \
  --prior-scale 1.0 \
  --hidden-dim 512 \
  --n-layers 3 \
  --t-embed-dim 16 \
  --beta-embed-dim 8 \
  --energy-w2-samples 2000

write_status analytic_rescore
"${PY}" scripts/generate_mb2d_reference_v2.py validate \
  --output data/mb2d_reference_v2 \
  --compare new_fm="${RUN}/fm_samples_beta_{beta}.npy" \
  --compare new_am="${RUN}/am_samples_beta_{beta}.npy" \
  > "${RUN}/analytic_grid_comparison.json"

"${PY}" main.py score-samples \
  --problem mb2d \
  --model mlp \
  --data-name mb2d_reference_v2/train \
  --reference-data-name mb2d_reference_v2/eval \
  --run-dir "${RUN}" \
  --seed 0 \
  --eval-betas 1.00 1.20 \
  --score-kinds fm am \
  --energy-w2-samples 20000 \
  --geometric-w2-samples 2000 \
  --score-ref-baseline-repeats 5

sha256sum \
  "${RUN}/fm_samples_beta_1.00.npy" \
  "${RUN}/fm_samples_beta_1.20.npy" \
  "${RUN}/am_samples_beta_1.20.npy" \
  "${RUN}/metrics.json" \
  "${RUN}/analytic_grid_comparison.json" \
  > "${RUN}/result_SHA256SUMS"

write_status complete
echo "MB2D reference-v2 seed-0 pilot complete: ${RUN}"
