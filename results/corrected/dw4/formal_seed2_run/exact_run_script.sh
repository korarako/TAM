#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ADTM_ROOT}
PY=${CONDA_ROOT}/envs/ab/bin/python
REFERENCE="${ROOT}/data/dw4_reference_v2"
RUN="${ROOT}/outputs/dw4/dw4_reference_v2_egnn128x5_anchor_0p8_1p2_target_1p0_seed2_run1"
LOG="${ROOT}/logs/dw4_reference_v2_egnn128x5_anchor_0p8_1p2_target_1p0_seed2_run1.log"
STATUS="${RUN}/status.txt"

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
export JAX_ENABLE_X64=true
export PYTHONUNBUFFERED=1

cp "${REFERENCE}/manifest.json" "${RUN}/reference_manifest.json"
cp "${REFERENCE}/convergence_audit.json" "${RUN}/reference_convergence_audit.json"
cp "${REFERENCE}/importance_audit.json" "${RUN}/reference_importance_audit.json"
cp "${REFERENCE}/SHA256SUMS" "${RUN}/reference_SHA256SUMS"
cp "${ROOT}/scripts/run_dw4_reference_v2_seed2.sh" "${RUN}/exact_run_script.sh"

write_status reference_validation
"${PY}" scripts/generate_dw4_reference_v2.py validate \
  --output data/dw4_reference_v2 \
  > "${RUN}/reference_validation.json"

write_status train_fm
"${PY}" main.py train-fm \
  --problem dw4 \
  --model egnn \
  --data-name dw4_reference_v2/train \
  --reference-data-name dw4_reference_v2/eval \
  --beta-grid 0.8 1.2 \
  --beta0 0.8 \
  --beta1 1.0 \
  --run-dir "${RUN}" \
  --seed 0 \
  --n-per-beta 100000 \
  --fm-steps 100000 \
  --batch-size 512 \
  --fm-lr 3.0e-4 \
  --fm-lr-min 3.0e-4 \
  --fm-samples-per-beta 0 \
  --fm-eval-every 0 \
  --hidden-dim 128 \
  --n-layers 5 \
  --t-embed-dim 16 \
  --beta-embed-dim 8 \
  --prior-scale 1.0 \
  --grad-clip 1.0 \
  --ode-steps 150 \
  --ode-method euler \
  --log-every 100
sha256sum "${RUN}/fm_params.pkl" > "${RUN}/fm_params.sha256"

write_status train_am
"${PY}" main.py train-am \
  --problem dw4 \
  --model egnn \
  --data-name dw4_reference_v2/train \
  --reference-data-name dw4_reference_v2/eval \
  --beta-grid 0.8 1.2 \
  --beta0 0.8 \
  --beta1 1.0 \
  --run-dir "${RUN}" \
  --seed 2 \
  --am-steps 1000 \
  --am-batch-size 512 \
  --am-lr 5.0e-7 \
  --am-lr-min 5.0e-9 \
  --K 40 \
  --am-num-loss-steps 20 \
  --am-keep-last-steps 10 \
  --max-sigma 50.0 \
  --energy-grad-scale 1.0 \
  --prior-scale 1.0 \
  --grad-clip 1.0 \
  --adam-b1 0.9 \
  --weight-decay 0.0 \
  --hidden-dim 128 \
  --n-layers 5 \
  --t-embed-dim 16 \
  --beta-embed-dim 8 \
  --log-every 100
sha256sum "${RUN}/am_params.pkl" > "${RUN}/am_params.sha256"

# Use the same evaluation seed and hence the same initial Gaussian samples for
# both controllers.  The chunked entry points avoid a 100k-sample device spike.
write_status eval_fm_100k
"${PY}" main.py eval-fm-betas \
  --problem dw4 \
  --model egnn \
  --data-name dw4_reference_v2/train \
  --reference-data-name dw4_reference_v2/eval \
  --beta-grid 0.8 1.2 \
  --eval-betas 1.0 \
  --beta0 0.8 \
  --beta1 1.0 \
  --run-dir "${RUN}" \
  --seed 0 \
  --n-eval 100000 \
  --eval-chunk-size 5000 \
  --ode-steps 150 \
  --ode-method euler \
  --energy-w2-samples 20000 \
  --geometric-w2-samples 2000 \
  --geometric-w2-chunk-size 16 \
  --prior-scale 1.0 \
  --hidden-dim 128 \
  --n-layers 5 \
  --t-embed-dim 16 \
  --beta-embed-dim 8

write_status eval_am_100k
"${PY}" main.py eval-am-betas \
  --problem dw4 \
  --model egnn \
  --data-name dw4_reference_v2/train \
  --reference-data-name dw4_reference_v2/eval \
  --beta-grid 0.8 1.2 \
  --eval-betas 1.0 \
  --beta0 0.8 \
  --beta1 1.0 \
  --run-dir "${RUN}" \
  --seed 0 \
  --n-eval 100000 \
  --eval-chunk-size 5000 \
  --ode-steps 150 \
  --ode-method euler \
  --energy-w2-samples 20000 \
  --geometric-w2-samples 2000 \
  --geometric-w2-chunk-size 16 \
  --prior-scale 1.0 \
  --hidden-dim 128 \
  --n-layers 5 \
  --t-embed-dim 16 \
  --beta-embed-dim 8

write_status score_samples
"${PY}" main.py score-samples \
  --problem dw4 \
  --model egnn \
  --data-name dw4_reference_v2/train \
  --reference-data-name dw4_reference_v2/eval \
  --run-dir "${RUN}" \
  --seed 0 \
  --eval-betas 1.0 \
  --score-kinds fm am \
  --energy-w2-samples 20000 \
  --geometric-w2-samples 2000 \
  --geometric-w2-chunk-size 16 \
  --score-ref-baseline-repeats 5

sha256sum \
  "${RUN}/fm_params.pkl" \
  "${RUN}/am_params.pkl" \
  "${RUN}/fm_beta_sweep/fm_samples_beta_1.00.npy" \
  "${RUN}/am_beta_sweep/am_samples_beta_1.00.npy" \
  "${RUN}/fm_beta_sweep/fm_beta_sweep_metrics.json" \
  "${RUN}/am_beta_sweep/am_beta_sweep_metrics.json" \
  > "${RUN}/result_SHA256SUMS"

write_status complete
echo "DW4 reference-v2 seed-2 pipeline complete: ${RUN}"
