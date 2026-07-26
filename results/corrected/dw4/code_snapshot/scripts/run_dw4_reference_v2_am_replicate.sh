#!/usr/bin/env bash
set -Eeuo pipefail

# Fixed-FM AM replicate for the corrected DW4 reference.
#
# Usage:
#   AM_SEED=1 CUDA_VISIBLE_DEVICES=2 \
#     bash scripts/run_dw4_reference_v2_am_replicate.sh
#
# The only training variable relative to the other arms is AM_SEED. Evaluation
# and metric subsampling always use seed 0.

ROOT=${ADTM_ROOT}
PY=${CONDA_ROOT}/envs/ab/bin/python
REFERENCE="${ROOT}/data/dw4_reference_v2"
PARENT="${ROOT}/outputs/dw4/dw4_reference_v2_egnn128x5_anchor_0p8_1p2_target_1p0_seed2_run1"
EXPECTED_FM_SHA=4746a1df5057301da085d20155b2e740d3d08126d1ea705863ed90cf9a61fbde
EXPECTED_REFERENCE_MANIFEST_SHA=3ae2b927765c37354e591808f90b86f4eb3078a7f9fac8ded2618157b025c932

AM_SEED="${AM_SEED:?Set AM_SEED to 1 or 3.}"
case "${AM_SEED}" in
  1|3) ;;
  *)
    echo "AM_SEED must be 1 or 3 for the pre-registered replicate matrix." >&2
    exit 2
    ;;
esac

RUN="${ROOT}/outputs/dw4/dw4_reference_v2_egnn128x5_anchor_0p8_1p2_target_1p0_seed${AM_SEED}_run1"
LOG="${ROOT}/logs/dw4_reference_v2_egnn128x5_anchor_0p8_1p2_target_1p0_seed${AM_SEED}_run1.log"
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
    printf 'shared_fm_seed=0\n'
    printf 'am_seed=%s\n' "${AM_SEED}"
    printf 'evaluation_seed=0\n'
    printf 'parent_fm_sha256=%s\n' "${EXPECTED_FM_SHA}"
    printf 'reference_manifest_sha256=%s\n' "${EXPECTED_REFERENCE_MANIFEST_SHA}"
  } > "${STATUS}"
}

on_error() {
  local rc=$?
  {
    printf 'phase=failed\n'
    printf 'exit_code=%s\n' "${rc}"
    printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'am_seed=%s\n' "${AM_SEED}"
  } > "${STATUS}"
  exit "${rc}"
}
trap on_error ERR

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
export PYTHONUNBUFFERED=1

write_status validate_bindings
test "$(sha256sum "${PARENT}/fm_params.pkl" | awk '{print $1}')" = "${EXPECTED_FM_SHA}"
test "$(sha256sum "${REFERENCE}/manifest.json" | awk '{print $1}')" = "${EXPECTED_REFERENCE_MANIFEST_SHA}"
(
  cd "${REFERENCE}"
  sha256sum -c SHA256SUMS
)

cp "${PARENT}/fm_params.pkl" "${RUN}/fm_params.pkl"
cp "${PARENT}/fm_loss_history.npy" "${RUN}/fm_loss_history.npy"
cp "${PARENT}/config.yaml" "${RUN}/parent_fm_config.yaml"
sha256sum "${RUN}/fm_params.pkl" > "${RUN}/fm_params.sha256"

cp "${REFERENCE}/manifest.json" "${RUN}/reference_manifest.json"
cp "${REFERENCE}/convergence_audit.json" "${RUN}/reference_convergence_audit.json"
cp "${REFERENCE}/importance_audit.json" "${RUN}/reference_importance_audit.json"
cp "${PARENT}/reference_exact_identity_audit.json" "${RUN}/reference_exact_identity_audit.json"
cp "${ROOT}/scripts/run_dw4_reference_v2_am_replicate.sh" "${RUN}/exact_run_script.sh"

mkdir -p "${RUN}/fm_beta_sweep"
cp "${PARENT}/fm_beta_sweep/fm_samples_beta_1.00.npy" "${RUN}/fm_beta_sweep/"
cp "${PARENT}/fm_beta_sweep/fm_beta_sweep_metrics.json" "${RUN}/fm_beta_sweep/"

{
  printf '{\n'
  printf '  "schema": "adtm.dw4_reference_v2.am_replicate.v1",\n'
  printf '  "shared_fm_seed": 0,\n'
  printf '  "am_seed": %s,\n' "${AM_SEED}"
  printf '  "evaluation_seed": 0,\n'
  printf '  "evaluation_initial_key": 7000,\n'
  printf '  "evaluation_chunk_size": 5000,\n'
  printf '  "evaluation_rows": 100000,\n'
  printf '  "parent_fm_sha256": "%s",\n' "${EXPECTED_FM_SHA}"
  printf '  "reference_manifest_sha256": "%s"\n' "${EXPECTED_REFERENCE_MANIFEST_SHA}"
  printf '}\n'
} > "${RUN}/replicate_binding.json"

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
  --seed "${AM_SEED}" \
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

write_status score_am
"${PY}" main.py score-samples \
  --problem dw4 \
  --model egnn \
  --data-name dw4_reference_v2/train \
  --reference-data-name dw4_reference_v2/eval \
  --run-dir "${RUN}" \
  --seed 0 \
  --eval-betas 1.0 \
  --score-kinds am \
  --energy-w2-samples 20000 \
  --geometric-w2-samples 2000 \
  --geometric-w2-chunk-size 16 \
  --score-ref-baseline-repeats 5

SCORE_METRICS="${RUN}/score_samples_metrics_dw4_reference_v2_eval_am_betas_1p00_geo2000_ew20000.json"
test -f "${SCORE_METRICS}"

write_status hash_results
sha256sum \
  "${RUN}/fm_params.pkl" \
  "${RUN}/am_params.pkl" \
  "${RUN}/fm_loss_history.npy" \
  "${RUN}/am_loss_history.npy" \
  "${RUN}/fm_beta_sweep/fm_samples_beta_1.00.npy" \
  "${RUN}/am_beta_sweep/am_samples_beta_1.00.npy" \
  "${RUN}/fm_beta_sweep/fm_beta_sweep_metrics.json" \
  "${RUN}/am_beta_sweep/am_beta_sweep_metrics.json" \
  "${SCORE_METRICS}" \
  "${RUN}/parent_fm_config.yaml" \
  "${RUN}/replicate_binding.json" \
  "${RUN}/exact_run_script.sh" \
  "${RUN}/reference_manifest.json" \
  "${RUN}/reference_convergence_audit.json" \
  "${RUN}/reference_importance_audit.json" \
  "${RUN}/reference_exact_identity_audit.json" \
  > "${RUN}/result_SHA256SUMS"

write_status complete
echo "DW4 corrected-reference AM seed ${AM_SEED} replicate complete: ${RUN}"
