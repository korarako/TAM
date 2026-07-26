#!/usr/bin/env bash
set -Eeuo pipefail

# Fixed-FM AM replicate for the validated LJ13 reference-v2 bundle.
#
# The replicate changes only AM_SEED relative to the primary AM arm.  All
# replicates reuse the bit-identical FM seed-0 checkpoint, corrected reference
# bundle, evaluation split, evaluation seed, ODE settings, and metric row
# subsampling.
#
# Usage (after the primary seed-2 run is complete):
#   AM_SEED=1 \
#   EXPECTED_FM_SHA=<sha256-of-primary-fm_params.pkl> \
#   CUDA_VISIBLE_DEVICES=0 \
#     bash scripts/run_lj13_reference_v2_am_replicate.sh
#
# Optional:
#   PRIMARY_RUN=/absolute/path/to/primary
#   RUN_TAG=custom_output_tag

ROOT=${ADTM_ROOT}
PY=${CONDA_ROOT}/envs/ab/bin/python
REFERENCE="${ROOT}/data/lj13_reference_v2"
EXPECTED_REFERENCE_MANIFEST_SHA=5487261ba70d2e4c3c10189432dfa31a67c6f32ba3575fe3ec1c388a7039fdb1
PRIMARY_RUN="${PRIMARY_RUN:-${ROOT}/outputs/lj13/lj13_reference_v2_painn128x5_anchor_0p8_1p2_target_1p0_seed2_run1}"

AM_SEED="${AM_SEED:?Set AM_SEED to 1 or 3.}"
case "${AM_SEED}" in
  1|3) ;;
  *)
    echo "AM_SEED must be 1 or 3 for the pre-registered replicate matrix." >&2
    exit 2
    ;;
esac

EXPECTED_FM_SHA="${EXPECTED_FM_SHA:?Set EXPECTED_FM_SHA to the independently recorded primary fm_params.pkl SHA256.}"
if [[ ! "${EXPECTED_FM_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_FM_SHA must be a lowercase 64-character SHA256 digest." >&2
  exit 2
fi

TAG="${RUN_TAG:-lj13_reference_v2_painn128x5_anchor_0p8_1p2_target_1p0_seed${AM_SEED}_run1}"
RUN="${ROOT}/outputs/lj13/${TAG}"
LOG="${ROOT}/logs/${TAG}.log"
STATUS="${RUN}/status.txt"
STRICT_METRICS="${RUN}/score_lj13_reference_v2.json"

if [[ -e "${RUN}" ]]; then
  echo "Refusing to overwrite existing run directory: ${RUN}" >&2
  exit 2
fi

mkdir -p "${RUN}" "${ROOT}/logs"
exec > >(tee -a "${LOG}") 2>&1

write_status() {
  {
    printf 'phase=%s\n' "$1"
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
    printf 'host=%s\n' "$(hostname)"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'shared_fm_seed=0\n'
    printf 'am_seed=%s\n' "${AM_SEED}"
  } > "${STATUS}"
  exit "${rc}"
}
trap on_error ERR

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export JAX_ENABLE_X64=false
export PYTHONUNBUFFERED=1

write_status validate_bindings
test -f "${PRIMARY_RUN}/status.txt"
grep -qx 'phase=complete' "${PRIMARY_RUN}/status.txt"
test -f "${PRIMARY_RUN}/fm_params.pkl"
test -f "${PRIMARY_RUN}/fm_params.sha256"
test -f "${PRIMARY_RUN}/experiment_binding.json"

ACTUAL_FM_SHA="$(sha256sum "${PRIMARY_RUN}/fm_params.pkl" | awk '{print $1}')"
RECORDED_FM_SHA="$(awk 'NR == 1 {print $1}' "${PRIMARY_RUN}/fm_params.sha256")"
test "${ACTUAL_FM_SHA}" = "${EXPECTED_FM_SHA}"
test "${RECORDED_FM_SHA}" = "${EXPECTED_FM_SHA}"
test "$(sha256sum "${REFERENCE}/manifest.json" | awk '{print $1}')" = \
  "${EXPECTED_REFERENCE_MANIFEST_SHA}"
test "$(readlink -f "${REFERENCE}/train")" != "$(readlink -f "${REFERENCE}/eval")"
(
  cd "${REFERENCE}"
  sha256sum -c SHA256SUMS
)

"${PY}" - "${PRIMARY_RUN}/experiment_binding.json" "${REFERENCE}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

binding = json.loads(Path(sys.argv[1]).read_text())
manifest = json.loads(Path(sys.argv[2]).read_text())
assert binding["schema"] == "adtm.lj13_reference_v2.primary_run.v1"
assert binding["fm_seed"] == 0
assert binding["am_seed"] == 2
assert binding["evaluation_seed"] == 0
assert binding["fm_anchor_betas"] == [0.8, 1.2]
assert binding["am_pair"] == [0.8, 1.0]
assert binding["training_split"] == "train"
assert binding["evaluation_split"] == "eval"
assert binding["legacy_reference_used"] is False
assert manifest["schema"] == "adtm.lj13_reference_v2.bundle.v1"
assert manifest["status"] == "validated_equilibrium_reference"
assert manifest["target"]["id"] == "lj13_bms_eq234_lj1_confinement1_comfree_v1"
assert manifest["cross_split_status"] == "pass"
assert manifest["legacy_reference_used"] is False
assert manifest["legacy_unadjusted_langevin_used"] is False
for split in ("train", "eval"):
    assert manifest["splits"][split]["audit_status"] == "pass"
    assert manifest["splits"][split]["publishable"] is True
PY

cp "${PRIMARY_RUN}/fm_params.pkl" "${RUN}/fm_params.pkl"
cp "${PRIMARY_RUN}/fm_loss_history.npy" "${RUN}/fm_loss_history.npy"
cp "${PRIMARY_RUN}/config.yaml" "${RUN}/parent_fm_config.yaml"
cp "${PRIMARY_RUN}/experiment_binding.json" "${RUN}/parent_experiment_binding.json"
sha256sum "${RUN}/fm_params.pkl" > "${RUN}/fm_params.sha256"

cp "${REFERENCE}/manifest.json" "${RUN}/reference_manifest.json"
cp "${REFERENCE}/SHA256SUMS" "${RUN}/reference_SHA256SUMS"
cp "${REFERENCE}/diagnostics/train_audit.json" "${RUN}/reference_train_audit.json"
cp "${REFERENCE}/diagnostics/eval_audit.json" "${RUN}/reference_eval_audit.json"
cp "${REFERENCE}/diagnostics/train_vs_eval.json" "${RUN}/reference_train_vs_eval.json"
cp "${ROOT}/scripts/run_lj13_reference_v2_am_replicate.sh" "${RUN}/exact_run_script.sh"
cp "${ROOT}/scripts/score_lj13_reference_v2.py" "${RUN}/exact_score_script.py"

mkdir -p "${RUN}/fm_beta_sweep"
cp "${PRIMARY_RUN}/fm_beta_sweep/fm_samples_beta_1.00.npy" "${RUN}/fm_beta_sweep/"
cp "${PRIMARY_RUN}/fm_beta_sweep/fm_beta_sweep_metrics.json" "${RUN}/fm_beta_sweep/"

cat > "${RUN}/replicate_binding.json" <<EOF
{
  "schema": "adtm.lj13_reference_v2.am_replicate.v1",
  "shared_fm_seed": 0,
  "am_seed": ${AM_SEED},
  "evaluation_seed": 0,
  "evaluation_rows": 100000,
  "evaluation_chunk_size": 1000,
  "ode_steps": 150,
  "ode_method": "euler",
  "energy_metric_rows": 2000,
  "geometric_metric_rows": 2000,
  "benchmark_protocol": "historically comparable LJ13 2k protocol for energy, pair distance, radius of gyration, minimum pair distance, and geometry",
  "reference_floor_repeats": 5,
  "parent_fm_sha256": "${EXPECTED_FM_SHA}",
  "reference_manifest_sha256": "${EXPECTED_REFERENCE_MANIFEST_SHA}",
  "claim_scope": "AM-seed robustness conditional on one shared FM seed-0 checkpoint"
}
EOF

COMMON=(
  --problem lj13
  --model painn
  --painn-backend jax
  --painn-atom-identity none
  --data-name lj13_reference_v2/train
  --reference-data-name lj13_reference_v2/eval
  --beta-grid 0.8 1.2
  --beta0 0.8
  --beta1 1.0
  --run-dir "${RUN}"
  --prior-scale 1.0
  --hidden-dim 128
  --n-layers 5
  --t-embed-dim 16
  --beta-embed-dim 8
)

write_status train_am
"${PY}" main.py train-am \
  "${COMMON[@]}" \
  --seed "${AM_SEED}" \
  --am-steps 1000 \
  --am-batch-size 64 \
  --am-lr 5.0e-7 \
  --am-lr-min 5.0e-9 \
  --K 40 \
  --am-num-loss-steps 20 \
  --am-keep-last-steps 10 \
  --max-sigma 50.0 \
  --energy-grad-scale 1.0 \
  --grad-clip 0.0 \
  --adam-b1 0.9 \
  --weight-decay 0.0 \
  --save-am-checkpoints 250 500 750 1000 \
  --log-every 100
sha256sum "${RUN}/am_params.pkl" > "${RUN}/am_params.sha256"

write_status eval_am_100k
"${PY}" main.py eval-am-betas \
  "${COMMON[@]}" \
  --seed 0 \
  --eval-betas 1.0 \
  --n-eval 100000 \
  --eval-chunk-size 1000 \
  --ode-steps 150 \
  --ode-method euler \
  --energy-w2-samples 2000 \
  --geometric-w2-samples 2000 \
  --geometric-w2-chunk-size 16

write_status score_strict
export JAX_ENABLE_X64=true
"${PY}" scripts/score_lj13_reference_v2.py \
  --reference "${REFERENCE}" \
  --run-dir "${RUN}" \
  --beta 1.0 \
  --kinds fm am \
  --energy-samples 2000 \
  --geometric-samples 2000 \
  --reference-floor-repeats 5 \
  --seed 0 \
  --output "${STRICT_METRICS}"
test -f "${STRICT_METRICS}"

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
  "${STRICT_METRICS}" \
  "${RUN}/parent_fm_config.yaml" \
  "${RUN}/parent_experiment_binding.json" \
  "${RUN}/replicate_binding.json" \
  "${RUN}/exact_run_script.sh" \
  "${RUN}/exact_score_script.py" \
  "${RUN}/reference_manifest.json" \
  "${RUN}/reference_train_audit.json" \
  "${RUN}/reference_eval_audit.json" \
  "${RUN}/reference_train_vs_eval.json" \
  > "${RUN}/result_SHA256SUMS"

write_status complete
echo "LJ13 corrected-reference AM seed ${AM_SEED} replicate complete: ${RUN}"
