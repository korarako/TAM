#!/usr/bin/env bash
set -Eeuo pipefail

# Corrected-reference LJ13 primary run.
# One shared FM seed (0), one pre-registered primary AM seed (2), and paired
# evaluation seed 0. AM seeds 1 and 3 are launched only after this run passes.

ROOT=${ADTM_ROOT}
PY=${CONDA_ROOT}/envs/ab/bin/python
REFERENCE="${ROOT}/data/lj13_reference_v2"
EXPECTED_REFERENCE_MANIFEST_SHA=5487261ba70d2e4c3c10189432dfa31a67c6f32ba3575fe3ec1c388a7039fdb1
TAG=${1:-lj13_reference_v2_painn128x5_anchor_0p8_1p2_target_1p0_seed2_run1}
RUN="${ROOT}/outputs/lj13/${TAG}"
LOG="${ROOT}/logs/${TAG}.log"
STATUS="${RUN}/status.txt"

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
    printf 'am_seed=2\n'
    printf 'evaluation_seed=0\n'
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

write_status validate_reference
test "$(sha256sum "${REFERENCE}/manifest.json" | awk '{print $1}')" = \
  "${EXPECTED_REFERENCE_MANIFEST_SHA}"
test "$(readlink -f "${REFERENCE}/train")" != "$(readlink -f "${REFERENCE}/eval")"
(
  cd "${REFERENCE}"
  sha256sum -c SHA256SUMS
)
"${PY}" scripts/generate_lj13_reference_v2.py validate \
  --output "${REFERENCE}" > "${RUN}/reference_validation.json"
"${PY}" -c '
import json
from pathlib import Path
m = json.loads(Path("'"${REFERENCE}"'/manifest.json").read_text())
assert m["schema"] == "adtm.lj13_reference_v2.bundle.v1"
assert m["status"] == "validated_equilibrium_reference"
assert m["target"]["id"] == "lj13_bms_eq234_lj1_confinement1_comfree_v1"
assert m["cross_split_status"] == "pass"
assert m["legacy_reference_used"] is False
assert m["legacy_unadjusted_langevin_used"] is False
for split in ("train", "eval"):
    assert m["splits"][split]["audit_status"] == "pass"
    assert m["splits"][split]["publishable"] is True
    assert m["splits"][split]["config"]["target_betas"] == [0.8, 1.0, 1.2]
'

cp "${REFERENCE}/manifest.json" "${RUN}/reference_manifest.json"
cp "${REFERENCE}/SHA256SUMS" "${RUN}/reference_SHA256SUMS"
cp "${REFERENCE}/diagnostics/train_audit.json" "${RUN}/reference_train_audit.json"
cp "${REFERENCE}/diagnostics/eval_audit.json" "${RUN}/reference_eval_audit.json"
cp "${REFERENCE}/diagnostics/train_vs_eval.json" "${RUN}/reference_train_vs_eval.json"
cp "${ROOT}/scripts/run_lj13_reference_v2_seed2.sh" "${RUN}/exact_run_script.sh"
cp "${ROOT}/scripts/score_lj13_reference_v2.py" "${RUN}/exact_score_script.py"

cat > "${RUN}/experiment_binding.json" <<EOF
{
  "schema": "adtm.lj13_reference_v2.primary_run.v1",
  "reference_manifest_sha256": "${EXPECTED_REFERENCE_MANIFEST_SHA}",
  "training_split": "train",
  "evaluation_split": "eval",
  "fm_seed": 0,
  "am_seed": 2,
  "evaluation_seed": 0,
  "fm_anchor_betas": [0.8, 1.2],
  "am_pair": [0.8, 1.0],
  "training_dtype": "float32",
  "reference_generation_dtype": "float64",
  "legacy_reference_used": false
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

write_status train_fm
"${PY}" main.py train-fm \
  "${COMMON[@]}" \
  --seed 0 \
  --n-per-beta 100000 \
  --n-eval 100000 \
  --fm-steps 100000 \
  --batch-size 64 \
  --fm-lr 3.0e-4 \
  --fm-lr-min 3.0e-4 \
  --fm-samples-per-beta 0 \
  --fm-eval-every 0 \
  --grad-clip 1.0 \
  --ode-steps 150 \
  --ode-method euler \
  --save-fm-checkpoints 10000 25000 50000 75000 100000 \
  --log-every 100
sha256sum "${RUN}/fm_params.pkl" > "${RUN}/fm_params.sha256"

write_status train_am
"${PY}" main.py train-am \
  "${COMMON[@]}" \
  --seed 2 \
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

write_status eval_fm_100k
"${PY}" main.py eval-fm-betas \
  "${COMMON[@]}" \
  --seed 0 \
  --eval-betas 0.8 1.0 1.2 \
  --n-eval 100000 \
  --eval-chunk-size 1000 \
  --ode-steps 150 \
  --ode-method euler \
  --energy-w2-samples 20000 \
  --geometric-w2-samples 0 \
  --geometric-w2-chunk-size 16

write_status eval_am_100k
"${PY}" main.py eval-am-betas \
  "${COMMON[@]}" \
  --seed 0 \
  --eval-betas 1.0 \
  --n-eval 100000 \
  --eval-chunk-size 1000 \
  --ode-steps 150 \
  --ode-method euler \
  --energy-w2-samples 20000 \
  --geometric-w2-samples 0 \
  --geometric-w2-chunk-size 16

write_status score_target_strict
JAX_ENABLE_X64=true "${PY}" scripts/score_lj13_reference_v2.py \
  --bundle "${REFERENCE}" \
  --run-dir "${RUN}" \
  --beta 1.0 \
  --kinds fm am \
  --energy-samples 20000 \
  --geometric-samples 2000 \
  --reference-floor-repeats 5 \
  --seed 0 \
  --output "${RUN}/score_lj13_reference_v2.json"

write_status hash_results
sha256sum \
  "${RUN}/fm_params.pkl" \
  "${RUN}/am_params.pkl" \
  "${RUN}/fm_loss_history.npy" \
  "${RUN}/am_loss_history.npy" \
  "${RUN}/fm_beta_sweep/fm_samples_beta_0.80.npy" \
  "${RUN}/fm_beta_sweep/fm_samples_beta_1.00.npy" \
  "${RUN}/fm_beta_sweep/fm_samples_beta_1.20.npy" \
  "${RUN}/am_beta_sweep/am_samples_beta_1.00.npy" \
  "${RUN}/config.yaml" \
  "${RUN}/experiment_binding.json" \
  "${RUN}/exact_run_script.sh" \
  "${RUN}/exact_score_script.py" \
  "${RUN}/score_lj13_reference_v2.json" \
  "${RUN}/reference_manifest.json" \
  "${RUN}/reference_train_audit.json" \
  "${RUN}/reference_eval_audit.json" \
  "${RUN}/reference_train_vs_eval.json" \
  > "${RUN}/result_SHA256SUMS"

write_status complete
echo "LJ13 corrected-reference primary seed-2 run complete: ${RUN}"
