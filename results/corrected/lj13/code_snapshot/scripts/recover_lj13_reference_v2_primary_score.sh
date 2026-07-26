#!/usr/bin/env bash
set -Eeuo pipefail

# Resume only the canonical 2k strict scoring/hash stage after the original
# absolute COM gate rejected a handful of very large but scale-correct float32
# samples and the draft 20k metric protocol was found not to match the
# historical LJ13 2k benchmark.
# No checkpoint, sample, optimizer state, training step, or sampling trajectory
# is changed.  All raw 100k samples remain in the score.

ROOT=${ADTM_ROOT}
PY=${CONDA_ROOT}/envs/ab/bin/python
REFERENCE="${ROOT}/data/lj13_reference_v2"
TAG=${1:-lj13_reference_v2_painn128x5_anchor_0p8_1p2_target_1p0_seed2_run1}
RUN="${ROOT}/outputs/lj13/${TAG}"
LOG="${ROOT}/logs/${TAG}.score_recovery.log"
STATUS="${RUN}/status.txt"

test -d "${RUN}"
test -f "${STATUS}"
grep -qx 'phase=failed' "${STATUS}"
test ! -e "${RUN}/score_lj13_reference_v2.json"
test ! -e "${RUN}/result_SHA256SUMS"

exec > >(tee -a "${LOG}") 2>&1
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONUNBUFFERED=1

cp "${STATUS}" "${RUN}/status_failed_pre_score_recovery.txt"
OLD_SCORE_SHA=$(sha256sum "${RUN}/exact_score_script.py" | awk '{print $1}')
NEW_SCORE_SHA=$(sha256sum scripts/score_lj13_reference_v2.py | awk '{print $1}')
cp scripts/score_lj13_reference_v2.py "${RUN}/exact_score_script.py"
cp scripts/recover_lj13_reference_v2_primary_score.sh \
  "${RUN}/exact_score_recovery_script.sh"

cat > "${RUN}/score_recovery_binding.json" <<EOF
{
  "schema": "adtm.lj13_reference_v2.score_recovery.v2",
  "scope": "evaluation health gate and strict score only",
  "training_changed": false,
  "sampling_changed": false,
  "samples_filtered": false,
  "samples_recentered": false,
  "raw_sample_rows_retained": 100000,
  "reason": "Six FM and one AM rows had very large coordinates; float32 summation left absolute COM residuals above 1e-5 while scale-normalized residuals remained below 1e-7.",
  "metric_protocol_correction": {
    "draft_protocol": "20k rows for energy, pair distance, radius of gyration, and minimum pair distance; 2k rows for geometry",
    "corrected_protocol": "2k rows for every benchmark metric",
    "reason": "The old 20k draft score was not historically comparable to the established LJ13 2k benchmark.",
    "training_changed": false,
    "sampling_changed": false,
    "retraining_or_resampling_required": false
  },
  "old_score_script_sha256": "${OLD_SCORE_SHA}",
  "new_score_script_sha256": "${NEW_SCORE_SHA}",
  "new_com_rule": "row_max_abs_com <= 1e-5 + 1e-7 * max(1, row_max_abs_coordinate)"
}
EOF

{
  printf 'phase=score_target_strict_recovery\n'
  printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "$(hostname)"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'training_changed=false\n'
  printf 'sampling_changed=false\n'
} > "${STATUS}"

JAX_ENABLE_X64=true "${PY}" scripts/score_lj13_reference_v2.py \
  --bundle "${REFERENCE}" \
  --run-dir "${RUN}" \
  --beta 1.0 \
  --kinds fm am \
  --energy-samples 2000 \
  --geometric-samples 2000 \
  --reference-floor-repeats 5 \
  --seed 0 \
  --output "${RUN}/score_lj13_reference_v2.json"

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
  "${RUN}/exact_score_recovery_script.sh" \
  "${RUN}/score_recovery_binding.json" \
  "${RUN}/status_failed_pre_score_recovery.txt" \
  "${RUN}/score_lj13_reference_v2.json" \
  "${RUN}/reference_manifest.json" \
  "${RUN}/reference_train_audit.json" \
  "${RUN}/reference_eval_audit.json" \
  "${RUN}/reference_train_vs_eval.json" \
  > "${RUN}/result_SHA256SUMS"

{
  printf 'phase=complete\n'
  printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "$(hostname)"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'shared_fm_seed=0\n'
  printf 'am_seed=2\n'
  printf 'evaluation_seed=0\n'
  printf 'training_changed=false\n'
  printf 'sampling_changed=false\n'
  printf 'evaluation_recovery=scale_aware_float32_com_gate\n'
} > "${STATUS}"

echo "Recovered strict score without retraining/resampling: ${RUN}"
