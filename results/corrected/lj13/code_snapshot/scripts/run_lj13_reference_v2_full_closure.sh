#!/usr/bin/env bash
set -Eeuo pipefail

# Continue the corrected-reference LJ13 experiment after the already-launched
# primary seed-2 run.  The wrapper waits for that run, validates its hashes,
# executes AM seeds 1 and 3 on the same GPU, aggregates, plots, and creates the
# checkpoint-free frozen result bundle.

ROOT=${ADTM_ROOT}
PY=${CONDA_ROOT}/envs/ab/bin/python
REFERENCE="${ROOT}/data/lj13_reference_v2"
PRIMARY_TAG=${PRIMARY_TAG:-lj13_reference_v2_painn128x5_anchor_0p8_1p2_target_1p0_seed2_run1}
PRIMARY="${ROOT}/outputs/lj13/${PRIMARY_TAG}"
SEED1_TAG=${SEED1_TAG:-lj13_reference_v2_painn128x5_anchor_0p8_1p2_target_1p0_seed1_run1}
SEED3_TAG=${SEED3_TAG:-lj13_reference_v2_painn128x5_anchor_0p8_1p2_target_1p0_seed3_run1}
SEED1="${ROOT}/outputs/lj13/${SEED1_TAG}"
SEED3="${ROOT}/outputs/lj13/${SEED3_TAG}"
PIPELINE_TAG=${PIPELINE_TAG:-lj13_reference_v2_full_closure_run1}
PIPELINE="${ROOT}/outputs/lj13/${PIPELINE_TAG}"
STATUS="${PIPELINE}/status.txt"
LOG="${ROOT}/logs/${PIPELINE_TAG}.log"
AGGREGATE="${PRIMARY}/aggregate_lj13_reference_v2_am_seeds"
PLOTS="${PRIMARY}/publication_figures_lj13_reference_v2"
FREEZE="${ROOT}/frozen_results/2026-07-26/lj13_adtm_reference_v2"
POLL_SECONDS=${POLL_SECONDS:-30}

if [[ -e "${PIPELINE}" ]]; then
  echo "Refusing to overwrite pipeline directory: ${PIPELINE}" >&2
  exit 2
fi
if [[ -e "${SEED1}" || -e "${SEED3}" ]]; then
  echo "Refusing to overwrite an existing AM replicate directory." >&2
  exit 2
fi
if [[ -e "${AGGREGATE}" || -e "${PLOTS}" || -e "${FREEZE}" ]]; then
  echo "Refusing to overwrite aggregate, plot, or freeze outputs." >&2
  exit 2
fi

mkdir -p "${PIPELINE}" "${ROOT}/logs"
exec > >(tee -a "${LOG}") 2>&1

current_phase=initializing
write_status() {
  current_phase=$1
  {
    printf 'phase=%s\n' "${current_phase}"
    printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host=%s\n' "$(hostname)"
    printf 'pid=%s\n' "$$"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'primary=%s\n' "${PRIMARY}"
    printf 'am_seed1=%s\n' "${SEED1}"
    printf 'am_seed3=%s\n' "${SEED3}"
  } > "${STATUS}"
}

on_error() {
  local rc=$?
  {
    printf 'phase=failed\n'
    printf 'failed_during=%s\n' "${current_phase}"
    printf 'exit_code=%s\n' "${rc}"
    printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host=%s\n' "$(hostname)"
    printf 'pid=%s\n' "$$"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  } > "${STATUS}"
  exit "${rc}"
}
trap on_error ERR

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONUNBUFFERED=1

write_status wait_primary
while true; do
  if [[ -f "${PRIMARY}/status.txt" ]]; then
    primary_phase=$(awk -F= '$1 == "phase" {print $2}' "${PRIMARY}/status.txt" | tail -n 1)
    if [[ "${primary_phase}" == "complete" ]]; then
      break
    fi
    if [[ "${primary_phase}" == "failed" ]]; then
      echo "Primary run failed; continuation stops." >&2
      false
    fi
  fi
  sleep "${POLL_SECONDS}"
done

write_status validate_primary
test -f "${PRIMARY}/result_SHA256SUMS"
sha256sum -c "${PRIMARY}/result_SHA256SUMS"
EXPECTED_FM_SHA=$(sha256sum "${PRIMARY}/fm_params.pkl" | awk '{print $1}')
test "${EXPECTED_FM_SHA}" = "$(awk 'NR == 1 {print $1}' "${PRIMARY}/fm_params.sha256")"

write_status am_seed1
AM_SEED=1 \
EXPECTED_FM_SHA="${EXPECTED_FM_SHA}" \
PRIMARY_RUN="${PRIMARY}" \
RUN_TAG="${SEED1_TAG}" \
bash "${ROOT}/scripts/run_lj13_reference_v2_am_replicate.sh"
grep -qx 'phase=complete' "${SEED1}/status.txt"
sha256sum -c "${SEED1}/result_SHA256SUMS"

write_status am_seed3
AM_SEED=3 \
EXPECTED_FM_SHA="${EXPECTED_FM_SHA}" \
PRIMARY_RUN="${PRIMARY}" \
RUN_TAG="${SEED3_TAG}" \
bash "${ROOT}/scripts/run_lj13_reference_v2_am_replicate.sh"
grep -qx 'phase=complete' "${SEED3}/status.txt"
sha256sum -c "${SEED3}/result_SHA256SUMS"

write_status aggregate
"${PY}" scripts/aggregate_lj13_reference_v2_am_seeds.py \
  --metric "1=${SEED1}/score_lj13_reference_v2.json" \
  --metric "2=${PRIMARY}/score_lj13_reference_v2.json" \
  --metric "3=${SEED3}/score_lj13_reference_v2.json" \
  --output-dir "${AGGREGATE}"
(
  cd "${AGGREGATE}"
  sha256sum -c AGGREGATE_SHA256SUMS
)

write_status plot
JAX_ENABLE_X64=true "${PY}" scripts/plot_lj13_reference_v2.py \
  --reference-bundle "${REFERENCE}" \
  --primary-run "${PRIMARY}" \
  --primary-am-seed 2 \
  --am-run "1=${SEED1}" \
  --am-run "3=${SEED3}" \
  --output-dir "${PLOTS}" \
  --expected-rows 100000 \
  --metric-rows 2000 \
  --reference-floor-repeats 5 \
  --metric-seed 0
test -f "${PLOTS}/figure_metadata.json"
test "$(find "${PLOTS}" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 5
test "$(find "${PLOTS}" -maxdepth 1 -type f -name '*.pdf' | wc -l)" -eq 5

write_status freeze
"${PY}" scripts/freeze_lj13_reference_v2_results.py \
  --reference-bundle "${REFERENCE}" \
  --primary-run "${PRIMARY}" \
  --primary-am-seed 2 \
  --am-run "1=${SEED1}" \
  --am-run "3=${SEED3}" \
  --plots-dir "${PLOTS}" \
  --output-dir "${FREEZE}" \
  --include-code scripts/aggregate_lj13_reference_v2_am_seeds.py \
  --include-code scripts/run_lj13_reference_v2_full_closure.sh
(
  cd "${FREEZE}"
  sha256sum -c SHA256SUMS
)
if find "${FREEZE}" -type f \( -name '*.pkl' -o -name '*.npy' -o -name '*.npz' -o -name '*.ckpt' -o -name '*.pt' \) | grep -q .; then
  echo "Forbidden checkpoint/sample payload found in frozen bundle." >&2
  false
fi

write_status complete
echo "LJ13 corrected-reference ADTM full closure complete: ${FREEZE}"
