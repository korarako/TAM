#!/usr/bin/env bash
set -Eeuo pipefail

# Cross-host safe queue for the canonical score2k recovery followed immediately
# by the remaining seed1/seed3/aggregate/plot/freeze closure on the same idle
# GPU.  Recovery changes no checkpoint or sample.

ROOT=${ADTM_ROOT}
PRIMARY_TAG=lj13_reference_v2_painn128x5_anchor_0p8_1p2_target_1p0_seed2_run1
PRIMARY="${ROOT}/outputs/lj13/${PRIMARY_TAG}"
PIPELINE_TAG=lj13_reference_v2_full_closure_run2
LOCK="${ROOT}/outputs/lj13/.${PIPELINE_TAG}.launch.lock"
STATUS="${ROOT}/outputs/lj13/${PIPELINE_TAG}.queue.$(hostname).status.txt"
LOG="${ROOT}/logs/${PIPELINE_TAG}.queue.$(hostname).log"
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_USED_MIB=${MAX_USED_MIB:-1000}
MAX_UTIL_PERCENT=${MAX_UTIL_PERCENT:-5}

mkdir -p "${ROOT}/outputs/lj13" "${ROOT}/logs"
exec >> "${LOG}" 2>&1

write_status() {
  {
    printf 'phase=%s\n' "$1"
    printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host=%s\n' "$(hostname)"
    printf 'pid=%s\n' "$$"
    printf 'candidate_gpu=%s\n' "${2:-none}"
  } > "${STATUS}"
}

find_idle_gpu() {
  nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
  awk -F, -v max_mem="${MAX_USED_MIB}" -v max_util="${MAX_UTIL_PERCENT}" '
    {
      gsub(/ /, "", $1);
      gsub(/ /, "", $2);
      gsub(/ /, "", $3);
      if (($2 + 0) <= max_mem && ($3 + 0) <= max_util) {
        print $1;
        exit;
      }
    }
  '
}

write_status waiting
while true; do
  if [[ -f "${ROOT}/outputs/lj13/${PIPELINE_TAG}/status.txt" ]]; then
    phase=$(awk -F= '$1 == "phase" {print $2}' \
      "${ROOT}/outputs/lj13/${PIPELINE_TAG}/status.txt" | tail -n 1)
    write_status "pipeline_${phase:-present}"
    exit 0
  fi

  gpu=$(find_idle_gpu || true)
  if [[ -z "${gpu}" ]]; then
    write_status waiting
    sleep "${POLL_SECONDS}"
    continue
  fi
  if ! mkdir "${LOCK}" 2>/dev/null; then
    write_status claimed_elsewhere "${gpu}"
    exit 0
  fi
  confirm_gpu=$(find_idle_gpu || true)
  if [[ "${confirm_gpu}" != "${gpu}" ]]; then
    rmdir "${LOCK}"
    write_status lost_candidate "${gpu}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  {
    printf 'host=%s\n' "$(hostname)"
    printf 'pid=%s\n' "$$"
    printf 'gpu=%s\n' "${gpu}"
    printf 'claimed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${LOCK}/owner.txt"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export JAX_DEFAULT_MATMUL_PRECISION=highest
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export PYTHONUNBUFFERED=1

  write_status score2k_recovery "${gpu}"
  bash "${ROOT}/scripts/recover_lj13_reference_v2_primary_score.sh" \
    "${PRIMARY_TAG}"

  write_status full_closure "${gpu}"
  PIPELINE_TAG="${PIPELINE_TAG}" \
  bash "${ROOT}/scripts/run_lj13_reference_v2_full_closure.sh"
  write_status complete "${gpu}"
  exit 0
done
