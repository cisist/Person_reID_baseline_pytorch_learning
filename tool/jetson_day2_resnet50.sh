#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-../Market/pytorch}"
TEST_DIR="${TEST_DIR:-$DATA_DIR}"
TRAIN_NAME="${TRAIN_NAME:-ft_ResNet50}"
TEST_EPOCHS="${TEST_EPOCHS:-best,last}"
GPU_IDS="${GPU_IDS:-0}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-300}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FORMAL_TRAIN="${RUN_FORMAL_TRAIN:-0}"
RUN_TEST_AFTER_TRAIN="${RUN_TEST_AFTER_TRAIN:-0}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/jetson_day2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ENV_LOG="$LOG_DIR/${STAMP}_env.log"
SMOKE_LOG="$LOG_DIR/${STAMP}_smoke.log"
TRAIN_LOG="$LOG_DIR/${STAMP}_train.log"
SUMMARY_LOG="$LOG_DIR/${STAMP}_summary.log"

log() {
  echo "[$(date "+%F %T")] $*" | tee -a "$SUMMARY_LOG"
}

run_and_log() {
  local logfile="$1"
  shift
  "$@" 2>&1 | tee "$logfile"
}

checkpoint_path() {
  local epoch="$1"
  if [[ "$epoch" =~ ^[0-9]+$ ]]; then
    printf "%s/model/%s/net_%03d.pth" "$ROOT_DIR" "$TRAIN_NAME" "$epoch"
  else
    printf "%s/model/%s/net_%s.pth" "$ROOT_DIR" "$TRAIN_NAME" "$epoch"
  fi
}

run_post_train_eval() {
  local raw_epoch
  IFS="," read -r -a eval_epochs <<< "$TEST_EPOCHS"
  for raw_epoch in "${eval_epochs[@]}"; do
    local epoch="${raw_epoch//[[:space:]]/}"
    local test_log
    local eval_log
    local result_txt="$ROOT_DIR/model/$TRAIN_NAME/result.txt"
    local ckpt
    [[ -n "$epoch" ]] || continue
    ckpt="$(checkpoint_path "$epoch")"
    if [[ ! -f "$ckpt" ]]; then
      log "WARNING: checkpoint not found for epoch=$epoch, skip: $ckpt"
      continue
    fi
    test_log="$LOG_DIR/${STAMP}_test_${epoch}.log"
    eval_log="$LOG_DIR/${STAMP}_eval_${epoch}.log"
    mkdir -p "$(dirname "$result_txt")"
    printf "
===== TEST_EPOCH=%s STAMP=%s =====
" "$epoch" "$STAMP" >> "$result_txt"
    log "Running test.py for epoch=$epoch"
    run_and_log "$test_log" "${BASE_TEST_CMD[@]}" --which_epoch "$epoch"
    log "Running evaluate.py for epoch=$epoch"
    run_and_log "$eval_log" python evaluate.py
  done
}

TRAIN_CMD=(
  "python"
  "train.py"
  "--gpu_ids"
  "$GPU_IDS"
  "--data_dir"
  "$DATA_DIR"
  "--name"
  "$TRAIN_NAME"
  "--train_all"
)

BASE_TEST_CMD=(
  "python"
  "test.py"
  "--gpu_ids"
  "$GPU_IDS"
  "--test_dir"
  "$TEST_DIR"
  "--name"
  "$TRAIN_NAME"
)

log "Jetson day-2 strict reproduction started"
log "ROOT_DIR=$ROOT_DIR"
log "DATA_DIR=$DATA_DIR"
log "TEST_DIR=$TEST_DIR"
log "TRAIN_NAME=$TRAIN_NAME"
log "TEST_EPOCHS=$TEST_EPOCHS GPU_IDS=$GPU_IDS"
log "RUN_SMOKE=$RUN_SMOKE RUN_FORMAL_TRAIN=$RUN_FORMAL_TRAIN RUN_TEST_AFTER_TRAIN=$RUN_TEST_AFTER_TRAIN"
log "README reference: python train.py --train_all"
log "NOTE: Strict reproduction mode: default training flags follow README Trained Model."

[[ -d "$DATA_DIR" ]] || { log "ERROR: DATA_DIR does not exist: $DATA_DIR"; exit 1; }
[[ -d "$TEST_DIR" ]] || { log "ERROR: TEST_DIR does not exist: $TEST_DIR"; exit 1; }

log "Collecting environment snapshot"
{
  echo "pwd=$(pwd)"
  echo "python=$(command -v python || true)"
  python --version || true
  nvidia-smi || true
  python - <<'PY'
mods = ['torch', 'torchvision']
for name in mods:
    try:
        mod = __import__(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=IMPORT_ERROR:{exc}")
PY
  echo "DATA_DIR=$DATA_DIR"
  echo "TEST_DIR=$TEST_DIR"
} 2>&1 | tee "$ENV_LOG"

if [[ "$RUN_SMOKE" == "1" ]]; then
  log "Running ResNet-50 smoke test with timeout=${SMOKE_TIMEOUT}s"
  set +e
  timeout "$SMOKE_TIMEOUT" "${TRAIN_CMD[@]}" 2>&1 | tee "$SMOKE_LOG"
  smoke_status=${PIPESTATUS[0]}
  set -e
  if [[ "$smoke_status" -ne 0 && "$smoke_status" -ne 124 ]]; then
    log "ERROR: smoke test failed with exit code $smoke_status"
    exit "$smoke_status"
  fi
  if [[ "$smoke_status" -eq 124 ]]; then
    log "Smoke test timed out as expected after ${SMOKE_TIMEOUT}s; startup path looks healthy"
  else
    log "Smoke test exited cleanly before timeout"
  fi
else
  log "Skipping smoke test"
fi

if [[ "$RUN_FORMAL_TRAIN" == "1" ]]; then
  log "Starting formal ResNet-50 training"
  run_and_log "$TRAIN_LOG" "${TRAIN_CMD[@]}"
  if [[ "$RUN_TEST_AFTER_TRAIN" == "1" ]]; then
    run_post_train_eval
  else
    log "Skipping test/evaluate after training"
  fi
else
  log "Formal training not started. To continue today, rerun with RUN_FORMAL_TRAIN=1"
fi

log "Day-2 strict reproduction completed"
log "Summary log: $SUMMARY_LOG"
