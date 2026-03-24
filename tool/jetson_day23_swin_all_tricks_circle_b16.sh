#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-../Market/pytorch}"
TEST_DIR="${TEST_DIR:-$DATA_DIR}"
TRAIN_NAME="${TRAIN_NAME:-swin_p0.5_circle_w5_b16_lr0.01}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"
LR="${LR:-0.01}"
ERASING_P="${ERASING_P:-0.5}"
WARM_EPOCH="${WARM_EPOCH:-5}"
GPU_IDS="${GPU_IDS:-0}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-300}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FORMAL_TRAIN="${RUN_FORMAL_TRAIN:-0}"
RUN_TEST_AFTER_TRAIN="${RUN_TEST_AFTER_TRAIN:-0}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/jetson_day23}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ENV_LOG="$LOG_DIR/${STAMP}_env.log"
SMOKE_LOG="$LOG_DIR/${STAMP}_smoke.log"
TRAIN_LOG="$LOG_DIR/${STAMP}_train.log"
TEST_LOG="$LOG_DIR/${STAMP}_test.log"
EVAL_LOG="$LOG_DIR/${STAMP}_eval.log"
SUMMARY_LOG="$LOG_DIR/${STAMP}_summary.log"

log() {
  echo "[$(date "+%F %T")] $*" | tee -a "$SUMMARY_LOG"
}

run_and_log() {
  local logfile="$1"
  shift
  "$@" 2>&1 | tee "$logfile"
}

TRAIN_CMD=(
  "python"
  "train.py"
  "--gpu_ids"
  "$GPU_IDS"
  "--data_dir"
  "$DATA_DIR"
  "--use_swin"
  "--name"
  "$TRAIN_NAME"
  "--lr"
  "$LR"
  "--batchsize"
  "$TRAIN_BATCH"
  "--erasing_p"
  "$ERASING_P"
  "--circle"
  "--warm_epoch"
  "$WARM_EPOCH"
)

TEST_CMD=(
  "python"
  "test.py"
  "--gpu_ids"
  "$GPU_IDS"
  "--test_dir"
  "$TEST_DIR"
  "--name"
  "$TRAIN_NAME"
)

log "Jetson day-23 strict reproduction started"
log "ROOT_DIR=$ROOT_DIR"
log "DATA_DIR=$DATA_DIR"
log "TEST_DIR=$TEST_DIR"
log "TRAIN_NAME=$TRAIN_NAME"
log "TRAIN_BATCH=$TRAIN_BATCH LR=$LR ERASING_P=$ERASING_P WARM_EPOCH=$WARM_EPOCH GPU_IDS=$GPU_IDS"
log "RUN_SMOKE=$RUN_SMOKE RUN_FORMAL_TRAIN=$RUN_FORMAL_TRAIN RUN_TEST_AFTER_TRAIN=$RUN_TEST_AFTER_TRAIN"
log "README reference: python train.py --use_swin --name swin_p0.5_circle_w5_b16_lr0.01 --lr 0.01 --batch 16 --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name swin_p0.5_circle_w5_b16_lr0.01"
log "NOTE: Strict reproduction mode: numeric defaults follow README. train.py uses --batchsize instead of --batch."

[[ -d "$DATA_DIR" ]] || { log "ERROR: DATA_DIR does not exist: $DATA_DIR"; exit 1; }
[[ -d "$TEST_DIR" ]] || { log "ERROR: TEST_DIR does not exist: $TEST_DIR"; exit 1; }

log "Collecting environment snapshot"
{
  echo "pwd=$(pwd)"
  echo "python=$(command -v python || true)"
  python --version || true
  nvidia-smi || true
  python - <<'PY'
mods = ['torch', 'torchvision', 'timm']
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
  log "Running Swin (all tricks+Circle+b16 224x224) smoke test with timeout=${SMOKE_TIMEOUT}s"
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
  log "Starting formal Swin (all tricks+Circle+b16 224x224) training"
  run_and_log "$TRAIN_LOG" "${TRAIN_CMD[@]}"
  if [[ "$RUN_TEST_AFTER_TRAIN" == "1" ]]; then
    log "Running test.py after training"
    run_and_log "$TEST_LOG" "${TEST_CMD[@]}"
    log "Running evaluate.py after test"
    run_and_log "$EVAL_LOG" python evaluate.py
  else
    log "Skipping test/evaluate after training"
  fi
else
  log "Formal training not started. To continue today, rerun with RUN_FORMAL_TRAIN=1"
fi

log "Day-23 strict reproduction completed"
log "Summary log: $SUMMARY_LOG"
