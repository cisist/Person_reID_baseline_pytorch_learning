#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-./data/Market/pytorch}"
TEST_DIR="${TEST_DIR:-$DATA_DIR}"
TRAIN_NAME="${TRAIN_NAME:-macm4_day17_resnet50_all_tricks}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
TEST_BATCH="${TEST_BATCH:-64}"
TEST_EPOCH="${TEST_EPOCH:-best}"
RUN_TEST_LAST="${RUN_TEST_LAST:-0}"
WORKERS="${WORKERS:-0}"
LR="${LR:-0.02}"
ERASING_P="${ERASING_P:-0.5}"
WARM_EPOCH="${WARM_EPOCH:-5}"
STRIDE="${STRIDE:-1}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-300}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FORMAL_TRAIN="${RUN_FORMAL_TRAIN:-0}"
RUN_TEST_AFTER_TRAIN="${RUN_TEST_AFTER_TRAIN:-0}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/macm4_day17}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ENV_LOG="$LOG_DIR/${STAMP}_env.log"
SMOKE_LOG="$LOG_DIR/${STAMP}_smoke.log"
TRAIN_LOG="$LOG_DIR/${STAMP}_train.log"
TEST_LOG="$LOG_DIR/${STAMP}_test.log"
LAST_TEST_LOG="$LOG_DIR/${STAMP}_test_last.log"
EVAL_LOG="$LOG_DIR/${STAMP}_eval.log"
SUMMARY_LOG="$LOG_DIR/${STAMP}_summary.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY_LOG"; }
run_and_log() { local logfile="$1"; shift; "$@" 2>&1 | tee "$logfile"; }

log "macm4 day-17 ResNet-50 all-tricks strategy started"
log "TRAIN_NAME=$TRAIN_NAME TRAIN_BATCH=$TRAIN_BATCH TEST_BATCH=$TEST_BATCH TEST_EPOCH=$TEST_EPOCH RUN_TEST_LAST=$RUN_TEST_LAST WORKERS=$WORKERS LR=$LR ERASING_P=$ERASING_P WARM_EPOCH=$WARM_EPOCH STRIDE=$STRIDE"

[[ -d "$DATA_DIR" ]] || { log "ERROR: DATA_DIR does not exist: $DATA_DIR"; exit 1; }
[[ -d "$TEST_DIR" ]] || { log "ERROR: TEST_DIR does not exist: $TEST_DIR"; exit 1; }

{
  python --version || true
  python - <<'PY'
import torch
print(f"cuda={torch.cuda.is_available()}")
print(f"mps={hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
PY
} 2>&1 | tee "$ENV_LOG"

if [[ "$RUN_SMOKE" == "1" ]]; then
  set +e
  python train.py --data_dir "$DATA_DIR" --name "$TRAIN_NAME" --warm_epoch "$WARM_EPOCH" --stride "$STRIDE" --erasing_p 0 --batchsize "$TRAIN_BATCH" --lr "$LR" --workers "$WORKERS" --total_epoch 1 2>&1 | tee "$SMOKE_LOG" &
  smoke_pid=$!
  ( sleep "$SMOKE_TIMEOUT"; kill -0 "$smoke_pid" 2>/dev/null && kill "$smoke_pid" 2>/dev/null || true ) &
  watcher_pid=$!
  wait "$smoke_pid"
  smoke_status=$?
  kill "$watcher_pid" 2>/dev/null || true
  wait "$watcher_pid" 2>/dev/null || true
  set -e
  [[ "$smoke_status" -eq 0 || "$smoke_status" -eq 143 ]] || { log "ERROR: smoke failed with $smoke_status"; exit "$smoke_status"; }
fi

if [[ "$RUN_FORMAL_TRAIN" == "1" ]]; then
  run_and_log "$TRAIN_LOG" python train.py --data_dir "$DATA_DIR" --name "$TRAIN_NAME" --warm_epoch "$WARM_EPOCH" --stride "$STRIDE" --erasing_p "$ERASING_P" --batchsize "$TRAIN_BATCH" --lr "$LR" --workers "$WORKERS"
  if [[ "$RUN_TEST_AFTER_TRAIN" == "1" ]]; then
    run_and_log "$TEST_LOG" python test.py --test_dir "$TEST_DIR" --name "$TRAIN_NAME" --which_epoch "$TEST_EPOCH" --batchsize "$TEST_BATCH" --workers "$WORKERS"
    if [[ "$RUN_TEST_LAST" == "1" && "$TEST_EPOCH" != "last" ]]; then
      run_and_log "$LAST_TEST_LOG" python test.py --test_dir "$TEST_DIR" --name "$TRAIN_NAME" --which_epoch last --batchsize "$TEST_BATCH" --workers "$WORKERS"
    fi
    run_and_log "$EVAL_LOG" python evaluate.py
  fi
else
  log "Formal training not started. To continue today, rerun with RUN_FORMAL_TRAIN=1"
fi

log "macm4 day-17 strategy completed"
