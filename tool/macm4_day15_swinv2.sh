#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-./data/Market/pytorch}"
TEST_DIR="${TEST_DIR:-$DATA_DIR}"
TRAIN_NAME="${TRAIN_NAME:-macm4_day15_swinv2}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
TEST_BATCH="${TEST_BATCH:-32}"
TEST_EPOCH="${TEST_EPOCH:-best}"
RUN_TEST_LAST="${RUN_TEST_LAST:-0}"
WORKERS="${WORKERS:-0}"
LR="${LR:-0.01}"
ERASING_P="${ERASING_P:-0.5}"
WARM_EPOCH="${WARM_EPOCH:-5}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-300}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FORMAL_TRAIN="${RUN_FORMAL_TRAIN:-0}"
RUN_TEST_AFTER_TRAIN="${RUN_TEST_AFTER_TRAIN:-0}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/macm4_day15}"
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

log "macm4 day-15 SwinV2 strategy started"
log "ROOT_DIR=$ROOT_DIR"
log "DATA_DIR=$DATA_DIR"
log "TEST_DIR=$TEST_DIR"
log "TRAIN_NAME=$TRAIN_NAME"
log "TRAIN_BATCH=$TRAIN_BATCH TEST_BATCH=$TEST_BATCH TEST_EPOCH=$TEST_EPOCH RUN_TEST_LAST=$RUN_TEST_LAST WORKERS=$WORKERS"
log "LR=$LR ERASING_P=$ERASING_P WARM_EPOCH=$WARM_EPOCH"
log "RUN_SMOKE=$RUN_SMOKE RUN_FORMAL_TRAIN=$RUN_FORMAL_TRAIN RUN_TEST_AFTER_TRAIN=$RUN_TEST_AFTER_TRAIN"

[[ -d "$DATA_DIR" ]] || { log "ERROR: DATA_DIR does not exist: $DATA_DIR"; exit 1; }
[[ -d "$TEST_DIR" ]] || { log "ERROR: TEST_DIR does not exist: $TEST_DIR"; exit 1; }

log "Collecting environment snapshot"
{
  echo "pwd=$(pwd)"
  echo "python=$(command -v python || true)"
  python --version || true
  python - <<'PY'
mods = ["torch", "torchvision", "timm", "yaml", "scipy"]
for name in mods:
    try:
        mod = __import__(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=IMPORT_ERROR:{exc}")

try:
    import torch
    print(f"cuda={torch.cuda.is_available()}")
    print(f"mps={hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
except Exception as exc:
    print(f"torch_runtime_error={exc}")
PY
  echo "DATA_DIR=$DATA_DIR"
  echo "TEST_DIR=$TEST_DIR"
} 2>&1 | tee "$ENV_LOG"

if [[ "$RUN_SMOKE" == "1" ]]; then
  log "Running SwinV2 smoke test with timeout=${SMOKE_TIMEOUT}s"
  set +e
  python train.py \
    --data_dir "$DATA_DIR" \
    --name "$TRAIN_NAME" \
    --use_swinv2 \
    --lr "$LR" \
    --batchsize "$TRAIN_BATCH" \
    --erasing_p 0 \
    --circle \
    --warm_epoch "$WARM_EPOCH" \
    --workers "$WORKERS" \
    --total_epoch 1 \
    2>&1 | tee "$SMOKE_LOG" &
  smoke_pid=$!
  (
    sleep "$SMOKE_TIMEOUT"
    kill -0 "$smoke_pid" 2>/dev/null && kill "$smoke_pid" 2>/dev/null || true
  ) &
  watcher_pid=$!
  wait "$smoke_pid"
  smoke_status=$?
  kill "$watcher_pid" 2>/dev/null || true
  wait "$watcher_pid" 2>/dev/null || true
  set -e
  [[ "$smoke_status" -eq 0 || "$smoke_status" -eq 143 ]] || { log "ERROR: smoke test failed with exit code $smoke_status"; exit "$smoke_status"; }
  [[ "$smoke_status" -eq 143 ]] && log "Smoke test was stopped after ${SMOKE_TIMEOUT}s; startup path looks healthy" || log "Smoke test exited cleanly"
else
  log "Skipping smoke test"
fi

if [[ "$RUN_FORMAL_TRAIN" == "1" ]]; then
  log "Starting formal SwinV2 training on macm4"
  run_and_log "$TRAIN_LOG" python train.py --data_dir "$DATA_DIR" --name "$TRAIN_NAME" --use_swinv2 --lr "$LR" --batchsize "$TRAIN_BATCH" --erasing_p "$ERASING_P" --circle --warm_epoch "$WARM_EPOCH" --workers "$WORKERS"
  if [[ "$RUN_TEST_AFTER_TRAIN" == "1" ]]; then
    log "Running test.py after training with which_epoch=$TEST_EPOCH"
    run_and_log "$TEST_LOG" python test.py --test_dir "$TEST_DIR" --name "$TRAIN_NAME" --use_swinv2 --which_epoch "$TEST_EPOCH" --batchsize "$TEST_BATCH" --workers "$WORKERS"
    if [[ "$RUN_TEST_LAST" == "1" && "$TEST_EPOCH" != "last" ]]; then
      log "Running extra comparison test for which_epoch=last"
      run_and_log "$LAST_TEST_LOG" python test.py --test_dir "$TEST_DIR" --name "$TRAIN_NAME" --use_swinv2 --which_epoch last --batchsize "$TEST_BATCH" --workers "$WORKERS"
    fi
    log "Running evaluate.py after test"
    run_and_log "$EVAL_LOG" python evaluate.py
  else
    log "Skipping test/evaluate after training"
  fi
else
  log "Formal training not started. To continue today, rerun with RUN_FORMAL_TRAIN=1"
fi

log "macm4 day-15 strategy completed"
log "Summary log: $SUMMARY_LOG"
