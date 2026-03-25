#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-./data/Market/pytorch}"
TEST_DIR="${TEST_DIR:-$DATA_DIR}"
TRAIN_NAME="${TRAIN_NAME:-macm4_day8_efficientnet_b4}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
TEST_BATCH="${TEST_BATCH:-64}"
WORKERS="${WORKERS:-0}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-300}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FORMAL_TRAIN="${RUN_FORMAL_TRAIN:-0}"
RUN_TEST_AFTER_TRAIN="${RUN_TEST_AFTER_TRAIN:-0}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/macm4_day8}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ENV_LOG="$LOG_DIR/${STAMP}_env.log"
SMOKE_LOG="$LOG_DIR/${STAMP}_smoke.log"
TRAIN_LOG="$LOG_DIR/${STAMP}_train.log"
TEST_LOG="$LOG_DIR/${STAMP}_test.log"
EVAL_LOG="$LOG_DIR/${STAMP}_eval.log"
SUMMARY_LOG="$LOG_DIR/${STAMP}_summary.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY_LOG"; }
run_and_log() { local logfile="$1"; shift; "$@" 2>&1 | tee "$logfile"; }

log "macm4 day-8 EfficientNet-b4 strategy started"
log "ROOT_DIR=$ROOT_DIR"
log "DATA_DIR=$DATA_DIR"
log "TEST_DIR=$TEST_DIR"
log "TRAIN_NAME=$TRAIN_NAME"
log "TRAIN_BATCH=$TRAIN_BATCH TEST_BATCH=$TEST_BATCH WORKERS=$WORKERS"
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
  log "Running EfficientNet-b4 smoke test with timeout=${SMOKE_TIMEOUT}s"
  set +e
  python train.py \
    --data_dir "$DATA_DIR" \
    --name "$TRAIN_NAME" \
    --use_efficient \
    --batchsize "$TRAIN_BATCH" \
    --workers "$WORKERS" \
    --total_epoch 1 \
    --erasing_p 0 \
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
  log "Starting formal EfficientNet-b4 training on macm4"
  run_and_log "$TRAIN_LOG" python train.py --data_dir "$DATA_DIR" --name "$TRAIN_NAME" --use_efficient --batchsize "$TRAIN_BATCH" --workers "$WORKERS"
  if [[ "$RUN_TEST_AFTER_TRAIN" == "1" ]]; then
    log "Running test.py after training"
    run_and_log "$TEST_LOG" python test.py --test_dir "$TEST_DIR" --name "$TRAIN_NAME" --use_efficient --batchsize "$TEST_BATCH" --workers "$WORKERS"
    log "Running evaluate.py after test"
    run_and_log "$EVAL_LOG" python evaluate.py
  else
    log "Skipping test/evaluate after training"
  fi
else
  log "Formal training not started. To continue today, rerun with RUN_FORMAL_TRAIN=1"
fi

log "macm4 day-8 strategy completed"
log "Summary log: $SUMMARY_LOG"
