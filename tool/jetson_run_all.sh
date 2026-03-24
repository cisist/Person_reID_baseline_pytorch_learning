#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

START_DAY="${START_DAY:-1}"
END_DAY="${END_DAY:-24}"
DAY_LIST="${DAY_LIST:-}"
RUN_FORMAL_TRAIN="${RUN_FORMAL_TRAIN:-0}"
RUN_TEST_AFTER_TRAIN="${RUN_TEST_AFTER_TRAIN:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/jetson_master}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/${STAMP}_master.log"
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

get_script() {
  case "$1" in
    1) echo "tool/jetson_day1_fp16.sh" ;;
    2) echo "tool/jetson_day2_resnet50.sh" ;;
    3) echo "tool/jetson_day3_resnet50_ibn.sh" ;;
    4) echo "tool/jetson_day4_densenet121.sh" ;;
    5) echo "tool/jetson_day5_densenet121_circle.sh" ;;
    6) echo "tool/jetson_day6_resnet50_usam.sh" ;;
    7) echo "tool/jetson_day7_resnet50_ibn_usam.sh" ;;
    8) echo "tool/jetson_day8_efficientnet_b4.sh" ;;
    9) echo "tool/jetson_day9_convnext.sh" ;;
    10) echo "tool/jetson_day10_hrnet18.sh" ;;
    11) echo "tool/jetson_day11_resnet50_adv.sh" ;;
    12) echo "tool/jetson_day12_pcb.sh" ;;
    13) echo "tool/jetson_day13_pcb_dg.sh" ;;
    14) echo "tool/jetson_day14_swin.sh" ;;
    15) echo "tool/jetson_day15_swinv2.sh" ;;
    16) echo "tool/jetson_day16_dinov3.sh" ;;
    17) echo "tool/jetson_day17_resnet50_all_tricks.sh" ;;
    18) echo "tool/jetson_day18_resnet50_all_tricks_circle.sh" ;;
    19) echo "tool/jetson_day19_resnet50_all_tricks_circle_dg.sh" ;;
    20) echo "tool/jetson_day20_densenet_all_tricks_circle.sh" ;;
    21) echo "tool/jetson_day21_hrnet_all_tricks_circle_dg.sh" ;;
    22) echo "tool/jetson_day22_swin_all_tricks_circle.sh" ;;
    23) echo "tool/jetson_day23_swin_all_tricks_circle_b16.sh" ;;
    24) echo "tool/jetson_day24_swin_all_tricks_circle_b16_dg.sh" ;;
    *) return 1 ;;
  esac
}

days=()
if [[ -n "$DAY_LIST" ]]; then
  IFS=',' read -r -a raw_days <<< "$DAY_LIST"
  for raw_day in "${raw_days[@]}"; do
    day="${raw_day//[[:space:]]/}"
    if ! [[ "$day" =~ ^[0-9]+$ ]]; then
      log "ERROR: DAY_LIST must contain comma-separated integers, got: $raw_day"
      exit 1
    fi
    if (( day < 1 || day > 24 )); then
      log "ERROR: DAY_LIST entries must be within 1..24, got: $day"
      exit 1
    fi
    days+=("$day")
  done
else
  if ! [[ "$START_DAY" =~ ^[0-9]+$ && "$END_DAY" =~ ^[0-9]+$ ]]; then
    log "ERROR: START_DAY and END_DAY must be integers"
    exit 1
  fi

  if (( START_DAY < 1 || END_DAY > 24 || START_DAY > END_DAY )); then
    log "ERROR: valid range is 1 <= START_DAY <= END_DAY <= 24"
    exit 1
  fi

  for day in $(seq "$START_DAY" "$END_DAY"); do
    days+=("$day")
  done
fi

log "Jetson master runner started"
log "ROOT_DIR=$ROOT_DIR"
log "START_DAY=$START_DAY END_DAY=$END_DAY"
log "DAY_LIST=${DAY_LIST:-<empty>}"
log "RUN_SMOKE=$RUN_SMOKE RUN_FORMAL_TRAIN=$RUN_FORMAL_TRAIN RUN_TEST_AFTER_TRAIN=$RUN_TEST_AFTER_TRAIN STOP_ON_ERROR=$STOP_ON_ERROR"

for day in "${days[@]}"; do
  script="$(get_script "$day")"
  if [[ ! -x "$script" ]]; then
    log "ERROR: script missing or not executable: $script"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  log "===== Day $day begin: $script ====="
  set +e
  RUN_SMOKE="$RUN_SMOKE" \
  RUN_FORMAL_TRAIN="$RUN_FORMAL_TRAIN" \
  RUN_TEST_AFTER_TRAIN="$RUN_TEST_AFTER_TRAIN" \
  "$ROOT_DIR/$script" 2>&1 | tee -a "$MASTER_LOG"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -ne 0 ]]; then
    log "Day $day failed with exit code $status"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      log "Stopping because STOP_ON_ERROR=1"
      exit "$status"
    fi
  else
    log "Day $day completed successfully"
  fi
  log "===== Day $day end ====="
done

log "Jetson master runner finished"
log "Master log: $MASTER_LOG"
