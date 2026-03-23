#!/usr/bin/env bash

set -euo pipefail

RUN_NAME="${1:-mac_smoke}"
TRAIN_BATCHSIZE="${TRAIN_BATCHSIZE:-8}"
TEST_BATCHSIZE="${TEST_BATCHSIZE:-64}"
TOTAL_EPOCH="${TOTAL_EPOCH:-1}"
DATA_DIR="${DATA_DIR:-./data/Market/pytorch}"

echo "[1/5] 检查当前目录"
if [[ ! -f "train.py" || ! -f "test.py" ]]; then
  echo "请在仓库根目录运行此脚本。"
  exit 1
fi

echo "[2/5] 检查 Python 环境"
python - <<'PY'
from importlib.util import find_spec
import sys

mods = ["torch", "torchvision", "yaml", "timm", "scipy", "tqdm", "matplotlib"]
missing = [m for m in mods if find_spec(m) is None]
print("python:", sys.executable)
if missing:
    print("缺少依赖:", ", ".join(missing))
    sys.exit(1)
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("mps:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY

echo "[3/5] 检查数据目录"
if [[ ! -d "$DATA_DIR/train" || ! -d "$DATA_DIR/val" || ! -d "$DATA_DIR/query" || ! -d "$DATA_DIR/gallery" ]]; then
  echo "未检测到完整的数据目录: $DATA_DIR"
  echo "尝试运行 prepare.py ..."
  python prepare.py
fi

if [[ ! -d "$DATA_DIR/train" || ! -d "$DATA_DIR/val" || ! -d "$DATA_DIR/query" || ! -d "$DATA_DIR/gallery" ]]; then
  echo "数据目录仍不完整，请检查 $DATA_DIR 或原始数据位置。"
  exit 1
fi

echo "[4/5] 开始最小训练"
python train.py \
  --data_dir "$DATA_DIR" \
  --name "$RUN_NAME" \
  --batchsize "$TRAIN_BATCHSIZE" \
  --workers 0 \
  --total_epoch "$TOTAL_EPOCH" \
  --erasing_p 0

echo "[5/5] 开始测试"
python test.py \
  --test_dir "$DATA_DIR" \
  --name "$RUN_NAME" \
  --batchsize "$TEST_BATCHSIZE" \
  --workers 0

echo "Smoke test 完成。结果目录: ./model/$RUN_NAME"
