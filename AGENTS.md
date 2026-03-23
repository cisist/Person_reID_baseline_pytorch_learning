# AGENTS.md

Codex 在本仓库默认以“优先适配 `macm4` 本机环境，并先稳定跑通训练/测试链路”为目标工作。

## 当前分支与合并背景

- 当前工作分支：`macm4`
- 已执行 `origin/br_01 -> macm4` 合并
- 本次合并结果为 `Already up to date.`，说明 `origin/br_01` 与本地 `macm4` 在已提交内容上没有差异
- 已执行 `remotes/origin/br_01 -> macm4` 合并确认
- 本次合并结果同样为 `Already up to date.`，说明 `remotes/origin/br_01` 与本地 `macm4` 在已提交内容上没有差异
- 合并前的 `AGENTS.md` 已备份为：`AGENTS.md.macm4.premerge.bak`
- 本轮针对 `origin/br_01` 的额外备份为：`AGENTS.md.origin_br01.premerge.bak`
- 本轮针对 `remotes/origin/br_01` 的额外备份为：`AGENTS.md.remotes_origin_br01.premerge.bak`

## 仓库定位

- 这是一个 PyTorch ReID 基线仓库，偏研究代码风格
- 不是工程化打包项目，默认从仓库根目录启动
- 大量脚本依赖相对路径
- 训练、测试、评估由多个脚本串联完成

核心脚本：

- `prepare.py`
- `train.py`
- `train_DDP.py`
- `test.py`
- `evaluate.py`
- `evaluate_gpu.py`
- `demo.py`
- `model.py`

## macm4 本地状态

- 机器环境：`macm4`
- 优先设备：`mps`
- 备选设备：`cpu`
- 不默认假设存在 CUDA

本地数据状态：

- 原始数据：`./data/Market-1501-v15.09.15`
- 已整理数据：`./data/Market/pytorch`
- 已存在目录：
  - `train`
  - `val`
  - `train_all`
  - `query`
  - `gallery`
  - `multi-query`

本地代码状态：

- `prepare.py` 已适配仓库内数据路径
- `train.py`、`test.py`、`utils.py`、`ODFA.py` 有本地兼容性修改
- 这些修改的目标是降低 CUDA 硬依赖，提升 macOS / Apple Silicon 可运行性
- 未经用户明确要求，不要回退这些本地适配

## macm4 默认训练参数

在 `macm4` 上，默认训练参数以“稳定优先”为原则：

- `batchsize=8`
- `workers=0`
- `erasing_p=0` 可用于 smoke test

除非用户明确要求做吞吐基准或速度实验，否则默认按 `batchsize=8` 给出训练命令与建议。

推荐的最小训练命令：

```bash
python train.py --data_dir ./data/Market/pytorch --name mac_smoke --batchsize 8 --workers 0 --total_epoch 1 --erasing_p 0
```

推荐的最小测试命令：

```bash
python test.py --test_dir ./data/Market/pytorch --name mac_smoke --batchsize 64 --workers 0
```

## 本机优先路径

如果用户在 `macm4` 上要求“先跑通”或“先验证流程”，默认顺序是：

1. 检查 Python / conda 环境
2. 检查 `torch` 与关键依赖能否导入
3. 检查 `./data/Market/pytorch` 是否完整
4. 先跑 `train.py`
5. 再跑 `test.py`
6. 最后查看 `./model/<run_name>/`

以下入口不作为 `macm4` 首选：

- `DDP.sh`
- `train_DDP.py`
- `evaluate_gpu.py`
- `test_with_TensorRT.py`

## macOS / Apple Silicon 约定

- 优先设备无关逻辑
- 避免新增无判断的 `.cuda()`
- 新逻辑尽量兼容：
  - `cuda`
  - `mps`
  - `cpu`
- 必须引入 GPU-only 逻辑时，需要先说明影响范围
- smoke test 优先 `train.py`、`test.py`
- 非 CUDA 环境优先 `evaluate.py`

## 修改与验证原则

- 优先兼容性修复和流程跑通
- 尽量做局部补丁，保留原始研究代码结构
- 为适配本机环境时优先使用向后兼容写法
- 改路径时保留原默认行为
- 声称修复前尽量完成语法检查
- 条件允许时完成最小 smoke test
- 若执行受阻，要区分代码问题与环境问题
- 不要把缺依赖、conda 异常、预训练权重下载失败直接归因于代码错误

## 常见坑位

- 默认路径常写成 `../Market/pytorch`
- 测试脚本可能默认走 GPU 评估
- backbone 预训练权重可能需要联网下载
- 旧版 `DDP.sh` 使用 `python -m torch.distributed.launch ... train_DDP.py`
- 这类旧 GPU / Linux 优先入口不适合作为 `macm4` 的第一验证路径

## 默认排障顺序

当用户说“分析报错”或“跑通训练测试”时，默认：

1. 先判断问题在环境、数据、训练还是测试
2. 再检查对应脚本的硬编码路径和设备依赖
3. 优先修正最小阻塞点
4. 先追求跑通，再考虑清理和优化

如果无法在沙箱内真实运行，也要尽量给出：

- 根因判断
- 修改建议
- 用户本机可直接执行的下一步命令
