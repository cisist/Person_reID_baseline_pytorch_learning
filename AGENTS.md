# AGENTS.md

Codex 在本仓库默认以“优先适配 `macm4` 本机环境，并先稳定跑通训练/测试链路”为目标工作。

## 当前目标

当前机器是 `macm4`。首要目标分为两层：

1. 在 `macm4` 上稳定跑通训练、测试、评估链路。
2. 参考 `README.md` 中 `Trained Model` 表格逐项复现实验，但执行顺序、资源策略和排障方式需要改造成适配 `macm4`。

这里的原则是：

- `README Trained Model` 是实验目标清单
- `macm4 推荐执行顺序` 只负责回答先跑谁、怎么排障、如何在本机资源约束下推进

## 仓库结构

这是一个以脚本为主的 PyTorch ReID 基线仓库。

核心脚本：

- `prepare.py`
- `train.py`
- `train_DDP.py`
- `test.py`
- `evaluate.py`
- `evaluate_gpu.py`
- `demo.py`
- `model.py`

相关模块：

- `utils.py`
- `random_erasing.py`
- `circle_loss.py`
- `instance_loss.py`
- `ODFA.py`

数据准备脚本采用 `prepare_*.py` 命名。训练输出保存在 `model/<run_name>/`。文档位于 `docs/`、`tutorial/`、`colab/`、`leaderboard*/`，重排序实现位于 `GPU-Re-Ranking/`。

## macm4 本地状态

- 当前工作分支：`macm4`
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

## macm4 脚本入口

仓库已经补充了一批 `tool/macm4_day*.sh` 脚本，用来把 `README Trained Model` 中的典型实验改造成适配 `macm4` 的执行入口。

当前已存在：

- `tool/macm4_day2_resnet50.sh`
- `tool/macm4_day3_resnet50_ibn.sh`
- `tool/macm4_day4_densenet121.sh`
- `tool/macm4_day5_densenet121_circle.sh`
- `tool/macm4_day6_resnet50_usam.sh`
- `tool/macm4_day7_resnet50_ibn_usam.sh`
- `tool/macm4_day8_efficientnet_b4.sh`
- `tool/macm4_day9_convnext.sh`
- `tool/macm4_day10_hrnet18.sh`
- `tool/macm4_day11_resnet50_adv.sh`
- `tool/macm4_day12_pcb.sh`
- `tool/macm4_day13_pcb_dg.sh`
- `tool/macm4_day14_swin.sh`
- `tool/macm4_day15_swinv2.sh`
- `tool/macm4_day16_dinov3.sh`
- `tool/macm4_day17_resnet50_all_tricks.sh`
- `tool/macm4_day18_resnet50_all_tricks_circle.sh`
- `tool/macm4_day19_resnet50_all_tricks_circle_dg.sh`
- `tool/macm4_day20_densenet_all_tricks_circle.sh`
- `tool/macm4_day21_hrnet_all_tricks_circle_dg.sh`
- `tool/macm4_day22_swin_all_tricks_circle.sh`
- `tool/macm4_day23_swin_all_tricks_circle_b16.sh`
- `tool/macm4_day24_swin_all_tricks_circle_b16_dg.sh`

这些脚本的共同约定：

- 默认数据目录是 `./data/Market/pytorch`
- 除特别说明外，默认训练参数是 `TRAIN_BATCH=8`
- 默认测试参数通常是 `TEST_BATCH=32` 或 `64`
- 默认 dataloader 参数是 `WORKERS=0`
- 默认只跑 smoke test，不会直接启动正式长训
- 默认训练后不会自动测试，除非显式打开开关

特殊说明：

- `tool/macm4_day2_resnet50.sh` 为了更贴近原始 ResNet-50 基线，默认使用 `TRAIN_BATCH=32`

推荐运行方式：

```bash
tool/macm4_day8_efficientnet_b4.sh
tool/macm4_day14_swin.sh
tool/macm4_day17_resnet50_all_tricks.sh
```

如果要正式训练：

```bash
RUN_FORMAL_TRAIN=1 tool/macm4_day8_efficientnet_b4.sh
```

如果要训练后自动测试和评估：

```bash
RUN_FORMAL_TRAIN=1 RUN_TEST_AFTER_TRAIN=1 tool/macm4_day8_efficientnet_b4.sh
```

如果要临时覆盖默认参数：

```bash
TRAIN_BATCH=8 TEST_BATCH=32 WORKERS=0 RUN_FORMAL_TRAIN=1 tool/macm4_day14_swin.sh
```

环境前提：

- 必须先进入装好依赖的 conda 环境
- 若脚本输出的 `python=` 指向 `base` 环境，优先判断为环境问题，不要先怀疑脚本本身
- 推荐先执行：

```bash
conda activate codex_reid
which python
python -c "import torch, timm, yaml, scipy; print(torch.__version__)"
```

日志位置：

- 每个脚本都会把日志写到 `logs/macm4_day*/`
- 排查时优先看 `*_env.log`、`*_smoke.log`、`*_train.log`、`*_test.log`

使用原则：

- 先用脚本做 smoke test，确认模型入口和依赖正常
- smoke test 通过后再打开 `RUN_FORMAL_TRAIN=1`
- `DG` 类脚本需要额外确认 DG-Market 数据是否已准备
- `SwinV2`、`DinoV3`、`PCB` 这类高成本实验不作为本机首个验证入口

## 模型名到推荐脚本映射

为了减少“README 模型名”和“macm4 实际入口脚本”之间的来回查找，默认按下表对应：

| README 模型名 | 推荐脚本 |
| ---- | ---- |
| ResNet-50 | `tool/macm4_day2_resnet50.sh`（默认 `TRAIN_BATCH=32`） |
| ResNet-50-ibn | `tool/macm4_day3_resnet50_ibn.sh` |
| DenseNet-121 | `tool/macm4_day4_densenet121.sh` |
| DenseNet-121 (Circle) | `tool/macm4_day5_densenet121_circle.sh` |
| ResNet-50 + USAM | `tool/macm4_day6_resnet50_usam.sh` |
| ResNet-50-ibn + USAM | `tool/macm4_day7_resnet50_ibn_usam.sh` |
| EfficientNet-b4 | `tool/macm4_day8_efficientnet_b4.sh` |
| ConvNeXt | `tool/macm4_day9_convnext.sh` |
| HRNet-18 | `tool/macm4_day10_hrnet18.sh` |
| ResNet-50 + adv defense | `tool/macm4_day11_resnet50_adv.sh` |
| PCB | `tool/macm4_day12_pcb.sh` |
| PCB + DG | `tool/macm4_day13_pcb_dg.sh` |
| Swin (224x224) | `tool/macm4_day14_swin.sh` |
| SwinV2 (all tricks+Circle 256x128) | `tool/macm4_day15_swinv2.sh` |
| DinoV3-base (all tricks+Circle 256x128) | `tool/macm4_day16_dinov3.sh` |
| ResNet-50 (all tricks) | `tool/macm4_day17_resnet50_all_tricks.sh` |
| ResNet-50 (all tricks+Circle) | `tool/macm4_day18_resnet50_all_tricks_circle.sh` |
| ResNet-50 (all tricks+Circle+DG) | `tool/macm4_day19_resnet50_all_tricks_circle_dg.sh` |
| DenseNet-121 (all tricks+Circle) | `tool/macm4_day20_densenet_all_tricks_circle.sh` |
| HRNet-18 (all tricks+Circle+DG) | `tool/macm4_day21_hrnet_all_tricks_circle_dg.sh` |
| Swin (all tricks+Circle 224x224) | `tool/macm4_day22_swin_all_tricks_circle.sh` |
| Swin (all tricks+Circle+b16 224x224) | `tool/macm4_day23_swin_all_tricks_circle_b16.sh` |
| Swin (all tricks+Circle+b16+DG 224x224) | `tool/macm4_day24_swin_all_tricks_circle_b16_dg.sh` |

补充说明：

- `ResNet-50 (fp16)` 当前没有单独的 `macm4_day*.sh`，因为 `macm4` 默认不以 CUDA/AMP 为首选验证路径
- 若需要在 `macm4` 上尝试 `fp16` 或 `bf16`，应先确认当前设备和 `torch` 后端是否真正支持并稳定
- `README` 中部分实验名和脚本默认 `TRAIN_NAME` 不完全一致时，以“脚本可执行、结果目录可区分”为优先

## 推荐执行顺序追踪表

这张表用于把“先跑谁”“用哪个脚本跑”“当前推进到哪一步”统一记录下来。

状态约定：

- `未开始`：还没有执行对应脚本
- `已验证`：至少完成过一次 smoke test，或已确认训练/测试链路正常
- `阻塞`：当前被环境、数据、依赖、脚本错误或资源问题卡住

备注约定：

- 优先记录“最近一次有效动作”的结果
- 若失败，尽量写清楚失败阶段和核心报错
- 若成功，至少写明是 smoke test 通过还是训练/测试全链路通过

当前默认状态按最近已知进展初始化；后续每推进一次实验，都应更新这张表，而不是只在对话里口头说明。

| 顺序 | README 模型名 | 推荐脚本 | 当前状态 | 最近一次结果/备注 |
| ---- | ---- | ---- | ---- | ---- |
| 1 | ResNet-50 | `tool/macm4_day2_resnet50.sh` | 已验证 | 最小链路已在 `macm4` 跑通；当前脚本默认 `TRAIN_BATCH=32` 以贴近原始基线 |
| 2 | ResNet-50-ibn | `tool/macm4_day3_resnet50_ibn.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 3 | DenseNet-121 | `tool/macm4_day4_densenet121.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 4 | DenseNet-121 (Circle) | `tool/macm4_day5_densenet121_circle.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 5 | ResNet-50 + USAM | `tool/macm4_day6_resnet50_usam.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 6 | ResNet-50-ibn + USAM | `tool/macm4_day7_resnet50_ibn_usam.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 7 | EfficientNet-b4 | `tool/macm4_day8_efficientnet_b4.sh` | 阻塞 | 最近一次 smoke test 失败，`python` 落到 `base`，报 `No module named 'torch'` |
| 8 | ConvNeXt | `tool/macm4_day9_convnext.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 9 | HRNet-18 | `tool/macm4_day10_hrnet18.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 10 | ResNet-50 + adv defense | `tool/macm4_day11_resnet50_adv.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 11 | PCB | `tool/macm4_day12_pcb.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 12 | PCB + DG | `tool/macm4_day13_pcb_dg.sh` | 未开始 | 脚本已创建，且后续还需确认 DG-Market 数据 |
| 13 | ResNet-50 (all tricks) | `tool/macm4_day17_resnet50_all_tricks.sh` | 阻塞 | 最近一次脚本检查阶段失败，`import torch` 即报错，根因仍是环境落到 `base` |
| 14 | ResNet-50 (all tricks+Circle) | `tool/macm4_day18_resnet50_all_tricks_circle.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 15 | ResNet-50 (all tricks+Circle+DG) | `tool/macm4_day19_resnet50_all_tricks_circle_dg.sh` | 未开始 | 脚本已创建，且后续还需确认 DG-Market 数据 |
| 16 | DenseNet-121 (all tricks+Circle) | `tool/macm4_day20_densenet_all_tricks_circle.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 17 | HRNet-18 (all tricks+Circle+DG) | `tool/macm4_day21_hrnet_all_tricks_circle_dg.sh` | 未开始 | 脚本已创建，且后续还需确认 DG-Market 数据 |
| 18 | Swin (224x224) | `tool/macm4_day14_swin.sh` | 阻塞 | 最近一次 smoke test 失败，`python` 落到 `base`，报 `No module named 'torch'` |
| 19 | SwinV2 (all tricks+Circle 256x128) | `tool/macm4_day15_swinv2.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 20 | DinoV3-base (all tricks+Circle 256x128) | `tool/macm4_day16_dinov3.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 21 | Swin (all tricks+Circle 224x224) | `tool/macm4_day22_swin_all_tricks_circle.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 22 | Swin (all tricks+Circle+b16 224x224) | `tool/macm4_day23_swin_all_tricks_circle_b16.sh` | 未开始 | 脚本已创建，尚未执行 smoke test |
| 23 | Swin (all tricks+Circle+b16+DG 224x224) | `tool/macm4_day24_swin_all_tricks_circle_b16_dg.sh` | 未开始 | 脚本已创建，且后续还需确认 DG-Market 数据 |

当前阻塞说明：

- `EfficientNet-b4`、`ResNet-50 (all tricks)`、`Swin (224x224)` 最近一次脚本验证都失败在同一类问题
- 根因不是脚本参数本身，而是运行时落到了 `base` Python 环境
- 已观测到的典型错误是 `ModuleNotFoundError: No module named 'torch'`
- 这类项在切换到正确 conda 环境并重新验证前，统一标记为 `阻塞`

## README Trained Model 复现清单

默认应把下列条目都视为正式复现目标。若某条目暂缓执行，也需要明确标记为“未开始 / 阻塞 / 延后”，而不是省略。

| 方法 | README 参考命令 |
| ---- | ---- |
| EfficientNet-b4 | `python train.py --use_efficient --name eff; python test.py --name eff` |
| ResNet-50 + adv defense | `python train.py --name adv0.1_40_w10_all --adv 0.1 --aiter 40 --warm 10 --train_all; python test.py --name adv0.1_40_w10_all` |
| ConvNeXt | `python train.py --use_convnext --name convnext; python test.py --name convnext` |
| ResNet-50 (fp16) | `python train.py --name fp16 --fp16 --train_all` |
| ResNet-50 | `python train.py --train_all` |
| ResNet-50 + USAM | `python train.py --train_all --name res-usam --usam; python test.py --name res-usam` |
| ResNet-50-ibn | `python train.py --train_all --name res-ibn --ibn` |
| ResNet-50-ibn + USAM | `python train.py --train_all --name ibn-usam --ibn --usam; python test.py --name ibn-usam` |
| DenseNet-121 | `python train.py --name ft_net_dense --use_dense --train_all` |
| DenseNet-121 (Circle) | `python train.py --name ft_net_dense_circle_w5 --circle --use_dense --train_all --warm_epoch 5` |
| HRNet-18 | `python train.py --use_hr --name hr18; python test.py --name hr18` |
| PCB | `python train.py --name PCB --PCB --train_all --lr 0.02` |
| PCB + DG | `python train.py --name PCB_DG --PCB --train_all --lr 0.02 --DG; python test.py --name PCB_DG` |
| ResNet-50 (all tricks) | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name warm5_s1_b8_lr2_p0.5` |
| ResNet-50 (all tricks+Circle) | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name warm5_s1_b8_lr2_p0.5_circle --circle` |
| ResNet-50 (all tricks+Circle+DG) | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name warm5_s1_b8_lr2_p0.5_circle_DG --circle --DG; python test.py --name warm5_s1_b8_lr2_p0.5_circle_DG` |
| DenseNet-121 (all tricks+Circle) | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name dense_warm5_s1_b8_lr2_p0.5_circle --circle --use_dense; python test.py --name dense_warm5_s1_b8_lr2_p0.5_circle` |
| HRNet-18 (all tricks+Circle+DG) | `python train.py --use_hr --name hr18_p0.5_circle_w5_b16_lr0.01_DG --lr 0.01 --batchsize 16 --DG --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name hr18_p0.5_circle_w5_b16_lr0.01_DG` |
| Swin (224x224) | `python train.py --use_swin --name swin; python test.py --name swin` |
| SwinV2 (all tricks+Circle 256x128) | `python train.py --use_swinv2 --name swinv2_p0.5_circle_w5_b16_lr0.03 --lr 0.03 --batchsize 16 --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name swinv2_p0.5_circle_w5_b16_lr0.03 --batchsize 32` |
| DinoV3-base (all tricks+Circle 256x128) | `python train.py --use_dino --name dino_p0.5_circle_w5_b32_lr0.04 --lr 0.04 --batchsize 32 --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name dino_p0.5_circle_w5_b32_lr0.04 --batchsize 32` |
| Swin (all tricks+Circle 224x224) | `python train.py --use_swin --name swin_p0.5_circle_w5 --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name swin_p0.5_circle_w5` |
| Swin (all tricks+Circle+b16 224x224) | `python train.py --use_swin --name swin_p0.5_circle_w5_b16_lr0.01 --lr 0.01 --batchsize 16 --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name swin_p0.5_circle_w5_b16_lr0.01` |
| Swin (all tricks+Circle+b16+DG 224x224) | `python train.py --use_swin --name swin_p0.5_circle_w5_b16_lr0.01_DG --lr 0.01 --batchsize 16 --DG --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name swin_p0.5_circle_w5_b16_lr0.01_DG` |

说明：若 README 原文与当前脚本参数名存在小差异，应以“仓库当前脚本可实际执行”为准，并在报告中记录偏差。

## macm4 推荐执行顺序

这一层只负责回答：

- 先跑谁，后跑谁
- 哪些模型适合先验证链路，哪些模型适合后续冲指标
- 本机资源、稳定性或兼容性出问题时优先怎么调整

### A 组：基础链路与基准组

优先用于确认数据、训练、测试、评估全链路正常。

- `ResNet-50`
- `ResNet-50-ibn`
- `DenseNet-121`
- `DenseNet-121 (Circle)`

说明：`README` 中的 `ResNet-50 (fp16)` 在 `macm4` 上不作为首选链路验证项，因为本机默认不以 CUDA 为前提。

### B 组：轻中度扩展组

用于确认额外模块或不同 backbone 在 `macm4` 上是否稳定。

- `ResNet-50 + USAM`
- `ResNet-50-ibn + USAM`
- `EfficientNet-b4`
- `ConvNeXt`
- `HRNet-18`
- `ResNet-50 + adv defense`

### C 组：高成本结构组

用于验证更复杂结构和更大资源占用的配置。

- `PCB`
- `PCB + DG`
- `Swin (224x224)`
- `SwinV2 (all tricks+Circle 256x128)`
- `DinoV3-base (all tricks+Circle 256x128)`

### D 组：冲榜技巧组

用于逼近 README 高分结果，通常更依赖 batch size、学习率和训练稳定性。

- `ResNet-50 (all tricks)`
- `ResNet-50 (all tricks+Circle)`
- `ResNet-50 (all tricks+Circle+DG)`
- `DenseNet-121 (all tricks+Circle)`
- `HRNet-18 (all tricks+Circle+DG)`
- `Swin (all tricks+Circle 224x224)`
- `Swin (all tricks+Circle+b16 224x224)`
- `Swin (all tricks+Circle+b16+DG 224x224)`

### macm4 默认推进顺序

推荐按以下顺序推进，前一项未闭环前不要进入下一项：

1. `ResNet-50`
2. `ResNet-50-ibn`
3. `DenseNet-121`
4. `DenseNet-121 (Circle)`
5. `ResNet-50 + USAM`
6. `ResNet-50-ibn + USAM`
7. `EfficientNet-b4`
8. `ConvNeXt`
9. `HRNet-18`
10. `ResNet-50 + adv defense`
11. `PCB`
12. `PCB + DG`
13. `ResNet-50 (all tricks)`
14. `ResNet-50 (all tricks+Circle)`
15. `ResNet-50 (all tricks+Circle+DG)`
16. `DenseNet-121 (all tricks+Circle)`
17. `HRNet-18 (all tricks+Circle+DG)`
18. `Swin (224x224)`
19. `SwinV2 (all tricks+Circle 256x128)`
20. `DinoV3-base (all tricks+Circle 256x128)`
21. `Swin (all tricks+Circle 224x224)`
22. `Swin (all tricks+Circle+b16 224x224)`
23. `Swin (all tricks+Circle+b16+DG 224x224)`

## macm4 资源策略

- 若目标是尽快确认链路，优先跑 A 组
- 若目标是扩大覆盖面，先完成 A 组，再推进 B 组
- 若目标是逼近 README 高指标，必须在 A/B 组稳定后再进入 C/D 组
- `macm4` 默认训练 `batchsize=8`
- 若显存或统一内存不足，先减小 `--batchsize`
- 若训练速度过慢，优先减少 dataloader 并发，默认保持 `workers=0`
- 不要在 `macm4` 上优先依赖 CUDA-only 技巧来解决问题
- 若同一模型连续两次因同类问题失败，先转为排障，不继续堆新实验

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

## 执行与记录要求

每个模型条目都至少记录以下内容：

1. README 行名
2. README 原命令
3. 实际执行命令
4. 是否偏离 README
5. 训练 batch size、测试 batch size、精度模式
6. Rank@1、mAP、耗时、稳定性
7. 若失败，失败原因与下一步排查项

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
