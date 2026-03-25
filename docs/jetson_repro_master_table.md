# Jetson Orin 64G 复现总表

## 说明
本文档用于把以下三类信息统一对齐：
- `README.md` 中 `Trained Model` 表格的模型条目
- `AGENTS.md` 中定义的 Jetson Orin 64G 分组与执行顺序
- `tool/` 目录下已经创建的 `day1-day24` 执行脚本

使用方式：
- 如果你想知道某个 README 模型该跑哪个脚本，先查第 1 张表。
- 如果你想按 Jetson 推荐顺序推进，查第 2 张表。
- 如果你想看某一行是否已经有脚本覆盖，查“脚本状态”列。

---

## 1. README Trained Model 对应表
| 序号 | README 模型名 | README 参考命令摘要 | 分组 | Jetson 天数 | 对应脚本 | 脚本状态 | 备注 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1 | EfficientNet-b4 | `--use_efficient --name eff` | B | Day 8 | [tool/jetson_day8_efficientnet_b4.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day8_efficientnet_b4.sh) | 已覆盖 | |
| 2 | ResNet-50 + adv defense | `--adv 0.1 --aiter 40 --warm 10 --train_all` | B | Day 11 | [tool/jetson_day11_resnet50_adv.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day11_resnet50_adv.sh) | 已覆盖 | 实际脚本使用 `--warm_epoch 10` |
| 3 | ConvNeXt | `--use_convnext --name convnext` | B | Day 9 | [tool/jetson_day9_convnext.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day9_convnext.sh) | 已覆盖 | |
| 4 | ResNet-50 (fp16) | `--name fp16 --fp16 --train_all` | A | Day 1 | [tool/jetson_day1_fp16.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day1_fp16.sh) | 已覆盖 | |
| 5 | ResNet-50 | `--train_all` | A | Day 2 | [tool/jetson_day2_resnet50.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day2_resnet50.sh) | 已覆盖 | |
| 6 | ResNet-50 + USAM | `--train_all --name res-usam --usam` | B | Day 6 | [tool/jetson_day6_resnet50_usam.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day6_resnet50_usam.sh) | 已覆盖 | |
| 7 | ResNet-50-ibn | `--train_all --name res-ibn --ibn` | A | Day 3 | [tool/jetson_day3_resnet50_ibn.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day3_resnet50_ibn.sh) | 已覆盖 | |
| 8 | ResNet-50-ibn + USAM | `--train_all --name ibn-usam --ibn --usam` | B | Day 7 | [tool/jetson_day7_resnet50_ibn_usam.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day7_resnet50_ibn_usam.sh) | 已覆盖 | |
| 9 | DenseNet-121 | `--name ft_net_dense --use_dense --train_all` | A | Day 4 | [tool/jetson_day4_densenet121.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day4_densenet121.sh) | 已覆盖 | |
| 10 | DenseNet-121 (Circle) | `--circle --use_dense --train_all --warm_epoch 5` | A | Day 5 | [tool/jetson_day5_densenet121_circle.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day5_densenet121_circle.sh) | 已覆盖 | |
| 11 | HRNet-18 | `--use_hr --name hr18` | B | Day 10 | [tool/jetson_day10_hrnet18.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day10_hrnet18.sh) | 已覆盖 | |
| 12 | PCB | `--name PCB --PCB --train_all --lr 0.02` | C | Day 12 | [tool/jetson_day12_pcb.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day12_pcb.sh) | 已覆盖 | |
| 13 | PCB + DG | `--name PCB_DG --PCB --train_all --lr 0.02 --DG` | C | Day 13 | [tool/jetson_day13_pcb_dg.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day13_pcb_dg.sh) | 已覆盖 | 依赖 DG-Market |
| 14 | ResNet-50 (all tricks) | `warm_epoch 5 stride 1 erasing_p 0.5 batchsize 8 lr 0.02` | D | Day 17 | [tool/jetson_day17_resnet50_all_tricks.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day17_resnet50_all_tricks.sh) | 已覆盖 | |
| 15 | ResNet-50 (all tricks+Circle) | `... --circle` | D | Day 18 | [tool/jetson_day18_resnet50_all_tricks_circle.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day18_resnet50_all_tricks_circle.sh) | 已覆盖 | |
| 16 | ResNet-50 (all tricks+Circle+DG) | `... --circle --DG` | D | Day 19 | [tool/jetson_day19_resnet50_all_tricks_circle_dg.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day19_resnet50_all_tricks_circle_dg.sh) | 已覆盖 | 依赖 DG-Market |
| 17 | DenseNet-121 (all tricks+Circle) | `... --circle --use_dense` | D | Day 20 | [tool/jetson_day20_densenet_all_tricks_circle.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day20_densenet_all_tricks_circle.sh) | 已覆盖 | |
| 18 | HRNet-18 (all tricks+Circle+DG) | `--use_hr ... --DG --circle --warm_epoch 5` | D | Day 21 | [tool/jetson_day21_hrnet_all_tricks_circle_dg.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day21_hrnet_all_tricks_circle_dg.sh) | 已覆盖 | 依赖 DG-Market |
| 19 | Swin (224x224) | `--use_swin --name swin` | C | Day 14 | [tool/jetson_day14_swin.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day14_swin.sh) | 已覆盖 | |
| 20 | SwinV2 (all tricks+Circle 256x128) | `--use_swinv2 ... --circle --warm_epoch 5` | C | Day 15 | [tool/jetson_day15_swinv2.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day15_swinv2.sh) | 已覆盖 | |
| 21 | DinoV3-base (all tricks+Circle 256x128) | `--use_dino ... --circle --warm_epoch 5` | C | Day 16 | [tool/jetson_day16_dinov3.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day16_dinov3.sh) | 已覆盖 | |
| 22 | Swin (all tricks+Circle 224x224) | `--use_swin ... --circle --warm_epoch 5` | D | Day 22 | [tool/jetson_day22_swin_all_tricks_circle.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day22_swin_all_tricks_circle.sh) | 已覆盖 | |
| 23 | Swin (all tricks+Circle+b16 224x224) | `--use_swin --lr 0.01 --batchsize 16 ...` | D | Day 23 | [tool/jetson_day23_swin_all_tricks_circle_b16.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day23_swin_all_tricks_circle_b16.sh) | 已覆盖 | |
| 24 | Swin (all tricks+Circle+b16+DG 224x224) | `--use_swin --lr 0.01 --batchsize 16 --DG ...` | D | Day 24 | [tool/jetson_day24_swin_all_tricks_circle_b16_dg.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_day24_swin_all_tricks_circle_b16_dg.sh) | 已覆盖 | 依赖 DG-Market |

---

## 2. Jetson 推荐顺序总表
| 分组 | 说明 | 覆盖天数 | 模型 |
| ---- | ---- | ---- | ---- |
| A 组 | 基础链路与基准组 | Day 1-5 | `ResNet-50(fp16)`、`ResNet-50`、`ResNet-50-ibn`、`DenseNet-121`、`DenseNet-121(Circle)` |
| B 组 | 轻中度扩展组 | Day 6-11 | `ResNet-50 + USAM`、`ResNet-50-ibn + USAM`、`EfficientNet-b4`、`ConvNeXt`、`HRNet-18`、`ResNet-50 + adv defense` |
| C 组 | 高成本结构组 | Day 12-16 | `PCB`、`PCB + DG`、`Swin`、`SwinV2`、`DinoV3` |
| D 组 | 冲榜技巧组 | Day 17-24 | `ResNet-50 all tricks`、`DenseNet all tricks`、`HRNet all tricks`、`Swin all tricks` |

---

## 3. 总控脚本对应关系
总控脚本：
[tool/jetson_run_all.sh](/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning/tool/jetson_run_all.sh)

常见用法：
```bash
bash tool/jetson_run_all.sh
```

```bash
START_DAY=1 END_DAY=7 RUN_FORMAL_TRAIN=1 bash tool/jetson_run_all.sh
```

```bash
START_DAY=12 END_DAY=24 RUN_FORMAL_TRAIN=1 RUN_TEST_AFTER_TRAIN=1 bash tool/jetson_run_all.sh
```

```bash
START_DAY=1 END_DAY=24 RUN_FORMAL_TRAIN=1 STOP_ON_ERROR=0 bash tool/jetson_run_all.sh
```

---

## 4. 特殊说明
- 带 `DG` 的脚本依赖 `DG-Market` 数据已经准备完成。
- `ResNet-50 + adv defense` 的 README 写法中出现 `--warm 10`，当前仓库脚本实际使用的是 `--warm_epoch 10`，脚本已按可执行参数处理。
- `Swin / SwinV2 / DinoV3` 在 Jetson 上属于高成本模型，脚本默认使用了更保守的 batch size。
- 测试阶段通常会从 `model/<name>/opts.yaml` 中恢复模型结构配置，因此部分脚本测试命令比训练命令更短，这是当前仓库的正常用法。

---

## 5. 建议推进方式
- 首次跑通：先执行 A 组，再决定是否进入 B 组。
- 想快速扩大覆盖：按 `A -> B -> C` 推进。
- 想逼近 README 高分结果：必须在 `A/B/C` 已稳定后再进入 D 组。
- 若出现连续两次同类失败，暂停推进新脚本，先整理问题日志和环境差异。
