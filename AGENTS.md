# Repository Guidelines

## 当前目标
这台机器是 Jetson Orin 64G，当前工作的首要目标是复现 `README.md` 中 `Trained Model` 表格里的结果。执行时分为两层：

1. `README Trained Model` 逐项复现总表：以 README 表格中的模型、命令和结果口径为主基准。
2. `Jetson Orin 64G 推荐执行顺序`：仅用于确定优先级、资源策略和排障顺序，不替代 README 原表。

## 项目结构
该仓库是一个以脚本为主的 PyTorch ReID 基线工程。训练与推理入口位于根目录，包括 `train.py`、`train_DDP.py`、`test.py`、`evaluate.py` 和 `demo.py`。主干网络定义集中在 `model.py`；通用工具位于 `utils.py`、`random_erasing.py`、`circle_loss.py`、`instance_loss.py`。数据准备脚本采用 `prepare_*.py` 命名。训练输出保存在 `model/<run_name>/`，文档位于 `docs/`、`tutorial/`、`colab/`、`leaderboard*/`，重排序实现位于 `GPU-Re-Ranking/`。

## 第一层：README Trained Model 逐项复现总表
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

## 第二层：Jetson Orin 64G 推荐执行顺序
这一层只负责回答三个问题：
- 先跑谁，后跑谁
- 哪些模型适合用来验证链路，哪些模型适合最后冲指标
- 显存、时长和稳定性出问题时优先怎么调整

### 2.1 推荐优先级分组
#### A 组：基础链路与基准组
优先用于确认数据、训练、测试、评估全链路正常。
- `ResNet-50 (fp16)`
- `ResNet-50`
- `ResNet-50-ibn`
- `DenseNet-121`
- `DenseNet-121 (Circle)`

#### B 组：轻中度扩展组
用于确认额外模块或不同 backbone 在 Jetson 上是否稳定。
- `ResNet-50 + USAM`
- `ResNet-50-ibn + USAM`
- `EfficientNet-b4`
- `ConvNeXt`
- `HRNet-18`
- `ResNet-50 + adv defense`

#### C 组：高成本结构组
用于验证更复杂结构和更大资源占用的配置。
- `PCB`
- `PCB + DG`
- `Swin (224x224)`
- `SwinV2 (all tricks+Circle 256x128)`
- `DinoV3-base (all tricks+Circle 256x128)`

#### D 组：冲榜技巧组
用于逼近 README 高分结果，通常更依赖 batch size、学习率和训练稳定性。
- `ResNet-50 (all tricks)`
- `ResNet-50 (all tricks+Circle)`
- `ResNet-50 (all tricks+Circle+DG)`
- `DenseNet-121 (all tricks+Circle)`
- `HRNet-18 (all tricks+Circle+DG)`
- `Swin (all tricks+Circle 224x224)`
- `Swin (all tricks+Circle+b16 224x224)`
- `Swin (all tricks+Circle+b16+DG 224x224)`

### 2.2 Jetson 上的默认执行顺序
推荐按以下顺序推进，前一项未闭环前不要进入下一项：
1. `ResNet-50 (fp16)`
2. `ResNet-50`
3. `ResNet-50-ibn`
4. `DenseNet-121`
5. `DenseNet-121 (Circle)`
6. `ResNet-50 + USAM`
7. `ResNet-50-ibn + USAM`
8. `EfficientNet-b4`
9. `ConvNeXt`
10. `HRNet-18`
11. `ResNet-50 + adv defense`
12. `PCB`
13. `PCB + DG`
14. `ResNet-50 (all tricks)`
15. `ResNet-50 (all tricks+Circle)`
16. `ResNet-50 (all tricks+Circle+DG)`
17. `DenseNet-121 (all tricks+Circle)`
18. `HRNet-18 (all tricks+Circle+DG)`
19. `Swin (224x224)`
20. `SwinV2 (all tricks+Circle 256x128)`
21. `DinoV3-base (all tricks+Circle 256x128)`
22. `Swin (all tricks+Circle 224x224)`
23. `Swin (all tricks+Circle+b16 224x224)`
24. `Swin (all tricks+Circle+b16+DG 224x224)`

### 2.3 资源策略
- 若目标是尽快确认链路，优先跑 A 组。
- 若目标是扩大覆盖面，先完成 A 组，再推进 B 组。
- 若目标是逼近 README 高指标，必须在 A/B 组稳定后再进入 C/D 组。
- 若显存不足，先减小 `--batchsize`，再考虑 `--fp16` / `--bf16`，最后才考虑延后高成本模型。
- 若训练速度过慢，优先减少 dataloader 并发和测试 batch size，不优先改评估逻辑。
- 若同一模型连续两次因同类问题失败，先转为排障，不继续堆新实验。

## 执行与记录要求
每个模型条目都至少记录以下内容：
1. README 行名
2. README 原命令
3. 实际执行命令
4. 是否偏离 README
5. 训练 batch size、测试 batch size、精度模式
6. Rank@1、mAP、耗时、稳定性
7. 若失败，失败原因与下一步排查项

## Jetson Orin 64G 注意事项
优先考虑可复现性和稳定性，而不是盲目追求吞吐。显存不足时先减小 `--batchsize`，再考虑切换到 `--fp16` 或 `--bf16`。如果训练速度过慢或系统负载过高，优先减少 dataloader 并发和测试 batch size，不要直接改评估逻辑。涉及 Jetson 适配时，允许为兼容性修复 `train.py`、`test.py`、`utils.py` 或数据加载逻辑，但不要顺手重构无关代码。

## 代码规范
保持现有 Python 风格：4 空格缩进，函数和变量使用 `snake_case`，`nn.Module` 类使用 `PascalCase`，命令行参数使用语义明确的长选项。新增脚本优先保持仓库根目录现有的扁平结构，扩展训练或评估逻辑时尽量沿用当前的 `argparse` 参数组织方式。

## 验证要求
仓库没有单元测试，验证主要依赖脚本结果。任何影响训练、测试或数据准备的修改后，至少执行一轮对应脚本验证。复现 `Trained Model` 结果时，默认检查以下内容：
1. 训练命令能正常启动并保存到 `model/<run_name>/`
2. `python test.py --name <run_name>` 能生成评估所需结果
3. `python evaluate.py` 或 `python evaluate_gpu.py` 能输出 Rank@1 和 mAP
4. 若命令或超参数偏离 `README.md`，需在说明中明确写出差异

## 提交与产出要求
提交信息保持简短直接，例如 `fix jetson train fp16`、`update convnext test config`。向用户汇报时，优先说明复现了哪一行模型、使用了什么命令、是否偏离 `README.md`、最终得到的 Rank@1 / mAP，以及 Jetson 上额外做了哪些资源调整。

## 安全与配置
不要提交数据集、模型权重或 `model/` 下的大文件。数据路径、GPU 选择、batch size 和实验参数应通过命令行传入，不要把本地路径或设备专用配置硬编码进代码。
