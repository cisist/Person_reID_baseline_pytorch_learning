# Jetson Orin 64G 上 README Trained Model 首周复现实验报告

## 封面信息
- 项目名称：Person ReID Baseline PyTorch 复现
- 报告主题：Jetson Orin 64G 首周复现实验报告
- 实验编号：JETSON-REID-W1-001
- 报告版本：v0.1
- 负责人：
- 协作者：
- 日期范围：2026-03-23 至 2026-03-29
- 报告日期：
- 仓库路径：`/home/r3644/work/zhiyuan/codex/Person_reID_baseline_pytorch_learning`
- 目标平台：Jetson Orin 64G

---

## 目录
1. 实验目标
2. 实验平台
3. 复现策略
4. 首周总体结论
5. 结果总表
6. 模型分项结果
7. 每日推进记录
8. 问题与排障记录
9. 代码与配置改动记录
10. 下周计划

---

## 1. 实验目标
本文档用于记录在 Jetson Orin 64G 平台上，对仓库 `README.md` 中 `Trained Model` 表格结果进行首周复现的过程与结果。

首周目标如下：
- 建立稳定、可重复的训练-测试-评估闭环
- 至少完成 `ResNet-50 (fp16)` 与 `ResNet-50` 两组结果
- 理想情况下补充完成 `ResNet-50-ibn` 或 `DenseNet-121`
- 明确 Jetson 平台上的资源瓶颈、兼容性问题与后续优化方向

## 2. 实验平台
### 2.1 硬件环境
- 设备：Jetson Orin 64G
- CPU：
- GPU：
- 内存：64 GB
- 存储：
- 功耗模式：

### 2.2 软件环境
- 日期：
- JetPack：
- CUDA：
- cuDNN：
- Python：
- PyTorch：
- Torchvision：
- `timm`：
- `pretrainedmodels`：
- 工作目录：
- 数据集路径：

### 2.3 数据集说明
- 数据集名称：
- 数据集原始路径：
- `prepare.py` 执行状态：
- 训练/测试目录结构检查结果：

## 3. 复现策略
### 3.1 复现原则
- 优先保持与 `README.md` 一致的模型结构、损失函数和评估流程。
- 若因 Jetson 资源限制必须调整参数，优先调整 `--batchsize`、低精度模式和数据加载并发，不优先改模型逻辑。
- 任何偏离 `README.md` 原命令的修改，都需要在结果表中明确记录。

### 3.2 首周推进顺序
1. `ResNet-50 (fp16)`
2. `ResNet-50`
3. `ResNet-50-ibn`
4. `DenseNet-121`
5. `DenseNet-121 (Circle)`
6. `ResNet-50 + USAM`

### 3.3 成功判定标准
- 训练命令可正常启动并完成
- `test.py` 可正常载入权重并输出结果
- `evaluate.py` 或 `evaluate_gpu.py` 可得到 Rank@1 与 mAP
- 结果可用于与 README 对照，或能明确解释偏差来源

## 4. 首周总体结论
### 4.1 结论摘要
- 首周整体进展：
- 是否完成基础闭环：是 / 否
- 已完成模型：
- 未完成模型：
- 当前主要瓶颈：

### 4.2 关键发现
- 
- 
- 

### 4.3 后续建议
- 
- 
- 

## 5. 结果总表
| 模型 | README 命令 | 实际命令 | 是否偏离 README | Rank@1 | mAP | 训练耗时 | 测试耗时 | 状态 | 备注 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| ResNet-50 (fp16) | `python train.py --name fp16 --fp16 --train_all` |  |  |  |  |  |  |  |  |
| ResNet-50 | `python train.py --train_all` |  |  |  |  |  |  |  |  |
| ResNet-50-ibn | `python train.py --train_all --name res-ibn --ibn` |  |  |  |  |  |  |  |  |
| DenseNet-121 | `python train.py --name ft_net_dense --use_dense --train_all` |  |  |  |  |  |  |  |  |
| DenseNet-121 (Circle) | `python train.py --name ft_net_dense_circle_w5 --circle --use_dense --train_all --warm_epoch 5` |  |  |  |  |  |  |  |  |
| ResNet-50 + USAM | `python train.py --train_all --name res-usam --usam` |  |  |  |  |  |  |  |  |

## 6. 模型分项结果
### 6.1 ResNet-50 (fp16)
#### 目标
验证 Jetson 上低精度训练、测试、评估链路是否完整可用。

#### README 参考命令
```bash
python train.py --name fp16 --fp16 --train_all
```

#### 实际执行命令
```bash
# 填写实际命令
```

#### 运行配置
- Train batch size:
- Test batch size:
- Precision mode: fp16
- 其他调整：

#### 实验结果
- Rank@1:
- mAP:
- 训练耗时:
- 测试耗时:
- 是否稳定:

#### 结果分析
- 

#### 遇到的问题
- 

#### 结论
- 

### 6.2 ResNet-50
#### 目标
建立 README 主基线，并与 fp16 结果做对照。

#### README 参考命令
```bash
python train.py --train_all
```

#### 实际执行命令
```bash
# 填写实际命令
```

#### 运行配置
- Train batch size:
- Test batch size:
- Precision mode:
- 其他调整：

#### 实验结果
- Rank@1:
- mAP:
- 训练耗时:
- 测试耗时:
- 与 ResNet-50(fp16) 对比：

#### 结果分析
- 

#### 遇到的问题
- 

#### 结论
- 

### 6.3 ResNet-50-ibn
#### README 参考命令
```bash
python train.py --train_all --name res-ibn --ibn
python test.py --name res-ibn
python evaluate.py
```

#### 实际执行命令
```bash
# 填写实际命令
```

#### 运行配置
- Train batch size:
- Test batch size:
- 其他调整：

#### 实验结果
- Rank@1:
- mAP:
- 训练耗时:
- 测试耗时:

#### 结果分析
- 

#### 遇到的问题
- 

#### 结论
- 

### 6.4 DenseNet-121
#### README 参考命令
```bash
python train.py --name ft_net_dense --use_dense --train_all
python test.py --name ft_net_dense
python evaluate.py
```

#### 实际执行命令
```bash
# 填写实际命令
```

#### 运行配置
- Train batch size:
- Test batch size:
- 其他调整：

#### 实验结果
- Rank@1:
- mAP:
- 训练耗时:
- 测试耗时:

#### 结果分析
- 

#### 遇到的问题
- 

#### 结论
- 

### 6.5 DenseNet-121 (Circle)
#### README 参考命令
```bash
python train.py --name ft_net_dense_circle_w5 --circle --use_dense --train_all --warm_epoch 5
python test.py --name ft_net_dense_circle_w5
python evaluate.py
```

#### 实际执行命令
```bash
# 填写实际命令
```

#### 运行配置
- Train batch size:
- Test batch size:
- 其他调整：

#### 实验结果
- Rank@1:
- mAP:
- 训练耗时:
- 测试耗时:
- 数值稳定性：

#### 结果分析
- 

#### 遇到的问题
- 

#### 结论
- 

### 6.6 ResNet-50 + USAM
#### README 参考命令
```bash
python train.py --train_all --name res-usam --usam
python test.py --name res-usam
python evaluate.py
```

#### 实际执行命令
```bash
# 填写实际命令
```

#### 运行配置
- Train batch size:
- Test batch size:
- 其他调整：

#### 实验结果
- Rank@1:
- mAP:
- 训练耗时:
- 测试耗时:

#### 结果分析
- 

#### 遇到的问题
- 

#### 结论
- 

## 7. 每日推进记录
### Day 1 环境预检与 Smoke Test
#### 当日目标
确认依赖、数据目录和短时训练启动是否正常。

#### 实际执行
```bash
# 填写当日命令
```

#### 当日结果
- 依赖状态：
- 数据状态：
- `prepare.py` 状态：
- Smoke test 状态：

#### 当日问题
- 

#### 当日决策
- 

### Day 2 ResNet-50 (fp16)
#### 当日目标
完成 `ResNet-50 (fp16)` 训练、测试与评估。

#### 实际执行
```bash
# 填写当日命令
```

#### 当日结果
- Rank@1:
- mAP:
- 耗时：

#### 当日问题
- 

#### 当日决策
- 

### Day 3 ResNet-50
#### 当日目标
完成 `ResNet-50` 基准线并与 Day 2 对比。

#### 实际执行
```bash
# 填写当日命令
```

#### 当日结果
- Rank@1:
- mAP:
- 耗时：

#### 当日问题
- 

#### 当日决策
- 

### Day 4 候选模型推进
#### 当日目标
优先完成 `ResNet-50-ibn`，若受阻则切换 `DenseNet-121`。

#### 实际执行
```bash
# 填写当日命令
```

#### 当日结果
- 完成模型：
- Rank@1:
- mAP:

#### 当日问题
- 

#### 当日决策
- 

### Day 5 补齐阶段并尝试 DenseNet-Circle
#### 当日目标
补齐未完成模型，并决定是否进入 Circle Loss 配置。

#### 实际执行
```bash
# 填写当日命令
```

#### 当日结果
- 完成项目：
- 新启动项目：
- 当前结论：

#### 当日问题
- 

#### 当日决策
- 

### Day 6 DenseNet-Circle 或 USAM
#### 当日目标
优先完成 `DenseNet-121 (Circle)`，再决定是否进入 `USAM`。

#### 实际执行
```bash
# 填写当日命令
```

#### 当日结果
- 完成模型：
- Rank@1:
- mAP:
- 资源代价：

#### 当日问题
- 

#### 当日决策
- 

### Day 7 周总结
#### 当日目标
补全本周实验闭环，形成下周优先级建议。

#### 本周完成情况
- 已完成：
- 未完成：
- 主要问题：
- 下周优先级：

## 8. 问题与排障记录
| 日期 | 模型 | 问题现象 | 初步原因 | 已采取措施 | 当前状态 |
| ---- | ---- | ---- | ---- | ---- | ---- |
|  |  |  |  |  |  |

## 9. 代码与配置改动记录
| 日期 | 文件 | 改动内容 | 改动原因 | 是否影响与 README 对齐 |
| ---- | ---- | ---- | ---- | ---- |
|  |  |  |  |  |

## 10. 下周计划
### 10.1 优先任务
- 
- 
- 

### 10.2 风险项
- 
- 
- 

### 10.3 需要补充验证的内容
- 
- 
- 
