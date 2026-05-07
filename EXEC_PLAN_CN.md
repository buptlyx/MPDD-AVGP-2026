# MPDD-AVG 执行方案（多赛道、多任务）

> 目标：按官方评测指标选模，覆盖所有赛道与任务，并在可控成本下持续提分。

## 0. 先统一“官方指标”

- 训练选模指标由 `--selection_metric` 控制，脚本已开放环境变量 `SELECTION_METRIC`。
- 如果官方指标是 Macro-F1 / Kappa / ScoreTrack 等，请在训练脚本里设置为官方指标。

可调整位置（任选其一）：
- 训练脚本里设置环境变量：
  - [scripts/Track1/A-V-G+P/run_binary.sh](scripts/Track1/A-V-G+P/run_binary.sh)
  - [scripts/Track1/A-V-G+P/run_ternary.sh](scripts/Track1/A-V-G+P/run_ternary.sh)
  - [scripts/Track1/A-V-P/run_binary.sh](scripts/Track1/A-V-P/run_binary.sh)
  - [scripts/Track1/A-V-P/run_ternary.sh](scripts/Track1/A-V-P/run_ternary.sh)
  - [scripts/Track1/G-P/run_binary.sh](scripts/Track1/G-P/run_binary.sh)
  - [scripts/Track1/G-P/run_ternary.sh](scripts/Track1/G-P/run_ternary.sh)
  - [scripts/Track2/A-V-G+P/run_binary.sh](scripts/Track2/A-V-G+P/run_binary.sh)
  - [scripts/Track2/A-V-G+P/run_ternary.sh](scripts/Track2/A-V-G+P/run_ternary.sh)
  - [scripts/Track2/A-V-P/run_binary.sh](scripts/Track2/A-V-P/run_binary.sh)
  - [scripts/Track2/A-V-P/run_ternary.sh](scripts/Track2/A-V-P/run_ternary.sh)
  - [scripts/Track2/G-P/run_binary.sh](scripts/Track2/G-P/run_binary.sh)
  - [scripts/Track2/G-P/run_ternary.sh](scripts/Track2/G-P/run_ternary.sh)
- 或者直接在训练入口调整参数：
  - [train.py](train.py)

## 1. 第一阶段：全覆盖基线（所有赛道/任务/子赛道）

**目标：** 找到每个赛道/任务的“最强特征组合”。

**训练脚本清单：**
- Track1 / Elder
  - A-V-G+P：
    - 二分类：[scripts/Track1/A-V-G+P/run_binary.sh](scripts/Track1/A-V-G+P/run_binary.sh)
    - 三分类：[scripts/Track1/A-V-G+P/run_ternary.sh](scripts/Track1/A-V-G+P/run_ternary.sh)
  - A-V+P：
    - 二分类：[scripts/Track1/A-V-P/run_binary.sh](scripts/Track1/A-V-P/run_binary.sh)
    - 三分类：[scripts/Track1/A-V-P/run_ternary.sh](scripts/Track1/A-V-P/run_ternary.sh)
  - G+P：
    - 二分类：[scripts/Track1/G-P/run_binary.sh](scripts/Track1/G-P/run_binary.sh)
    - 三分类：[scripts/Track1/G-P/run_ternary.sh](scripts/Track1/G-P/run_ternary.sh)
- Track2 / Young
  - A-V-G+P：
    - 二分类：[scripts/Track2/A-V-G+P/run_binary.sh](scripts/Track2/A-V-G+P/run_binary.sh)
    - 三分类：[scripts/Track2/A-V-G+P/run_ternary.sh](scripts/Track2/A-V-G+P/run_ternary.sh)
  - A-V+P：
    - 二分类：[scripts/Track2/A-V-P/run_binary.sh](scripts/Track2/A-V-P/run_binary.sh)
    - 三分类：[scripts/Track2/A-V-P/run_ternary.sh](scripts/Track2/A-V-P/run_ternary.sh)
  - G+P：
    - 二分类：[scripts/Track2/G-P/run_binary.sh](scripts/Track2/G-P/run_binary.sh)
    - 三分类：[scripts/Track2/G-P/run_ternary.sh](scripts/Track2/G-P/run_ternary.sh)

**A/V 特征组合扫参：**
- 需要手动遍历 3×3 的特征组合：
  - `AUDIO_FEATURE`: mfcc / opensmile / wav2vec
  - `VIDEO_FEATURE`: densenet / resnet / openface
- 建议用你们自己的调度脚本或 shell 循环批量跑，并设置 `EXPERIMENT_NAME` 防止覆盖。

**可调关键参数（环境变量）：**
- `AUDIO_FEATURE` `VIDEO_FEATURE` `ENCODER_TYPE`
- `EPOCHS` `BATCH_SIZE` `LR` `WEIGHT_DECAY`
- `HIDDEN_DIM` `DROPOUT` `TARGET_T`
- `SEED` `VAL_RATIO` `PATIENCE` `MIN_DELTA`
- `WEIGHTED_SAMPLER` `CLS_LOSS_WEIGHT` `REG_LOSS_WEIGHT`

## 2. 第二阶段：编码器对比（固定最强 A/V 组合）

**目标：** 在每个赛道/任务的最佳特征组合上对比编码器。

- `ENCODER_TYPE=bilstm_mean`
- `ENCODER_TYPE=hybrid_attn`

修改位置：同第 1 阶段训练脚本（见上方链接）。

## 3. 第三阶段：小范围超参微调

**目标：** 用小范围网格快速找峰值区域。

建议尝试：
- `HIDDEN_DIM`: 64 / 128 / 256
- `DROPOUT`: 0.2 / 0.3 / 0.5
- `LR`: 3e-5 / 8e-5 / 1e-4
- `TARGET_T`: 128 / 256
- `BATCH_SIZE`: 2 / 4 / 8（按显存调整）

修改位置：同第 1 阶段训练脚本（环境变量）。

## 4. 第四阶段：类别不平衡与损失权重

**目标：** 提升分类指标并稳定回归头。

可调参数：
- `WEIGHTED_SAMPLER=1` 或 `0`
- `CLS_LOSS_WEIGHT`（默认 3.0）
- `REG_LOSS_WEIGHT`（默认 0.1）

修改位置：同第 1 阶段训练脚本（环境变量）。

## 5. 第五阶段：多随机种子 + 集成

**目标：** 稳定提升官方指标。

- 改 `SEED` 多跑 3–5 次。
- 为每次训练设置不同的 `EXPERIMENT_NAME` 便于区分。
- 使用测试脚本对每个 checkpoint 跑测试集并产出 CSV：
  - Track1：
    - [test_scripts/Track1/A-V-G+P/run_binary.sh](test_scripts/Track1/A-V-G+P/run_binary.sh)
    - [test_scripts/Track1/A-V-G+P/run_ternary.sh](test_scripts/Track1/A-V-G+P/run_ternary.sh)
    - [test_scripts/Track1/A-V-P/run_binary.sh](test_scripts/Track1/A-V-P/run_binary.sh)
    - [test_scripts/Track1/A-V-P/run_ternary.sh](test_scripts/Track1/A-V-P/run_ternary.sh)
    - [test_scripts/Track1/G-P/run_binary.sh](test_scripts/Track1/G-P/run_binary.sh)
    - [test_scripts/Track1/G-P/run_ternary.sh](test_scripts/Track1/G-P/run_ternary.sh)
  - Track2：
    - [test_scripts/Track2/A-V-G+P/run_binary.sh](test_scripts/Track2/A-V-G+P/run_binary.sh)
    - [test_scripts/Track2/A-V-G+P/run_ternary.sh](test_scripts/Track2/A-V-G+P/run_ternary.sh)
    - [test_scripts/Track2/A-V-P/run_binary.sh](test_scripts/Track2/A-V-P/run_binary.sh)
    - [test_scripts/Track2/A-V-P/run_ternary.sh](test_scripts/Track2/A-V-P/run_ternary.sh)
    - [test_scripts/Track2/G-P/run_binary.sh](test_scripts/Track2/G-P/run_binary.sh)
    - [test_scripts/Track2/G-P/run_ternary.sh](test_scripts/Track2/G-P/run_ternary.sh)

提示：测试脚本会遍历对应 checkpoint 目录下 `best_model_*.pth`，并写入日志目录。

## 6. 第六阶段：打包提交

**推荐一键流程：**
- 使用 [scripts/run_track_pipeline.py](scripts/run_track_pipeline.py) 在一个命令中完成：训练 -> 测试 -> 生成提交包。
- 该脚本会调用对应的训练脚本与测试脚本，并最终执行打包工具。

**打包工具：**
- [make/make.py](make/make.py)

输出位置参考：
- Checkpoints： [checkpoints](checkpoints)
- 训练日志： [logs](logs)
- 运行产物： [runs](runs)

## 7. Track2 特别注意

- Young 测试集视频存在空文件问题，含视频的子赛道可能在测试阶段退化。
- 建议优先保证 G+P 的强度，再决定是否保留含 V 的子赛道作为提交备选。

---

## 附：4×3090 并行跑法建议

- 以“赛道×子赛道”为粒度分配 GPU：
  - GPU0：Track1 / A-V-G+P
  - GPU1：Track1 / A-V+P
  - GPU2：Track2 / A-V-G+P
  - GPU3：Track2 / G+P
- 同一 GPU 上再通过 `SEED` 或 `A/V 特征组合` 顺序跑。
