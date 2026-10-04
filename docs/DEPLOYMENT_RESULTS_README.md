# RoboTwin 部署评测结果

## 1. 评测口径

- 第 2 节并列展示 `stack_blocks_two` 常规场景与 `hanging_mug` Easy；第 3 节只比较 `stack_blocks_two`，第 4–5 节只比较 `hanging_mug`。两个任务的成功率不能直接相减
- 指令划分：`unseen`
- 正式评测：`stack_blocks_two` 常规场景使用 `100000–100099` 共 100 个 expert-valid seeds；`hanging_mug` Easy 使用同一批 100 个 held-out expert-valid seeds，但不是连续的 `100000–100099`。`hanging_mug` Randomized 各模型共用另一批 100 个有效种子；Easy 和 Randomized 的种子集合不能混用
- 控制执行：官方 Mplib TOPP；线性 fallback 不作为正式结果
- 成功判定：RoboTwin 原始任务成功谓词
- 去重键：模型权重、训练输入、部署输入、种子集合和任务配置
- 去重规则：同一键只保留时间最新、通过完成标记与动作运动门禁、且没有 `INVALID_INPUT_COLOR_ORDER` 标记的正式批次
- 排除项：smoke、preflight、正常中断且未满 100 个种子的结果、无机械臂运动结果和诊断结果。第 5 节的预设“75 轮零成功即停止”另行标明实测轮数，不写成实测 0/100
- 场景配置：`stack_blocks_two` 常规场景的深度版 π0.5 使用 `demo_clean_depth`，RGB-only 官方 π0.5 JAX 使用 `demo_clean`，ACT3 使用 `depth_master_clean`；两个 `demo_clean` 配置除深度采集开关外相同。随机化场景使用 `demo_randomized_depth_20260924`，启用随机背景、杂物桌面、桌面高度与灯光扰动。`hanging_mug` Easy 使用 `depth_master_clean`
- 随机化场景会跳过未通过 expert 有效性检查的种子；各单元的 100 个有效种子不保证完全相同，因此跨单元差值仅作描述性比较

深度环境定义：

- `D0`：干净 GT metric depth
- `D1`：确定性的 RealSense D435/D405 噪声深度
- `D3`：保留 D1 有效像素，仅使用 LingBot-Depth v0.5 填补 D1 空洞

历史在线 D1/D3 部署存在 RGB/BGR 错配；第 2–3 节的 D1/D3 成绩只采用完成重评且通过运行检查的正式批次。旧值的影响范围与重评位置见第 6 节。部署 D0 不受该颜色顺序错误影响，但仍须独立通过动作执行等有效性检查。

## 2. 最终部署主表

| 模型 | 架构与训练/部署输入 | 训练预算 | `stack_blocks_two` 常规 | `hanging_mug` Easy |
| --- | --- | ---: | ---: | ---: |
| `ACT0_RGB` | 官方 ACT；三视角共享 RGB ResNet18 + joint | 6000 epochs | **18%** | **8%** |
| `ACT1_EARLY_RGBD` | 每视角 RGB + D0 metric depth 四通道早期融合 | 6000 epochs | **23%** | **16%** |
| `ACT2_DUAL_SHARED` | 共享 RGB ResNet18 + 共享 Depth ResNet18；D0 部署 | 6000 epochs | **7%** | **10%** |
| `ACT3_DUAL_PER_VIEW` | 3 个 RGB ResNet18 + 3 个 Depth ResNet18；D0 训练、D0 部署 | 6000 epochs | **31%** | **10%** |
| `ACT4_XYZMAP` | RGB ResNet18 + D0 Depth 转相机坐标 XYZ 点图 | 6000 epochs | **20%** | **7%** |
| `ACT5_POINT_TOKENS` | RGB ResNet18 + D0 XYZ Point Tokens | 6000 epochs | **20%** | **6%** |
| `ACT6_LINGBOT_DEPTH` | RGB ResNet18 + 冻结 LingBot-Depth v0.5 Tokens；D0 部署 | 6000 epochs | **11%** | **3%** |
| `ACT7_DEPTH_TRANSFORMER` | RGB ResNet18 + D0 Depth Transformer Tokens | 6000 epochs | **16%** | **11%** |
| 官方 pi0.5 JAX | 三视角 RGB + joint + prompt，全参微调；两任务分别使用 `demo_clean`、`depth_master_clean` | 20000 steps | **63%** | **21%** |
| `PI05_RGBD4`，D0 权重 | 每视角 RGB + D0 metric depth 四通道早期融合 | 20000 steps | 未评测 | **25%** |
| `PI05_DUAL_PER_VIEW`，D0 权重 | pi0.5 + 三个独立 Depth ResNet18；D0 部署 | 20000 steps | **66%** | **28%** |
| `PI05_DUAL_PER_VIEW`，D1 权重 | 同一双流架构；D1 训练、D0 部署 | 20000 steps | 未评测 | **27%** |
| `PI05_DUAL_PER_VIEW`，D3 权重 | 同一架构；D3 训练、D3 部署 | 20000 steps | **64%** | 未评测 |
| LingBot-VLA 2.0 | Qwen3-VL-4B + MoE Action Expert；三视角 RGB + joint + prompt | 30000 steps | **65%** | 未评测 |

两列使用不同任务数据与种子，仅便于查阅，不能作为同任务泛化差值。`ACT3_DUAL_PER_VIEW` 的 `stack_blocks_two` 与 `hanging_mug` 成绩来自各自训练的不同权重；后者使用新训练的 2-worker D0 权重。LingBot-VLA 2.0 使用 MoGe/LingBot-Depth 和 DINO-Video 提供训练期几何与时序蒸馏监督，但本次部署不直接读取 RoboTwin GT Depth 通道。

## 3. SOTA模型架构在不同深度环境的表现

### 常规场景：ACT 与 π0.5

任务为 `stack_blocks_two`。ACT3 与深度版 π0.5 的各自部署矩阵均使用 `100000–100099` 共 100 个有效种子；所有 D1/D3 部署格采用完成颜色通道修正后的正式成绩。

| 模型架构 | 训练深度环境 | 部署 D0 | 部署 D1 | 部署 D3 |
| --- | --- | ---: | ---: | ---: |
| `ACT3_DUAL_PER_VIEW` | D0 clean GT | **31%** | **26%** | **33%** |
| `ACT3_DUAL_PER_VIEW` | D1 RealSense noise | **16%** | **24%** | **16%** |
| `ACT3_DUAL_PER_VIEW` | D3 LingBot sensor-fused | **19%** | **12%** | **20%** |
| 官方 pi0.5 JAX | RGB-only；`demo_clean` | **63%** | **63%** | **63%** |
| `PI05_DUAL_PER_VIEW` | D0 clean GT | **66%** | **57%** | **56%** |
| `PI05_DUAL_PER_VIEW` | D1 RealSense noise | **55%** | **60%** | **58%** |
| `PI05_DUAL_PER_VIEW` | D3 LingBot sensor-fused | **59%** | **61%** | **64%** |

官方 pi0.5 JAX **仅实际评测 RGB-only 输入 63/100**；D1、D3 两列借用 D0 的成功率作为同一 RGB-only 基线的展示值，**不是两次独立评测**，不能据此推断深度处理对它的影响。官方 JAX 的 AutoDL 正式日志位于 `state/autodl_pi05_official_eval/formal.log`，完成标志为同目录 `PI05_EVAL_COMPLETE`。`demo_clean` 与深度版 `demo_clean_depth` 并非完全相同配置，跨模型比较时须注明。

### 随机化场景：ACT 与 π0.5

任务为 `stack_blocks_two`，配置为 `demo_randomized_depth_20260924`。ACT0–ACT7 的汇总成功率按本次确认的正式评测结论填写；各架构、权重及训练深度组合的逐项记录不在这一行展开。

| 模型架构 | 训练深度环境 | 部署 D0 | 部署 D1 | 部署 D3 |
| --- | --- | ---: | ---: | ---: |
| `ACT0–ACT7` | D0/D1/D3；ACT0 为 RGB-only | **0%** | **0%** | **0%** |
| 官方 pi0.5 JAX | RGB-only | **21%** | **21%** | **21%** |
| `PI05_DUAL_PER_VIEW` | D0 clean GT | **15%** | **18%** | **24%** |
| `PI05_DUAL_PER_VIEW` | D1 RealSense noise | **16%** | **18%** | **15%** |
| `PI05_DUAL_PER_VIEW` | D3 LingBot sensor-fused | **17%** | **21%** | **24%** |

ACT0 只读取 RGB，因此汇总行中的 D0/D1/D3 不表示 ACT0 使用深度训练。官方 pi0.5 JAX 只独立评测 RGB-only 输入，其 D1、D3 列复用 D0 的 **21/100**，并非独立评测。随机化场景不同批次的 expert 有效性筛选可能造成种子集合差异，跨批次差值仅作描述性比较。

## 4. `hanging_mug` Easy：ACT 与 π0.5

### ACT 架构评测

任务配置为 `depth_master_clean`。八种架构均使用同一批 **100 个 expert-valid held-out seeds**：原 50 个种子加上 50 个不同的新种子；筛掉无效种子后，最终集合不等于连续的 `100000–100099`。每格均有 `FORMAL_COMPLETE`、`final_result.json` 和一致的 seed SHA256（`ca71e696dd855cd0e11aa8862febc36bf9cdbd9d26d50001207a080450fa3241`）。评测检查记录了非零目标动作和实际关节运动。部署均为 D0，模型权重均训练至 6000 epoch；表中 ACT3 使用 AutoDL 新训练的 2-worker D0 权重。

| 模型架构 | 训练深度 | 部署 D0（100 轮） |
| --- | --- | ---: |
| `ACT0_RGB` | D0 数据，模型仅用 RGB | **8/100 = 8%** |
| `ACT1_EARLY_RGBD` | D0 clean GT | **16/100 = 16%** |
| `ACT2_DUAL_SHARED` | D0 clean GT | **10/100 = 10%** |
| `ACT3_DUAL_PER_VIEW`（新权重） | D0 clean GT | **10/100 = 10%** |
| `ACT4_XYZMAP` | D0 clean GT | **7/100 = 7%** |
| `ACT5_POINT_TOKENS` | D0 clean GT | **6/100 = 6%** |
| `ACT6_LINGBOT_DEPTH` | D0 clean GT | **3/100 = 3%** |
| `ACT7_DEPTH_TRANSFORMER` | D0 clean GT | **11/100 = 11%** |

表中七个非 ACT3 架构的评测记录位于 AutoDL `evaluations/hanging_mug_easy100_20260929/state/`，原 50 轮的逐 seed 结果已并入 100 轮，**不可再与表中结果相加**。新版 ACT3 首次独立评满同一批 100 个种子的记录位于 `evaluations/hanging_mug_act3_autodl_d0_easy100_20260929/state/ACT3_DUAL_PER_VIEW_d0_to_d0/`；下方跨深度矩阵的最新 D0→D0 批次同为 **10/100**，只计一次。旧 ACT3 的 100 轮 D0→D0 结果为 **3/100**，使用此前在 56 服务器训练的 6-worker 权重，仅保留追溯。旧 50 轮跨深度记录位于 `evaluations/hanging_mug_easy_matrix_20260928/state/`，不与新版 100 轮结果混用。旧 ACT3 使用 6 个 worker，新 ACT3 与其他七个架构均使用 2 个；**3%→10% 是两次不同权重的结果，不能归因于 worker 数量本身**。

### ACT 跨深度部署

下表使用 AutoDL 的新版 ACT3 权重及 ACT1 权重，均训练至 **6000 epoch**；各完成单元评满同一批 100 个 Easy held-out seeds，seed SHA256 与上述架构表一致。正式结果均有 `FORMAL_COMPLETE`、`SMOKE_COMPLETE` 与 `final_result.json`，逐 seed 成功计数与汇总一致，未发现无效标记。

| 模型架构 | 训练深度环境 | 部署 D0 | 部署 D1 | 部署 D3 |
| --- | --- | ---: | ---: | ---: |
| `ACT1_EARLY_RGBD` | D0 clean GT | **16%** | 未评测 | 未评测 |
| `ACT1_EARLY_RGBD` | D1 RealSense noise | **15%** | **10%** | **11%** |
| `ACT1_EARLY_RGBD` | D3 LingBot sensor-fused | **13%** | **13%** | **13%** |
| `ACT3_DUAL_PER_VIEW` | D0 clean GT | **10%** | **10%** | **12%** |
| `ACT3_DUAL_PER_VIEW` | D1 RealSense noise | **0%** | **4%** | **0%** |
| `ACT3_DUAL_PER_VIEW` | D3 LingBot sensor-fused | **6%** | **13%** | **6%** |

ACT3 D0、D1 矩阵位于 AutoDL `evaluations/hanging_mug_act3_new_depth_matrix_20260930/state/`；ACT1 D1、D3 与 ACT3 D3 矩阵位于 `evaluations/hanging_mug_act1_act3_depth_matrix_20260930/state/`。ACT3 D3→D3 已续评至 **6/100**，有正式完成标记。ACT3 D1→D0 与 D1→D3 的 **0/100** 均为完成批次，不与未完成的零成功进度混淆。

### π0.5 跨深度部署

下表权重均针对 `hanging_mug` 训练至 **20000 steps**。AutoDL 各实际评测单元均评满同一批 100 个 Easy held-out seeds，seed SHA256 与 ACT Easy 表一致；均有 `SMOKE_COMPLETE`、`FORMAL_COMPLETE`、`final_result.json`，逐 seed 计数与成功汇总一致。正式日志每轮均有动作执行探针，未发现动作失效或输入颜色顺序失效标记。

| 模型架构 | 训练深度环境 | 部署 D0 | 部署 D1 | 部署 D3 |
| --- | --- | ---: | ---: | ---: |
| 官方 pi0.5 JAX | RGB-only | **21%** | **21%** | **21%** |
| `PI05_RGBD4` | D0 clean GT | **25%** | **27%** | **23%** |
| `PI05_DUAL_PER_VIEW` | D0 clean GT | **28%** | **34%** | **26%** |
| `PI05_DUAL_PER_VIEW` | D1 RealSense noise | **27%** | **22%** | **23%** |

官方 pi0.5 JAX 仅实际评测 RGB-only 输入 **21/100**；D1、D3 两列复用同一结果，**不是独立评测**。四通道早期融合与双流 Depth ResNet18 是不同架构，不能把两者差值单独归因于深度处理方法。原 D0 双流记录位于 AutoDL `evaluations/hanging_mug_pi05_d0_matrix_20261001/state/`，权重 SHA256 为 `922b03d5e75ce0f73b64661f1fd257bfb1c896f0282bd8d585c4a36b20d062b1`；新增记录分别位于 `evaluations/hanging_mug_pi05_official_20261002/state/`、`evaluations/hanging_mug_pi05_rgbd4_matrix_20261002/state/` 和 `evaluations/hanging_mug_pi05_d1_matrix_20261003/state/`。

## 5. `hanging_mug` Randomized：ACT 与 π0.5

### ACT0–ACT7 统一评测

AutoDL 已按 `demo_randomized_depth_20260924` 筛出同一批 **100 个 expert-valid held-out seeds**，八个架构共用固定种子文件，SHA256 为 `2081a9b67e0210e664999bdf96a77b30939544b48830df44af39587ce14d6c4c`。部署深度均为 D0 clean GT，训练权重均为 D0 数据、6000 epoch；ACT0 仅使用 RGB。ACT3 使用 AutoDL 新训练的 2-worker D0 权重，其余七个架构沿用第 4 节对应的 6000-epoch 权重。八项的官方 Mplib TOPP、动作执行探针及深度输入探针均通过，未发现 `ACTION_EXECUTION_INVALID`。

| 模型架构 | 实测进度 | 状态/记分 |
| --- | ---: | ---: |
| `ACT0_RGB` | 0/75 | 提前停止；按约定记 0% |
| `ACT1_EARLY_RGBD` | 0/75 | 提前停止；按约定记 0% |
| `ACT2_DUAL_SHARED` | 0/75 | 提前停止；按约定记 0% |
| `ACT3_DUAL_PER_VIEW`（新权重） | 0/75 | 提前停止；按约定记 0% |
| `ACT4_XYZMAP` | 0/75 | 提前停止；按约定记 0% |
| `ACT5_POINT_TOKENS` | 0/70 | 已停止；未评满，无正式成功率 |
| `ACT6_LINGBOT_DEPTH` | 0/29 | 已停止；未评满，无正式成功率 |
| `ACT7_DEPTH_TRANSFORMER` | 0/43 | 已停止；未评满，无正式成功率 |

队列逐 seed 保存进度。若某架构评满 **75 轮仍零成功**，按本批次约定提前停止并报告 **0%**，同时在结果中保留 `observed_episodes=75`、`untested_episodes=25` 和 `EARLY_STOP_ZERO_AT_75`；这不是实测 0/100。ACT5–ACT7 已按要求停止，保留进度但尚未达到正式判定门槛，不能把当前零成功写成最终 0%。评测记录位于 AutoDL `evaluations/hanging_mug_random100_20260929/state/`；队列已停止并设置 `PAUSED` 标记。尚未记录杯子到达双臂中间等阶段指标，零成功不能据此定位失败阶段。

### π0.5 跨深度部署

下表权重均针对 `hanging_mug` 训练至 **20000 steps**，任务配置为 `demo_randomized_depth_20260924`。AutoDL 各实际评测单元均评满同一批 **100 个 expert-valid held-out seeds**，seed SHA256 为 `2081a9b67e0210e664999bdf96a77b30939544b48830df44af39587ce14d6c4c`，与上述 ACT Randomized 的固定种子文件一致；均有 `FORMAL_COMPLETE`、`SMOKE_COMPLETE` 与 `final_result.json`，逐 seed 成功计数一致，动作执行探针通过。

| 模型架构 | 训练深度环境 | 部署 D0 | 部署 D1 | 部署 D3 |
| --- | --- | ---: | ---: | ---: |
| 官方 pi0.5 JAX | RGB-only | **17%** | **17%** | **17%** |
| `PI05_RGBD4` | D0 clean GT | **16%** | **18%** | **18%** |
| `PI05_DUAL_PER_VIEW` | D0 clean GT | **18%** | **18%** | **16%** |
| `PI05_DUAL_PER_VIEW` | D1 RealSense noise | **19%** | **18%** | **14%** |

官方 pi0.5 JAX 仅实际评测 RGB-only 输入 **17/100**；D1、D3 两列复用同一结果，**不是独立评测**。原 D0 双流记录位于 AutoDL `evaluations/hanging_mug_pi05_d0_matrix_20261001/state/`，与 Easy 表共用同一权重 SHA256；新增记录位于上节列出的三个独立批次目录。ACT 的提前停止批次没有完整覆盖 100 个种子，比较时须保留其实际评测轮数。

## 6. 受影响的评测配置与重评位置

下表按**唯一模型/训练输入/场景/部署输入**计数；历史主表中的 π0.5 D3→D3 旧 68% 与常规场景表是同一项，不另算一次。原表中常规 ACT3 有 6 格、常规 π0.5 有 6 格、随机化 π0.5 有 6 格，合计 **18 格旧 D1/D3 成绩无效**；这 18 格均已取得修正后的正式成绩。旧值仅供追溯，不参与排名。

| 场景与模型权重 | 受影响部署输入及旧记录 | 修复版重评位置 |
| --- | --- | --- |
| 常规 ACT3，D0 训练 | D1：27%；D3：33% | AutoDL `stack_blocks_two_doc_colorfix_20260926`：**26% / 33%** |
| 常规 ACT3，D1 训练 | D1：23%；D3：17% | AutoDL `stack_blocks_two_easy_colorfix_autodl_20260926`：**24% / 16%** |
| 常规 ACT3，D3 训练 | D1：13%；D3：18% | AutoDL 同一 Easy 队列：**12% / 20%** |
| 常规 π0.5，D0 训练 | D1：59%；D3：61% | 56 修复版已完成：**57% / 56%** |
| 常规 π0.5，D1 训练 | D1：61%；D3：64% | 56 修复版已完成：**60% / 58%** |
| 常规 π0.5，D3 训练 | D1：68%；D3：68% | AutoDL 修复版已完成：**61% / 64%** |
| 随机化 π0.5，D0 训练 | D1：20%；D3：26% | AutoDL 修复版已完成：**18% / 24%** |
| 随机化 π0.5，D1 训练 | D1：17%；D3：21% | AutoDL 修复版已完成：**18% / 15%** |
| 随机化 π0.5，D3 训练 | D1：旧 17%；D3：旧 17% | AutoDL 修复版已完成：**21% / 24%** |
| 随机化 ACT3，D0/D1/D3 训练 | 六个 D1/D3 部署格，旧批次含完成与部分进度，未在上表展示 | AutoDL `stack_blocks_two_random_matrix_colorfix_20260926` |

**不受此次错误影响：**各表的部署 D0 列、主表其他 D0 或 RGB-only 模型，以及离线 D1/D3 训练数据和 checkpoint。这里的“无效”仅针对预期的训练/部署深度定义匹配；不等于模型或权重本身无效。旧结果保留并加 `INVALID_INPUT_COLOR_ORDER` 标记，新结果写入独立目录，不接续旧 D1/D3 进度。详见 [颜色通道审计说明](EVAL_INPUT_COLOR_ORDER_AUDIT_20260926.md)。

## 7. 双任务训练 RGB-D 视频

[交互视频与成功率页面](https://expolrer.github.io/depth-process-model/robotwin/) 提供 `stack_blocks_two` 与 `hanging_mug` 各一条**训练集 episode 0 的完整轨迹**，每条轨迹分别展示 D0、D1、D3。视频上排为头部、左腕、右腕 RGB，下排为对应深度；每个任务、每个相机的深度色标跨三种方法固定，黑色表示无效深度。视频仅作训练输入的可视化，不代表全部 50 条训练轨迹，也不能替代第 2–5 节的部署成功率。

| 训练任务 | D0 干净 GT | D1 RealSense 模拟噪声 | D3 LingBot 修复 |
| --- | --- | --- | --- |
| `stack_blocks_two` | [播放](https://expolrer.github.io/depth-process-model/robotwin/?scene=stack_blocks_two&method=d0) | [播放](https://expolrer.github.io/depth-process-model/robotwin/?scene=stack_blocks_two&method=d1) | [播放](https://expolrer.github.io/depth-process-model/robotwin/?scene=stack_blocks_two&method=d3) |
| `hanging_mug` | [播放](https://expolrer.github.io/depth-process-model/robotwin/?scene=hanging_mug&method=d0) | [播放](https://expolrer.github.io/depth-process-model/robotwin/?scene=hanging_mug&method=d1) | [播放](https://expolrer.github.io/depth-process-model/robotwin/?scene=hanging_mug&method=d3) |

视频为 30 FPS、H.264 MP4，`stack_blocks_two` 310 帧，`hanging_mug` 337 帧。D0 来自 `/ssd/hhw/depth-model/datasets/master/`，D1 和 D3 分别来自 `/ssd/hhw/depth-model/datasets/derived/realsense_d1/` 与 `/ssd/hhw/depth-model/datasets/derived/lingbot_d3/`。逐文件 SHA256、深度有效像素比例和具体源文件见 [媒体清单](https://github.com/expolrer/depth-process-model/blob/main/docs/robotwin/media/manifest.json)；导出脚本见 [export_training_rgbd_videos.py](https://github.com/expolrer/depth-process-model/blob/main/scripts/export_training_rgbd_videos.py)。
