# 总体计划 v2（2026-11-27 ~ 11-28）

> 本文件是**项目级**计划：时间节点、并行工作流、算力排程、交付物、风险。
> 每个 idea 的**科学设计方案与实施计划**是独立文档，不写在这里：
> - idea1（LTLf 任务逻辑层，v1 已完成）：`work/idea1.md`
> - idea2（STL 定量鲁棒层）：`work/idea2-stl.md`
> - idea3（逻辑对齐后训练）：`work/idea3-posttrain.md`
>
> 记录纪律：实现过程连续记录在 `work/implementation-log.md`；未验证的东西一律标「未验证」。

## 0. 现状（本轮开始时的真实状态）

| 项 | 状态 |
|---|---|
| 代码 | `/mnt/world_foundational_model/kai/MaxHE/ltlf`，分支 `dev`@`f1f46c7`；`AeroVLA/` 与 `baseline-v0` 逐字节一致 |
| idea1 (LTLf) | v1 完成：62 个单测通过；仿真上只跑过 ModernCityMap 的 12/8/8 条小样本 |
| 地图覆盖 | 只用过 ModernCityMap**一张图** |
| 训练 | 未做任何后训练；`AeroVLA/src/train_aerovla.py` + `scripts/train.sh` 可用（LoRA r=64, α=128, lr 2e-4, micro-bs 2, accum 8, 5 epoch）|
| 训练数据 | `AeroVLA/data/aerovla_train_dataset.json`：**410,450 条**样本（含 instruction + 连续动作标签 `{fwd, down, yaw}`）|
| 动作空间 | **连续 3 维**：`fwd`（米，沿新航向位移）、`down`（米）、`yaw`（弧度，**增量**）；执行时 cruise 1.0 m/s，`duration=fwd/1.0`，再停 0.5 s；`|yaw|≥0.25 rad` 先原地转 |
| 状态采样 | 每个宏动作只产生**一个**新位姿（`for _ in range(5)` 是同一帧复制），所以轨迹是"宏步"序列 |

### 服务器算力实况（07:00 实测）
- GPU0 A800-80G：已用 21.5 GB（他人），可用 ≈58 GB；GPU1：已用 64.7 GB（他人，仅剩 ≈17 GB，不敢用）
- 机器 load ≈56，多人共享；pfs `/mnt/kai_ckp` 拥塞时会卡模型加载（已有 `/dev/shm` 模型副本方案）
- 实测吞吐：两臂并行 ≈18 宏步/分钟/臂，≈1.6 min/episode（RT 80 步上限）

### 可用地图（本轮实测结论，覆盖上轮"只有一张图"的说法）
- 仿真场景：`envs/closeloop_envs/` 下 **6 个** UE4 场景（ModernCityDowntown→ModernCityMap、ModularEuropean、ModularPark、NYCEnvironmentMegapa、NewYorkCity、TropicalIsland）
- 数据集：`dataset_raw/` 下 `ModernCityMap(718)`、`NewYorkCity(1139)`、`NYCEnvironmentMegapa(2015)`、`TropicalIsland(1418)`、`ModularPark(958)`、`Carla_Town01(954, 缺 merged_data.json → 跳过)`；Carla_Town04/05/06/07/10HD/15 只有分卷压缩包未解压
- spawn-area 元数据 `data/meta/map_spawnarea_info.json` 覆盖以上全部 UE4 地图
- 仿真器按 `item['map_name']` 自动开场景（`env_uav.py:257` + `AirVLNSimulatorServerTool.py:242`），**因此加地图不需要改代码，只需要 split + tasks**
- **可跑地图数 = 5**（ModernCityMap / NewYorkCity / NYCEnvironmentMegapa / TropicalIsland / ModularPark，其中 ModularPark 是官方 *unseen map* 划分）
- M0 任务已为 4 张新图构建完成：NewYorkCity 145、TropicalIsland 164、NYCEnvironmentMegapa 280、ModularPark 构建中

## 1. 并行工作流

| 流 | 内容 | 谁做 | 阻塞关系 |
|---|---|---|---|
| **W1 实验矩阵** | LTLf 在多地图上跑对照（baseline vs 本层） | 我（脚本+后台作业） | 无；依赖已就绪 |
| **W2 文献** | Logic-VLA（STL+两阶段后训练）、STL Translation&Repair、RefineFly 开源核查 | 3 个 subagent 并行 | 无 |
| **W3 idea2 STL** | 设计 → 实施计划 → 实现 → 离线数值 → 小量仿真 | 我 | 依赖 W2 结论与 W1 的 rollout 日志 |
| **W4 idea3 后训练** | 可行性论证 → 数据管线 → 极小 LoRA 冒烟 | 我 | 依赖 W2 的 Logic-VLA 两阶段细节 |
| **W5 交付物** | 周报（18:00）、组会材料（周六 20:00）、日志 | 我 | 依赖 W1–W4 的实数 |

## 2. 算力排程（两臂并行，每臂一条 GPU 上的独立 port）

| 波次 | 时间窗 | 内容 | 目的 |
|---|---|---|---|
| W1-a | 07:08–08:10 | ModernCityMap **RT 全部 35 条**：G5 本层 fixed_order + G4 baseline+长指令 | 把主结论从 N=8 扩到 N=35 |
| W1-b | ~08:15–09:20 | ModularPark（**unseen map**）G5+G4 各 12 条 | 首次跨图泛化 |
| W1-c | ~09:25–10:30 | NewYorkCity G5+G4 各 12 条 | 跨图第二张 |
| W1-d | ~10:35–11:40 | TropicalIsland G5+G4 各 12 条 | 跨图第三张 |
| W1-e | ~11:45–12:50 | NYCEnvironmentMegapa G5+G4 各 12 条 | 跨图第四张（大图） |
| W1-f | ~12:55–14:00 | ModernCityMap **ST 89 条**：G3+G2 各 30 条 | 短任务上确认不退化 |
| W4-a | 与 W1 错峰 | LoRA 冒烟（50–200 步） | 证明后训练管线可跑 |
| W3-a | ~14:00–16:00 | idea2 STL 在线小量测试（8 条） | 新点位的闭环证据 |

> 若某张新图起场景失败（资产缺失/崩溃），立即记录失败原因并跳过，不阻塞后续波次——**这正是周报里"失败原因"要写的东西**。

## 3. 交付物与时间节点

| 时间 | 交付物 |
|---|---|
| 每波结束 5 分钟内 | 结果 JSON + 一句话结论，写入 `work/results/` 与 `work/implementation-log.md` |
| **18:00** | `work/morning-report.md`（升级为 `work/weekly-report-2026-11-27.md`）：多地图实验矩阵实数、STL 设计+实现+小测、后训练可行性结论、RefineFly 结论 |
| 周六 20:00 前 | 组会版：跨图汇总表 + 每图失败原因 + STL/LTLf 对照表 + 后训练路线图 |

## 4. 文档框架（明确分工，互不重叠）

```
work/
  plan-v2.md                 # 项目级：时间/资源/交付（本文）
  idea2-stl.md               # idea 级：STL 的科学设计 + 实施计划 + review 记录
  idea3-posttrain.md         # idea 级：后训练的科学设计 + 实施计划 + review 记录
  implementation-log.md      # 连续实现记录（含每次失败与修正）
  results/                   # 机器产出（json）
  splits/  tasks/            # 可复现的评测清单与 M0 任务
  weekly-report-2026-11-27.md
```

原则：**项目级文档只回答"什么时候做什么、用什么资源、产出什么"；idea 级文档只回答"为什么这么做、科学上是否成立、怎么验证"**。

## 5. 风险与回退

| 风险 | 触发判据 | 回退 |
|---|---|---|
| 新图起场景失败 | 5 分钟内 server 未 listening / UE4 崩溃 | 记录 `server.log` 尾部，跳过该图，在报告中标"资产不可用" |
| pfs 拥塞 | 模型加载 >6 min 无进展 | 用 `/dev/shm` 模型副本；减少并行臂数 |
| 他人占用 GPU | GPU0 可用 <20 GB | 降到单臂，优先 RT 主对照 |
| STL 实现超时 | 15:30 未跑通离线数值 | 只交付设计+离线 ρ 计算，在线测试顺延到周六 |
| 后训练不可行 | LoRA 冒烟 OOM / 无训练脚本入口 | 只交付可行性论证 + 数据管线，训练顺延 |
| 结论过强 | 样本 <30 条 | 报告一律标注"方向性，未做显著性检验" |
