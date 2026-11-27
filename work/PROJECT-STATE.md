# 项目状态快照（2026-11-27 22:50）

> 用途：**替代长对话的上下文压缩件**。新会话只需读这一份 + `work/weekly-report-2026-11-27.md` 即可继续。
> 纪律：本文件只写"已验证的事实"和"明确未验证的假设"，不写推测。

## 0. 一句话现状

在外挂逻辑层上做了三件事（LTLf 骨架 / STL 定量补充 / 逻辑对齐后训练），**都跑通了、但都没有拿到经得起检验的成功率提升**；
本周最大的产出是**定位清楚问题**：失败主因是起飞段碰撞（70–74%），不是逻辑；另外发现评测侧有两个坑（split 语义、NYC 起飞位姿）。

## 1. 环境

| 项 | 值 |
|---|---|
| 服务器 | `root@10.40.0.190:7735`（**2026-11-27 21:55 容器被平台重建，22:46 起本地到该 IP 路由异常，待恢复**） |
| 宿主机名 | `aibox-r155db332cb6-669f64bffd-828lv`（重建前 `-4fnw6`）；cgroup 内存上限 **240 GB** |
| 仓库 | `/mnt/world_foundational_model/kai/MaxHE/ltlf`（分支 `dev`，tag `baseline-v0`）；GitHub `MaxHE911/aerovla-ltlf`（服务器无凭证，经 Windows 桥接 `work/.gitbridge/ltlf.git` 推送） |
| 数据/模型 | `/mnt/kai_ckp/yiding/MaxHE/`（`dataset_raw/`、`eval_results/_ltlf/`、`ltlf_logs/`、`models/refinefly`、`envs_py/aerovla_py310`） |
| Python | `/mnt/kai_ckp/yiding/MaxHE/envs_py/aerovla_py310/bin/python`（有 pytest、matplotlib） |
| 仿真 | Xvfb `:99` + UE4（`envs/closeloop_envs/*`）；UE4 拒绝 root → `LD_PRELOAD` uid/gid shim |
| 硬约束 | `AeroVLA/` 与 `baseline-v0` **逐字节一致**（`git diff --stat baseline-v0 -- AeroVLA/` 为空）；所有新增文件在 `ltlf_logic/ ltlf_tests/ ops/ work/` |

## 2. 代码地图

| 模块 | 职责 | 规模 |
|---|---|---|
| `ltlf_logic/formula.py` | LTLf AST、`normalize`/`prog`/`accept`/`parse` | 601 行 |
| `ltlf_logic/semantics.py` | 独立的有限迹 `sat()`（交叉验证用） | 55 行 |
| `ltlf_logic/automaton.py` | 残差自动机构建、接受路径枚举 | 213 行 |
| `ltlf_logic/monitor.py` | `Monitor`：事件检测（12 m 3-D 球）、残差推进 | 150 行 |
| `ltlf_logic/tasks.py` | **M0**：从专家轨迹挖地标序列 → 任务/公式/长指令/ST-RT 标签 | 658 行 |
| `ltlf_logic/offline.py` | 离线回放、`independent_esr`（判据 oracle） | 225 行 |
| `ltlf_logic/runner.py` | 配置（含 `mode=stl`）、阶段机、prompt 构造、gate/takeover | 399 行 |
| `ltlf_logic/eval_loop.py` | 评测主循环（subclass baseline `EvalBatchState`） | 501 行 |
| `ltlf_logic/stl.py` | 采样时刻 STL：AST + min/max 鲁棒度 + 在线上下界 | — |
| `ltlf_logic/stl_tasks.py` | chain / schedule / 安全包络构建 + 离线打分 CLI | — |
| `ltlf_logic/stl_layer.py` | 在线：`observe`（控制步）、`deadline_hint`、`shield`、`summary` | — |
| `ops/*` | 评测驱动（`run_ltlf_eval.sh`）、split/汇总/配对检验/绘图/训练/数据处理 | 15 个脚本 |
| `ltlf_tests/` | 80 个测试（idea1 62 + idea2 18），`pytest` 全过 | — |

## 3. 架构：逻辑的四个插入点

```
离线 M0（一次）：专家轨迹 ──► 任务规范（阶段序列 / 截止期 / 安全包络）
在线 每宏步：
 ① 观测（前置+下视拼图）
 ② 输入侧：残差自动机定当前地标 → 改写 prompt        ← idea1；idea2 的截止期提示
 ③ AeroVLA 前向 + generate（20 token, do_sample=False） ← idea3 改这里的权重 π
 ④ 解码：文本 "NN NN NN" → bin → 反量化 → (fwd,down,yaw)  ← 推理期候选重排可插这里
 ⑤ 执行：仿真运动学（已知解析式）；idea2 的 shield 裁剪 down/fwd  ← 输出侧
 ⑥ 新位姿 → 事件监视 + ρ 更新 → 回 ②
```

**四个插入点**：改输入（②）、改权重（③）、改解码（③→④→⑤，**未做**）、改输出（⑤）。

## 4. 实验矩阵与结果（唯一有效地图：ModernCityMap）

判定口径：SR = 目标 20 m 内并 LAND；OSR = oracle；NE = 末距；SPL = 路径效率；
ESR = **离线独立判据**（`offline.independent_esr`）下完整规范满足；阶段完成率 = 按顺序走完的地标比例。

| 组 | 臂 | N | SR | OSR | NE | SPL | ESR | 阶段完成率 |
|---|---|---|---|---|---|---|---|---|
| idea1 RT（长任务） | 本层 `fixed_order` | 33 | .273 | .364 | 45.8 | .193 | .273 | **.624** |
| | baseline+长指令 | 35 | .229 | .229 | 54.3 | .169 | .229 | .229 |
| idea2 在线 | `stl` | 12 | .333 | .333 | 41.2 | .238 | .333 | .632 |
| | `fixed_order`（对照） | 12 | .333 | .333 | 46.4 | .264 | .333 | .597 |
| 高度下限 6 m | `stlalt` | 12 | .167 | .333 | 46.3 | .082 | .250 | .576 |
| 高度带 [7,12] | `stlband` | 12 | .250 | .250 | 45.5 | .201 | .250 | .542 |
| 后训练 v1（裸底座） | `sa_order` | 10 | .000 | .000 | 95.3 | .000 | .000 | .283 |
| 后训练 v3（裸底座+mix90） | `sa3` | 8 | .000 | .000 | 94.6 | .000 | .000 | .281 |
| **后训练 v4（checkpoint 续训）** | `sa4` | 12 | **.417** | .500 | 31.1 | .300 | .417 | .667 |
| RefineFly 权重 | `rf-a` | 12 | .250 | .250 | 49.2 | .199 | .250 | .250 |
| AeroVLA 权重 | `rf-b` | 12 | .167 | .167 | 55.1 | .135 | .167 | .167 |
| ST 阴性对照 | `st-ours` / `st-base` | 2 / 57 | — | — | — | — | — | 被容器重启打断 |

**配对精确 McNemar（同任务）**

| 对比 | 配对 n | 都成 | 仅 A | 仅 B | 都不成 | p |
|---|---|---|---|---|---|---|
| idea1 vs baseline+长指令 | 33 | 5 | 4 | 3 | 21 | **1.000** |
| stl vs fixed_order | 12 | 2 | 2 | 2 | 6 | **1.000** |
| 高度带 vs 下限 | 12 | 2 | 1 | 2 | 7 | 1.000 |
| sa4 vs 原权重 | 12 | 2 | 3 | 2 | 5 | **1.000** |
| RefineFly vs AeroVLA | 12 | 1 | 2 | 1 | 8 | **1.000** |

**过程指标（真正有差别的地方）**：成功 episode 里"阶段全走完"= 本层 7/9 vs baseline 2/8；
有序完成汇总 57/121 (.471) vs 33/128 (.258)；离线"阶段从未到达"率 .354 vs .544；ρ_chain 中位 −14.4 vs −32.4 m。

**STL 判据一致性**：8 组、上百条 episode，`STL chain 的布尔值 ≡ LTLf 离线判据`，一致率 **1.000**（前提证据）。

## 5. 设计复核：11 条风险 / 待确认（按重要性）

1. **事件半径 12 m ≠ 成功半径 20 m**。spec 用 `at(g,12 m)`，SR 用 20 m → 存在"SR 成功但规范不满足"的合法情形（实测有：`6badc126`、`982c8942` 成功但阶段 0/4）。**论文必须同时说明两套口径**。
2. **M0 是启发式黑箱**：`max_landmarks=3`、`MIN_PAIRWISE_SEPARATION_M=24 m`（=2×半径，避免一点命中两标）、`OFF_LINE_THRESHOLD_M=12 m`。阈值一换，RT/ST 数量就变 → **需要敏感性说明**。
3. **`st_rt` 只是"顺序可检验"的代理**：`RT if n_off_line>0`，离线路 13 m 与 40 m 同归 RT，**未控制难度**。
4. **shield 是逐轴裁剪、不是联合优化**：`down` 与 `fwd` 分开裁 → 会出现"不许下降但仍前进"= 贴地飞（这正是 §4.6 的教训）；正解是 bin 邻域内的**候选重排（联合）**。
5. **M5 的 gate / takeover：gate 有行为、takeover 完全没触发**（此前"两者都是 0"的说法要更正）。
   实测：`w1-g5` 在 1/33 个 episode 里抑制了 2 次模型的 LAND 请求（`93481a81`，抑制后仍碰撞失败）；
   `stl-a` 抑制 2 次、`sa4-a` 抑制 4 次；**`takeovers` 在所有臂里都是 0**。
   → gate 的效果**样本太少无法判定**（n=1~4 个 episode）；takeover 从未触发，**不能写进贡献**。
6. **STL 两个指标容易混**：包络满足率 = 整条 episode 是否始终在带内（对单次越界极敏感）；带内时间占比 = 步数比例。报告里必须并列注明。
7. **评测非确定性来自 spawn/物理 settle**，不是随机种子；NYC 上 settle 让起点掉到地下（数据集 2.9–5.2 m → 实测 −5.3~+21.7 m，同任务两臂不一致）→ **新图必须先做"起飞快检"**（第 0 步位姿 == 数据集起点 ±1 m）。
8. **split 的 `frame` 字段被 loader 忽略**（`env_uav.py:116-126` 用 `set()` 去重）：2873 条记录 → **124 个 episode**，全部从第 0 帧起飞。
9. **idea3 数据只来自 MCM 训练轨迹**（594 条 → 13,715 条样本），v4 只用 3k/200 步；**从裸底座训必塌**（v1 2/139、v3 10/105 种动作、SR 0），必须 `--init-adapter` 续训。
10. **loss 不是策略质量的代理**：v1 loss 6.20→0.82 同时策略退化成常量输出；判据应用**动作多样性**（1 条 episode、4 分钟）。
11. **GPU 共享、2 臂是安全上限**：容器内存上限 240 GB 不是瓶颈；真正瓶颈是显存被别人占（21↔79 GB 波动），本周 4 次 CUDA OOM。**不要再用 `/dev/shm` 常驻模型副本**（15 GB 不必要，pfs 空闲时直接加载即可）。

## 6. 未解决（按优先级）

| P | 问题 | 判据/做法 |
|---|---|---|
| **P0** | idea1 拉回初始预期：SR 不降、阶段完成率或 OSR 提升 | 配对 p<0.05 或方向稳定 |
| **P0** | NYC 起飞位姿 | 起飞快检 gate（±1 m）；不通过就不跑 |
| **P0** | 起飞段净空（前 N 步爬升 + 近场净空 + 可达性） | 碰撞率↓、SR↑ |
| P1 | ST 阴性对照补完（base 剩 32、ours 剩 87） | 回答"ST 上是否打平" |
| P1 | 推理期方案：候选重排（K 条采样 + ρ 打分） | 先做 30 分钟探针看候选多样性 |
| P1 | idea2 定位：保留补充层 vs 做完整并行 STL | 先 ≥30 条/臂比一次 |
| P2 | 多地图（TI/NYCEM/ModularPark） | 每图 ≥15 条/臂，先过快检 |
| P2 | 全量后训练（13,715 条）+ Stage B（DPO） | 需 ~7 h 独占 GPU |

## 7. 复现命令

```bash
R=/mnt/kai_ckp/yiding/MaxHE; PY=$R/envs_py/aerovla_py310/bin/python
# M0 任务 / 显式子集
$PY -m ltlf_logic.tasks --dataset-root AeroVLA/dataset_raw/ --spawn-area-json AeroVLA/data/meta/map_spawnarea_info.json \
    --split-json AeroVLA/data/uav_dataset/seen_valset_splits/ModernCityMap.json --map ModernCityMap \
    --split-name seen_valset --out-dir work/tasks/
$PY ops/make_split.py --split ... --tasks work/tasks/tasks_ModernCityMap_seen_valset.json --select rt:12 --out work/splits/rt12.json
# 单臂评测（两臂并行就换 PORT）
TAG=x MODE=fixed_order PORT=32100 GPU=0 STEPS=80 SPLIT=work/splits/rt12.json \
  TASK_JSON=work/tasks/tasks_ModernCityMap_seen_valset.json bash ops/run_ltlf_eval.sh
# 任意配置覆盖（消融）
LTLF_EXTRA_ARGS="--ltlf-config-stl_alt_min_m=6.0"   # 加到上一行末尾
# 离线 STL 打分 / 汇总 / 配对 / 绘图
$PY -m ltlf_logic.stl_tasks score --logs $R/ltlf_logs/x --tasks work/tasks/tasks_ModernCityMap_seen_valset.json \
    --dataset-root AeroVLA/dataset_raw/ --out work/results/stl_off_x.json --label x
$PY ops/finalize_results.py        # 表 A/B/C/D 一次出
$PY ops/paired_compare.py --a TAG1 --b TAG2 --metric success
$PY -m pytest ltlf_tests/ -q       # 80 passed
```

## 8. 文件索引

| 用途 | 路径 |
|---|---|
| 周报（对外） | `E:\MaxHE\MyData\MSD\周报\week3-2026.11.27-maxhe.md` |
| 周报全文+附录 A/B | `work/weekly-report-2026-11-27.md` |
| 晨报 | `work/morning-report.md` |
| 组会材料 | `work/group-meeting-2026-11-28.md` / `.pptx` |
| idea 级设计 | `work/idea2-stl.md`、`work/idea3-posttrain.md`、`E:\...\work\idea1.md` |
| 项目级计划 | `work/plan-v2.md` |
| 实现记录（含全部踩坑） | `work/implementation-log.md`（679 行） |
| 结果汇总（自动生成） | `work/results/final_summary.md`、`stl_off2_*.json`、`fig_altitude_sr.png`、`fig_arms_sr.png` |
| 扩展摘要骨架 | `work/extended-abstract-skeleton.md` |

## 9. 术语

| 词 | 含义 |
|---|---|
| **M0–M5** | 逻辑层的六个阶段：M0 规范构建 / M1 残差自动机 / M2 规划分解 / M3 prompt 注入 / M4 事件监视 / M5 门控与接管 |
| **ST / RT** | 短任务（地标都在直线上，89 条）/ 长任务（至少一个地标偏离直线，35 条） |
| **ESR** | 完整规范（含顺序）满足，用**离线独立判据**算 |
| **ρ（鲁棒度）** | STL 的定量裕度，单位米；>0 表示满足 |
| **chain / schedule** | STL 的两套规范：chain = 与 LTLf 等价的顺序约束；schedule = 带截止期的版本 |
| **sr∧esr** | 同时成功且满足规范（论文主指标之一） |
