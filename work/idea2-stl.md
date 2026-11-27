# idea2：STL 定量鲁棒层（设计 / 实施 / review）

> 本文是 **idea 级**文档：只回答"为什么做、科学上是否成立、怎么验证"。
> 项目级排程见 `work/plan-v2.md`；连续实现记录见 `work/implementation-log.md`。

## 0. 一句话

把 idea1 的**布尔**事件层升级为**采样时刻 STL**：同一条规范同时给出"是否满足"和"离违反还有多少米"，
并补上 LTLf 表达不了的两件事——**时间（截止期）**与**度量安全包络**；消费端仍然只是"改写提示词 + 投影动作"，不训练、不引入优化器。

## 1. 为什么需要第二个点（问题定位）

idea1 的 LTLf 层能表达"先经过 L0，再到 L1，最后到 g"这类**顺序**约束，但它有三个结构性缺口：

| 缺口 | LTLf 的表现 | 后果 |
|---|---|---|
| **没有裕度** | 12 m 半径内外是 0/1 跳变 | "擦边通过"与"从容通过"无法区分；临近失败无法预警 |
| **没有时间** | `F` 无截止期，"最后到达"也算满足 | 长任务可以无限拖延；专家演示的**节奏**信息完全浪费 |
| **没有度量安全** | 只能写"是否越界"的离散谓词 | 高度/速度这类连续量只能用布尔阈值，无法给出"离违反多少米" |

**离线实测证据（见 §7）**：idea1 的层把"阶段从未到达"从 48.3% 压到 13.8%，但**到达者中 34.5% 是迟到的**
（超过专家时间窗）——**剩下的缺口恰好就是"时间"**。这就是 idea2 的立论点。

## 2. 相关工作怎么用 STL（结论，决定了我们的边界）

| 论文 | STL 的角色 | 消费端 | 我们能否照搬 |
|---|---|---|---|
| Logic-VLA | STL 公式作为**条件输入**（语法图 GCN → spec token），NL→STL **不做**（公式从模板采样） | 策略本身 | 编码器部分不可搬（要重训）；"STL 作为策略输入"的思路可搬 |
| LLM-Enabled UAV NL Nav via STL Translation & Repair | NL→STL 翻译器（Qwen2.5-0.5B + LoRA SFT + GRPO，精确匹配 98%），冲突修复靠 **MILP 不可行→IIS→LLM 只选松弛方向** | **MILP 轨迹合成**（开环） | 不可搬：我们没有约束集，永不产生 infeasible；也没有低级优化器 |
| STeP（原 STL-VLM，arXiv 2607.18580） | VLM 产出 STL → 时间切片成子任务 | **10 Hz MPC 的鲁棒度目标项 −λρ**（单积分器 + RRT） | 不可搬：我们无动力学模型、无 MPC；且它是**机械臂**不是 UAV |

**由此冻结的边界（写进论文要明确声明）**：我们做的是 **STL-lite**——
✅ 做：采样时刻 STL 的**规范构建 + 在线鲁棒度监控 + 截止期压力提示 + 安全包络动作投影**
❌ 不做：NL→STL 翻译器、规格修复（repair）、鲁棒度优化/MPC、可微鲁棒度损失

> 负面证据（Logic-VLA p.7 §V-B）：把可微鲁棒度直接加进损失，2× 权重时满足率 78.8% 但 NL 任务从 68.5% 崩到 45.0%。
> 这条反过来支持我们的选择：**鲁棒度只用于监控与筛选，不进梯度**。

## 3. 设计

### 3.1 时间基：为什么是"宏步"而不是秒

baseline 的动作是连续 3 维 `(fwd, down, yaw)`，但**执行时钟是离散且非均匀的**：
`duration = fwd / 1.0` 秒 + 0.5 s 停顿，`|yaw| ≥ 0.25 rad` 时先原地转；每个宏动作只产生**一个**新位姿
（`AirVLNSimulatorClientTool_AeroVLA.py:388-395` 把同一帧状态复制 5 次）。

所以在实时语义下写 STL 是虚构的。我们采用**整数步的采样时刻 STL**：`t ∈ ℕ`，一步 = 一个宏动作。
好处：截止期可以直接和专家演示的"经过该地标的时刻"比较；坏处：与真实秒数不是线性关系（**已在局限里声明**）。

### 3.2 规范（每个任务两条，用于对照）

* `chain`（= idea1 的 LTLf 公式，但读成 STL，无截止期）：
  `F(L0 ∧ F(L1 ∧ … ∧ F(g)))`，区间无界。
  → 它的**布尔**满足性必须与 LTLf 监视器一致（§6 已验证 39/39）。
* `schedule`（新增能力）：
  `∧_k F_[lo_k, hi_k](at(L_k, 12 m))`，最后一项是 `g`。
  窗口由**专家到达时刻**推出：`arrival_step_k = (expert_route_length × frame_k / expert_frames) / STEP_LEN`，
  `lo_k = max(prev_lo, arrival_k − slack)`，`hi_k = arrival_k + slack`，`slack = max(2, 0.35 × arrival_k)`。
  `STEP_LEN = 4.3 m/步`（从真实 rollout 实测中位数得到，三轮实测 4.24–4.41）。
  每个窗口上线前**先在专家轨迹上自检**，不满足则放宽并记录（rule-based 修复，不涉及 LLM）。
* 安全包络（数据驱动）：
  `G(alt_p01 ≤ 高度 ≤ alt_p99)`，由**专家真实飞行轨迹**的高度分位数标定
  （ModernCityMap：1.72 m – 33.64 m，中位 9.84 m）。

### 3.3 三个在线机制（都可单独消融）

1. **鲁棒度监控**：每步算 `ρ_safety`（到包络边界的距离，米）与每阶段最佳裕度 `ρ_stage = 12 − d_min`。
   记录 `ρ` 轨迹、每阶段 `best_margin/best_step/是否在窗口内/是否迟到/是否从未进入`。
2. **截止期压力提示**：当前阶段剩余预算 < 窗口宽度的 60%（且 ≥2 步）时，在阶段描述后**追加一句**
   `"You only have about N more steps to reach it."`；每步最多一次、每个截止期最多一次。
   基线提示词的两个锚点（`<image>\nFly … and find the target.` / `\nAction: `）保持不变。
3. **安全包络动作投影（shield）**：动作用 `(fwd, down, yaw)` 表示，下一步高度 = `−(z+down)`，
   所以"保持包络"就是 `down` 的区间约束：`down ∈ [−alt_max−z, −alt_min−z]`；
   同时把 `fwd` 裁到 `[0, 8] m`。这是**精确的最小投影**：不需要梯度、不需要额外模型调用、天然幂等。

### 3.4 明确不改变的东西

- 阶段推进仍然由 **LTLf 残差自动机**驱动（已证明正确），STL 不重写阶段机 → 风险隔离，两个点可独立消融。
- 不改 baseline 任何文件；`AeroVLA/` 与 `baseline-v0` 逐字节一致。
- 默认 `mode=stl` 时 gate/takeover 与 `fixed_order` 相同（便于单变量比较）。

## 4. 实施计划（按阶段，每阶段自带检查点）

| 阶段 | 内容 | 检查点（通过才进下一阶段） |
|---|---|---|
| S1 语义核 `stl.py` | AST + min/max 鲁棒度 + 部分（在线）上下界 | 手算用例逐条对齐；`bounds()` 在截断轨迹上给出 `(≤, ≥)` 而非伪造确定值 |
| S2 规范构建 `stl_tasks.py` | chain/schedule/安全包络 + 脚本级离线打分 | 窗口单调；专家轨迹自检通过；`chain` 布尔 ≡ LTLf |
| S3 在线层 `stl_layer.py` | 监控 + 截止期提示 + shield | shield 幂等；投影后高度必在包络内；提示只在窗口尾段触发一次 |
| S4 集成 `eval_loop.py` | `mode=stl`，逐步记录 STL 字段，episode 摘要含 STL 段 | 旧模式行为不变（回归：G0/G1 提示词逐字节不变） |
| S5 离线数值 | 对**已跑完**的 rollout 重新打分（不占 GPU） | 8 条/组也能给出结论；链式一致性 100% |
| S6 在线小量 | `stl` vs `fixed_order` 同 12 条 RT | 记录 shield 触发率、警告次数、ρ 改善 |

## 5. 实现清单（文件 / 行数）

| 文件 | 作用 |
|---|---|
| `ltlf_logic/stl.py` | 采样时刻 STL：`Top/Bot/Pred/Not/And/Or/F/G/Until`、`robustness`、`bounds`（在线上下界）、谓词构造器（`at`/`altitude_range`/`displacement_max`） |
| `ltlf_logic/stl_tasks.py` | `build_spec`（chain+schedule+safety）、`evaluate_trace`、`expert_envelope`（专家飞行高度分位）、`derive_step_len_from_logs`、`run_offline`、CLI `build/score` |
| `ltlf_logic/stl_layer.py` | `StlLayer`：`observe`（按控制步而非去重位姿计数）、`deadline_hint`、`shield`、`summary` |
| `ltlf_logic/eval_loop.py` | +63 行：`mode=stl` 时实例化、每步 observe、gate 前 shield、阶段切换后提示、每步/每 episode 落盘 STL 字段 |
| `ltlf_logic/runner.py` | `MODES` 加 `stl`；`LTLfConfig` 加 `stl_*` 7 个字段（含 `stl_step_len_m=4.3`） |
| `ltlf_tests/test_stl.py` | 18 个测试：语义、边界、部分界、规范、shield、提示、迟到判定 |

测试总数：idea1 的 62 个 + 本 idea 18 个 = **80 个，全部通过**（`python -m pytest ltlf_tests/ -q`）。

## 6. 交叉验证：STL 的布尔语义 ≡ LTLf 监视器

对 5 组已完成的 rollout（共 39 条 episode）同时算：
`STL chain 的 ρ > 0` 与 `offline.independent_esr()`（有限迹直接语义，用 `(()` 独立的 `sat()` 实现）。
**一致率 39/39 = 1.00**。这排除了"我悄悄改了判据"的可能，也是 STL 层可信的前提。

> 踩坑记录：最初我用 JSONL 里记录的 `accepted` 字段对照，G4 组只有 6/8。
> 查明原因是**模式差异**——`long_instruction`/`single_stage` 只跟踪 `F(g)` 单目标，
> 所以"落地即 accepted"，与完整链无关。**不是 bug，是我的对照变量选错了**；改用它后 100% 一致。
> 这条差异本身也是一个可用的事实：baseline 侧的逻辑判据本来就只到"到没到目标"。

## 7. 离线结果（对已跑完的 rollout 重新打分，0 GPU）

环境：ModernCityMap；事件半径 12 m；`STEP_LEN=4.3`；安全包络 p01/p99 = 1.72/33.64 m；ρ 报告时截断到 ±50 m。

| arm | n | SR | OSR | chain 满足 | schedule 满足 | 安全包络满足 | ρ_chain 中位 | ρ_sched 中位 | 阶段迟到率 | 阶段从未到达率 |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 baseline 单阶段 | 12 | .333 | .500 | .417 | .333 | .75 | −15.77 | −15.77 | .125 | .250 |
| G2 baseline+长指令 (ST) | 8 | .375 | .375 | .375 | .375 | .75 | −37.81 | −37.81 | .050 | .300 |
| G3 本层 fixed_order (ST) | 8 | .625 | .625 | .625 | .625 | .75 | **+4.08** | **+4.08** | .150 | .200 |
| G4 baseline+长指令 (RT) | 8 | .250 | .250 | **.000** | **.000** | .875 | −36.25 | −41.30 | .069 | **.483** |
| G5 本层 fixed_order (RT) | 8 | .625 | .750 | .625 | .375 | .75 | **+0.48** | **−5.06** | **.345** | **.138** |

**读法（三条）**

1. `chain` 与 SR 在 G5 上同为 .625，但 G4 的 chain=.000 而 SR=.250 →
   baseline 能"撞到目标"，但**几乎不可能按顺序经过地标**（chain 中位 −36 m）。
2. RT 组：阶段"从未到达"率 G4 **48.3%** → G5 **13.8%**（3.5×），
   但 G5 的"到达者中迟到"率升到 **34.5%** → **顺序修好了，时间没修**。这正是 §3.2 `schedule` 规范的靶子。
3. `ρ_sched 中位`：G5 −5.06 / G3 +4.08 / G4 −41.30 → 我们的层把"离按时的差距"从几十米压到个位数米。

## 8. 在线实验设计（S6，随 GPU 队列执行）

| 臂 | 模式 | 与谁的差异 | 想回答 |
|---|---|---|---|
| `stl-ctl` | `fixed_order`（idea1） | 基线逻辑层 | 对照 |
| `stl-g5` | `stl`（idea1 + shield + 截止期提示） | 上面一臂 | 加 STL 是否提升 schedule 满足率/安全包络满足率而不掉 SR |

同 12 条 RT episode、同 split、同 STEPS=80、同 GPU、串行两臂（避免相互干扰）。
消融（时间允许）：`stl_shield=0`（只留截止期）、`stl_deadline=0`（只留 shield）。

## 9. Review 记录（自查发现 → 修正）

| # | 问题 | 怎么发现的 | 修正 |
|---|---|---|---|
| 1 | `observe()` 用"不同位姿个数"当步数，悬停时步数不增长，截止期永远不到期 | 写迟到判定测试时结果恒为 False | `observe(i, pose, step=...)` 显式接收控制步；位姿去重只影响轨迹，不影响时钟 |
| 2 | `bounds()` 声称给在线上下界，实现却返回确定值（`F` 在未走完窗口时返回已见最大值） | `test_partial_bounds_are_optimistic_vs_pessimistic` 失败 | 拆成两套语义：`robustness()` = 完整轨迹语义；`bounds()` = 独立 `_online_bounds()` 递归 |
| 3 | 用线上 `accepted` 字段做交叉验证 → 一致率 6/8 | 与 STL 结果不符，逐条回放 | 改用 `offline.independent_esr`；查明 `accepted` 依模式而异（`F(g)` vs 完整链） |
| 4 | 安全包络最初用"地标中心 z"标定 → 把地面高度当飞行高度，包络退化成 0–7.8 m，安全满足率只有 25% | 首次离线打分 `safety_sat=0.25` 明显异常 | 改为读专家 `trajectory_raw_detailed` 的真实飞行高度取 p01/p99 |
| 5 | `ρ` 均值出现 `−inf`（episode 提前结束时窗口整体落在轨迹外） | 首版聚合表里 `rho_schedule_mean: -Infinity` | 报告用截断 ±50 m + 中位数；原始值保留在 `rho_*_raw` |
| 6 | 18 个新测试首轮 6 个失败，其中 3 个是**测试写错**（把点积当 3-D 距离、把 min 当 max） | 逐个手算复核 | 修测试；并在文档里保留"哪些是测试错、哪些是实现错"，避免以后误判 |

## 10. 局限（写论文/汇报必须一起说）

1. **时间单位是宏步**，不是秒：动作执行时长随 `fwd` 变化，无法给出真实时间保证。
2. **截止期是估计**：专家到达帧 → 弧长 → 宏步，用了固定 `STEP_LEN=4.3 m/步` 的换算；专家速度非均匀时会偏。
3. **安全包络来自专家分布**（p01/p99），是"操作包络"而非物理硬限；因此"违反"不等于危险。
4. **样本小**：离线 8–12 条/组、在线 12 条/组，全部是**方向性结论**，未做显著性检验。
5. **未做**：心跳/多速率信号的 STL（我们只有 1 个位姿/宏步）、规格修复、MPC 级消费。
