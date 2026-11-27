# idea3：逻辑对齐后训练（设计 / 可行性 / 实施计划）

> 本文是 **idea 级**文档。项目级排程见 `work/plan-v2.md`；连续实现记录见 `work/implementation-log.md`。
> 状态标注纪律：未经我实测的推断一律标「未验证」。

## 0. 一句话

前两个点都**不动模型**（推理期外挂），代价是"新颖的提示词"与训练分布不匹配；
idea3 把逻辑层**训进模型**：先做**逻辑条件 SFT**（把专家序列改写成"分阶段提示"，零新标注），
再用闭环失败做**偏好对齐**。这是 Logic-VLA 与 RefineFly 都验证过的杠杆，且我们的底座与数据都支持。

## 1. 两篇参考论文的后训练到底做了什么

### 1.1 Logic-VLA（正面：两阶段都涨点）

| 阶段 | 做什么 | 监督信号 | 增量（Seen/UP/US） |
|---|---|---|---|
| Stage 1 | **STL 条件 SFT**：STL 公式经语法图 GCN 编码成 spec token，与图像/语言 token 一起输入；**没有任何 STL 专用损失** | 示教动作（flow-matching 原生损失） | +20.4 / +9.5 / +7.7 pp（满足率）；**NL 任务掉 −3.3/−2.7/−4.1 pp** |
| Stage 2 | **轨迹级 IPO 偏好优化** + 单向 anchor | 偏好对（满足 vs 违反） | 再 +20.3 / +15.3 / +17.5 pp；**NL 任务全部捞回并反超** +1.0~2.3 pp |

**反面证据（关键）**：把可微鲁棒度当损失（`L = L_FM + λρ·L_ρ`），2× 权重时满足率 78.8%，
但 NL 任务从 68.5% **崩到 45.0%**；论文自述"directly maximizing robust semantics alters the policy
behavior beyond what is necessary"。→ **不要走 robustness reward，走偏好对**。

另外两个可直接借的结论：
- **prompt-only 的天花板被量化了**：Text-prompt 基线（公式渲染成文本塞进指令，但**跑完整两阶段后训练**）
  = 66.5 / 70.1；结构化编码器 = 82.0 / 74.8。我们 idea1/2 的"分阶段提示词"正落在前者这档。
- 他们用 IPO 是因为 π0.5 是 flow-matching、动作似然比算不起；**我们是离散 token，序列 log-prob 直接可算**，
  可以用严格 DPO，超参更少（这是我们的相对优势，不是劣势）。

### 1.2 RefineFly（正面：同一个底座上的 RL 后训练，且权重已开源）

- 底座 = **openvla-7b + LoRA**，跑在 **AeroVLA 评测管线**上 —— 和我们是**同一个 baseline**。
- 方法：token 级 PPO + 失败记忆 `ξ=(初始化,场景,失败类型,状态,重试数)` + 两阶段场景课程 + 参考策略 KL(β=0.03)。
- 结果：SR 56.33 / 42.97 / 59.72（Seen / Unseen Map / Unseen Object），对 AerialVLA **+8.37 / +5.39 / +3.12 pp**，
  约 30% rollout 预算、8×A100 21 h。
- **开源状态（本轮实测，2026-11-27）**：代码未开源；但 HF 有匿名适配器 `refinefly/refinefly`
  （462 MB，`r=64, alpha=128, dropout=0.05, base=openvla/openvla-7b`，与论文 Table 6 逐项一致）。
  → **可下载、可直接跑**。已下载到 `$R/models/refinefly`（397 MB），并列入实验队列作为对照臂。
- 关键消融：**失败重放单用会伤泛化**（Unseen Map −6.37 pp），必须配 KL + 两阶段课程才翻正。

## 2. 我们这个 baseline 能不能做后训练（可行性论证）

| 条件 | 事实（本轮实测） | 判定 |
|---|---|---|
| 有没有训练入口 | `AeroVLA/src/train_aerovla.py` + `scripts/train.sh`，`peft` LoRA，bf16 | ✅ |
| 超参 | LoRA `r=64, α=128, dropout=0.05`，lr `2e-4`，micro-bs 2，grad-accum 8，5 epoch | ✅ 已冻结可复现 |
| 训练数据 | `data/aerovla_train_dataset.json` = **410,450 条**样本（`traj_rel_dir/img_name/instruction/label{fwd,down,yaw}`） | ✅ 够 |
| 动作标签 | 连续 3 维，训练时量化成 3 个 bin token（`_quantize_action`），推理再反量化 | ✅ 与逻辑层动作投影兼容 |
| 有没有现成"逻辑条件"数据 | ❌ 原始 instruction 是单目标模板 | ⚠️ **需要自己生成（这正是本 idea 的核心工序，见 §3.1）** |
| 算力 | GPU0 80 GB 共享（他人占 21.5 GB），当前被评测队列占满；单卡 LoRA 可跑 | ⚠️ 今日只能冒烟，全量需过夜 |
| 时间成本估算 | 40k 样本 × 1 epoch ÷ (bs 2×accum 8=16) ≈ 2500 优化步；A800 上约 1.2–1.8 s/步 → **≈50–75 min** | ✅ 可行 |
| 评测闭环 | 已有 5 臂评测脚本 + 12 图/多地图 split | ✅ |

**结论：可以做，且比 Logic-VLA 更简单**（我们有连续动作标签 + 现成 LoRA 管线 + 可算 log-prob）。
**唯一真实风险是算力排队**，不是方法。

## 3. 设计

### 3.1 Stage A：逻辑条件 SFT（零新标注）

**思路**：不采新数据、不改标签，只**改写指令文本**。
对专家轨迹的每一帧 `i`，先由 M0 任务的"地标到达帧"判断专家此刻处于哪个阶段 `k(i)`，
然后生成**分阶段提示**（与推理期 `mode=stl/fixed_order` 完全同构）：

```
<image>\nFly {dir_text(pose_i → L_k)}and find the target. {desc(L_k)}\nAction: 
```

`dir_text` 用 baseline 自己的语义方向函数（import，不复制），保证与推理期逐字节同构；
`desc` 用 M0 已生成的 40–60 词场景描述。标签仍是原始示教动作。

**为什么这一步是关键**：idea1/2 的增益损失在"分阶段提示是分布外输入"。这一步把分布外输入变成分布内，
是**唯一能同时保住 NL 能力和逻辑能力**的做法（Logic-VLA 的 Stage 1 增量正是来源于此）。

### 3.2 Stage B：闭环失败对齐（两种实现，按算力选）

**B1（首选，不需要新 rollout）— 从已有 rollout 造偏好对**
同一任务下，取"满足 chain/schedule"的 episode 为正样本、"阶段从未到达/迟到"的为负样本，
按**逐帧状态最接近**配对（用 `git diff` 相同的位姿距离最近邻），得到 `(state, y_w, y_l)` 三元组；
用**序列 log-prob 的 DPO**（β=0.1）训练，loss 只在动作 token 上算，加 KL anchor 到 Stage A 权重。

**B2（更强，需要一点算力）— 边界纠正**
从闭环日志里筛"阶段已切换但动作仍在飞离/擦边"的步（`stage_switch=1` 且下一阶段 `ρ_stage<0`），
用专家轨迹最近帧的动作作为纠正标签，得到几百条 `(state, a_expert)`，做**小学习率 SFT 刷新**（~200 步）。

> 依据：RefineFly 的消融说明"失败重放单用会掉泛化"，所以我们**必须**保留 KL anchor 到 Stage A，
> 并且**只做一次轻量刷新**，不做多轮 PPO。

### 3.3 与 idea1/2 的关系（论文故事）

```
任务逻辑层（外挂，训练无关） ─┬─ idea1 LTLf：顺序
                              └─ idea2 STL ：裕度 + 截止期 + 安全包络
                                        │
                                        ▼  提供"分阶段提示"与"失败信号"
                         idea3 逻辑对齐后训练（把上述信号训进模型）
                                        │
                                        ▼
                            推理时只需轻量监控（甚至可去掉 gate）
```

三点形成"**外挂 → 定量 → 内化**"的闭环，且每点都能单独消融。

## 4. 实施计划

| 阶段 | 内容 | 产物 | 检查点 |
|---|---|---|---|
| P0 | 数据管线：生成分阶段提示 SFT 数据 | `ops/make_posttrain_data.py` → `work/posttrain/sft_stageA_*.json` | 抽样 20 条人工核对：提示词逐字节符合 `build_stage_instruction` 格式；`dir_text` 与 baseline 一致 |
| P1 | 训练入口（**不改 baseline**） | `ops/train_aerovla_ltlf.py`（复制并参数化 `SPLIT_JSON/OUTPUT_DIR/步数`） | 能加载 7B + LoRA、能跑 10 步不 OOM |
| P2 | 冒烟：40 样本 × 20 步 | loss 曲线下降、checkpoint 可保存/可被评测脚本加载 | loss 低于随机初始化基线 |
| P3 | 全量 Stage A（过夜） | `checkpoints/aero_vla_ltlf_stageA` | 在 12 条 RT 上评测：SR 不掉、ESR/schedule 提升 |
| P4 | Stage B（B1 或 B2） | 偏好对数据 + DPO | 相对 Stage A 再提升，且 NL（单阶段）不退化 |

## 5. 风险与回退

| 风险 | 判据 | 回退 |
|---|---|---|
| 算力排队 | 评测队列占满 GPU0 | 只交付 P0 数据管线 + P1/P2 冒烟；全量训练排到夜间/周六 |
| 图像 I/O 成瓶颈（pfs） | 冒烟时 step 时间 > 3 s | 只用单图子集 + `image_feature_path` 预提取特征 |
| Stage A 过拟合到分阶段提示 | 单阶段（`single_stage`）SR 掉 >5 pp | 混采：50% 原始指令 + 50% 分阶段指令（最稳） |
| Stage B 掉泛化 | Unseen map（ModularPark）掉 | 按 RefineFly 教训：加 KL anchor、减步数、只训一次 |

## 6. 当前进度（滚动更新）

- [x] 参考论文结论提取（Logic-VLA / RefineFly / STL-repair 三份报告在 `work/lit/`）
- [x] 可行性事实核实（训练脚本、410k 数据、LoRA 超参、动作量化、RefineFly 权重可下载）
- [x] RefineFly 权重已下载（`$R/models/refinefly`，397 MB）并列入评测队列
- [x] 数据管线 `ops/make_posttrain_data.py`（见 implementation-log）
- [ ] 训练入口 `ops/train_aerovla_ltlf.py`
- [ ] 冒烟（P2）
- [ ] 全量 Stage A / Stage B（受算力约束，排到夜间）

## 7. 局限

1. 410k 样本是全图全帧，**没有按"逻辑阶段"重新平衡**；Stage A 的提示分布仍以单目标为主（需 50/50 混采）。
2. 偏好对来自**少量** rollout（十几到几十条/任务），DPO 容易过拟合 → 必须小步数 + KL anchor。
3. 未验证：分阶段提示是否会让模型"过度依赖描述"而忽视视觉（需 Unseen Object 地图检验）。
4. 所有训练结论都要在 **未参与训练的地图**（ModularPark）上复核，否则无法区分记忆与泛化。
