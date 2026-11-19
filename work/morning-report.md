# 晨间汇报（v1 实现，autonomous session，2026-11-20 03:00–05:00+）

## 0. 一句话

**阶段 0–4 全部做完并自测通过；阶段 5 完成 ModernCityMap 六组对照（RT 8v8 与 ST 8v8），
方向与设计一致（RT：baseline+长指令 0/8 满足路线 vs ours 5/8；ST：0.375 vs 0.625）；
阶段 6（其他地图/消融/噪声敏感性）没有做；阶段 5 的探针 `d_ε` 也没做。**
代码、日志、结果都在服务器仓库 `dev` 分支（`aa9dd6a`），已推送到 GitHub。

关键数字（同一批 RT episode，独立离线判定）：

| 组 | SR | ESR | **SR∧ESR** | 阶段完成率 |
|---|---|---|---|---|
| baseline + 长指令（G4） | 0.250 (2/8) | 0.000 | **0.000** | 0.448 |
| ours 固定顺序（G5） | 0.625 (5/8) | 0.625 | **0.625** | 0.854 |

无害性（12 条轨迹，同 episode）：baseline 与 ours 单阶段退化的 **SR/OSR/ESR/SR∧ESR 完全一致**
（0.333 / 0.500 / 0.417 / 0.333）。

---

## 1. 总体状态

| 阶段 | 内容 | 状态 | 证据 |
|---|---|---|---|
| Step 0 | 版本管理、baseline-v0、GitHub 推送 | ✅ 完成 | 见 §7；`baseline-v0` tag + `main`/`dev` 分支已在 GitHub |
| 阶段 0 | 调用链审计 + baseline 冒烟 + 风险清单 | ✅ 完成 | `work/stage0_audit.md`；冒烟 20 步、122.3→48.9 m、101 帧落盘 |
| 阶段 1 | 纯逻辑内核（M1/M4 逻辑部分） | ✅ 完成 | 62 个单测全过；残余状态数 2/3/4/8/4（与 idea1 的 3/4/8/3 基本一致，G 例差 1，已说明） |
| 阶段 2 | M0 离线规格构造 + L0 离线回放 | ✅ 完成 | 123/124 可用任务；123/123 专家轨迹通过自身公式；497/497 负例被拒；0 判定不一致 |
| 阶段 3 | 单阶段退化 + baseline 无害性 | ✅ 完成 | 12 条轨迹：G0 vs G1 离散指标全一致；接口级单测证明单阶段 prompt 逐字节相同 |
| 阶段 4 | 最小多阶段闭环（固定顺序） | ✅ 完成（2 条 RT 多阶段） | fake-executor 5 例全过；真机 4 阶段 episode 43 步成功 + ESR ✓ |
| 阶段 5 | ModernCityMap 第一版对照 + 探针 | ⚠️ 对照完成、探针未做 | RT 8v8 与 ST 8v8 主对照均完成（六组指标在 `work/results/all_arms_metrics.json`）；**探针 `d_ε` 没做** |
| 阶段 6 | 可选扩展（其他地图/消融/噪声） | ⬜ 未做 | 时间与资产限制，明确未做 |

---

## 2. 关键结果（全部为本机实测，非预期值）

### 2.1 无害性（阶段 3，`work/results/stage3_metrics.json`，N=12）

| 组 | SR | OSR | NE (m) | SPL | ESR | SR∧ESR |
|---|---|---|---|---|---|---|
| G0 原版 baseline | 0.333 (4/12) | 0.500 (6/12) | 38.66 | 0.265 | 0.417 | 0.333 |
| G1 ours 单阶段退化 | 0.333 (4/12) | 0.500 (6/12) | 41.34 | 0.301 | 0.417 | 0.333 |

离散指标逐项相同；NE/SPL 的差异（2.7 m / 0.036）与下面测到的仿真非确定性同量级。
G1 的 `suppressed_stops=0`、`takeovers=0`、`stage_switches=0`（该模式本就关闭门控与接管）。

### 2.2 主对照（阶段 5，RT 组 8 条，`work/results/stage5_rt_metrics.json`）

| 组 | episodes | SR | OSR | NE (m) | SPL | ESR | SR∧ESR | 阶段完成率 | 阶段切换 |
|---|---|---|---|---|---|---|---|---|---|
| G4 baseline+长指令 | 8 | 0.250 | 0.250 | 39.65 | 0.227 | **0.000** | **0.000** | 0.448 | — |
| G5 ours 固定顺序 | 8 | **0.625** | **0.750** | **13.20** | **0.470** | **0.625** | **0.625** | **0.854** | 20 |

- 在线逻辑层与离线独立判定器（只用保存的位姿重建事件、直接 LTLf 语义）**一致率 8/8**；
- G5 成功的 5 条事件序列均严格按公式顺序 `L0→L1→L2→g`（含重复进入不影响）；
- G4 的 2 条"到点"轨迹分别漏了 L2 或没在 L2 之后进入 g 的 12 m 圈 → ESR 严格判 0；
- **G5 的门控/兜底在这批运行里 0 次触发**（模型不会把中间地标误判为终点），
  所以 G5 的收益目前只能归因于"逐阶段引导点 + 阶段描述"，**不能**归因于 M5。

### 2.3 逻辑内核与离线规格（阶段 1–2）

- 残余状态数：`F(g)`=2、`F(a∧F g)`=3、`F(a∧F(b∧F g))`=4、`F a∧F b∧F g`=8、
  `G¬z∧F(a∧F g)`=4（idea1 记 3，差 1 的原因是我们不做 `∧` 对 `∨` 的分配律，见实现记录）。
- 反向验证：`prog+accept` 与独立有限迹语义在 >1000 条小迹上逐个一致（含多事件采样点）。
- M0：123 条可用任务（2/3/4 阶段 = 59/33/31；ST 88 / RT 35），地标间距全部 >24 m，
  每阶段描述都是数据集里的真实 40–60 词文本（记录来源 seq）。
- L0：专家轨迹 123/123 接受；497/497 人工负例（漏事件/漏终点/交换/逆序/违反 G）被拒。

### 2.4 ST 组（G2 baseline+长指令 vs G3 ours，8 条，`work/results/stage5_st_metrics.json`）

| 组 | SR | OSR | NE (m) | SPL | ESR | SR∧ESR | 阶段完成率 |
|---|---|---|---|---|---|---|---|
| G2 baseline+长指令（ST） | 0.375 | 0.375 | 47.02 | 0.329 | 0.375 | 0.375 | 0.625 |
| G3 ours 固定顺序（ST） | **0.625** | **0.625** | **28.57** | **0.594** | **0.625** | **0.625** | **0.812** |

**这组结果支持 idea1 的判别设计**：ST 任务的地标在"起点→终点"直线附近，直飞会**偶然**满足
部分中间地标 → baseline+长指令的 ESR 从 RT 的 0.000 抬到 0.375；RT 任务必须真的绕行 → 它的 ESR 是 0。

| 同 8 条 episode 对比 | baseline+长指令 | ours |
|---|---|---|
| ST（顺路地标） | 0.375 | 0.625 |
| RT（偏线地标） | **0.000** | 0.625 |

差距在 RT 上最大 → 改善来自"按路线逐阶段执行"，而不是"任务整体变简单"。
但样本只有 8+8 条，仍是方向性结论。

---

## 2.5 数字汇总（一张表）

| 组 | split | N | SR | OSR | NE | SPL | ESR | SR∧ESR |
|---|---|---|---|---|---|---|---|---|
| G0 baseline | 混合 12 | 12 | 0.333 | 0.500 | 38.66 | 0.265 | 0.417 | 0.333 |
| G1 ours 单阶段 | 混合 12 | 12 | 0.333 | 0.500 | 41.34 | 0.301 | 0.417 | 0.333 |
| G2 baseline+长指令 | ST 8 | 8 | 0.375 | 0.375 | 47.02 | 0.329 | 0.375 | 0.375 |
| G3 ours 固定顺序 | ST 8 | 8 | 0.625 | 0.625 | 28.57 | 0.594 | 0.625 | 0.625 |
| G4 baseline+长指令 | RT 8 | 8 | 0.250 | 0.250 | 39.65 | 0.227 | 0.000 | 0.000 |
| G5 ours 固定顺序 | RT 8 | 8 | 0.625 | 0.750 | 13.20 | 0.470 | 0.625 | 0.625 |

（ST 的 G3 与 RT 的 G5 不是同一批 episode，不能横着比；正确的纵向对比见 2.2 / 2.4）

---

## 3. 自 review：逐阶段验收对照（work1.md 的验收条款）

| 阶段 | work1.md 验收要求 | 我的结论 | 证据/缺口 |
|---|---|---|---|
| 0 | 一页调用链与风险清单；baseline 闭环可跑可存可读；接入点明确；可回退 | ✅ | `work/stage0_audit.md`；冒烟产物 101 帧；4 个注入点；`baseline-v0` tag |
| 1 | 单测覆盖 7 条必测；每步事件/残余/接受可读；状态数有限 | ✅ | 62 例全过；JSONL 每步记录 `events/residual/accepted` |
| 2 | 专家任务全通过离线接受；乱序/漏事件/违规被拒；抽查一致；给数量与跳过原因 | ✅ | 123/123、497/497；跳过 1 条有明确原因（专家末帧 18.9 m>12 m） |
| 3 | prompt/动作一致（接口测试）+ 少量 episode 对照 SR/OSR/NE/SPL；无阶段误切/误拦 LAND | ✅ | 接口逐字节一致；12 条轨迹离散指标相同。**注意**：仿真非确定性使"动作逐位相同"无法在仿真里验证（已证明不可复现） |
| 4 | 先 mock 验证阶段机与 LAND 处理；再 1 条简单多阶段；再少量代表性；能按序完成、不把中间 LAND 当结束、指标可算 | ✅（部分） | fake-executor 5 例；真机 2 条 RT（1 成功按序完成，1 第 1 步碰撞）。**中间 LAND 抑制在真机未被触发**（模型不误停），只有单测覆盖 |
| 5 | 独立离线指标脚本可重算；至少一组现代城对照+样本数+失败原因；能回答三个问题；不美化 | ⚠️ 部分 | RT 组完成（见 §2.2）；ST 组进行中；**探针没做**（无 d_ε 估计） |
| 6 | 其他地图/消融/噪声 | ⬜ 未做 | 明确未做 |

---

## 4. 已验证 / 只做了离线验证 / 尚未验证

**已验证（有实测输出）**
- baseline 闭环可跑；`baseline-v0` 快照与原目录**逐字节一致**（diff -r 通过）；
- LTLf 内核语义（对独立语义互验）、残余状态数、换序判据、`G` 不可用 `¬` 代替的反例；
- M0 规格构造 + L0 回放（123 条真实 episode）；
- 单阶段退化与 baseline 离散指标一致（12 条）；单阶段 prompt 与 baseline 逐字节相同；
- 多阶段固定顺序闭环可完成 4 阶段任务（真机 1 条 + 后续 G5 的 5 条）；
- RT 组 G4 vs G5 对照；离线指标脚本可独立重算（在线/离线一致率 1.0）。

**只做了离线/单测验证（没有真机证据）**
- LAND 门控与兜底接管：fake-executor 5 例 + 门控单测；**真机 0 次触发**；
- `G` 型安全约束（全程避让）在真机上的表现：只有构造/回放与单测，**没有跑真机 G 组**；
- 换序（Swap）、粒度合并：只有判据/算法实现与单测，**没有做真机消融**。

**尚未验证**
- other maps（无数据）、多种子稳定性、探针 `d_ε`（AeroVLA 跟随中途点的横向偏差）、
  `σ∈{0,1,3,6} m` 位置噪声敏感性、`λ`（本就是 v1 不启用）。

---

## 5. 已知问题与风险

1. **仿真不可逐位复现**（实测：同配置连跑两次，reset 后 z 相差 0.36 m，首帧图像 md5 不同）。
   → 所有对照只能统计比较；无害性用接口级证明 + 离散指标对照。**若要更强的无害性结论，需要多种子重复。**
2. **样本量小**：RT 8 条、ST 8 条、G0/G1 12 条。方向清楚，但**不能宣称统计显著**。
3. **门控未在真机被检验**：这批 episode 里模型不误停，M5 的主要卖点尚无真机数据。
4. **G4（baseline+长指令）的公平性**：长指令塞进英文 `obj_desc` 槽位，与训练分布不一致
   （训练集是中位 43 词的物体描述，长指令 + 终点描述 ~30–40 词但语义是路线要求）。
   这既是设计意图（"信息对等但接口不匹配"），也可能是 G4 SR 偏低的部分原因——报告里必须写清。
5. **UE4 子进程泄漏 + pfs 拥塞**（已修，但会反复咬人）：脚本已加按端口清理；
   模型/adapter 需从 `/dev/shm` 加载（做不到时加载可能卡 8 分钟以上）。**服务器是共享机**，
   别人占满 pfs 时所有实验都会变慢。
6. **已知 cosmetic bug**（不影响结论）：初始采样点若已在地标半径内，该阶段的
   `entered/start_step` 字段不会写（`stages` 里显示 `entered=False`），但事件顺序与判定正确。
7. **`work/` 目录只提交了文档与派生数据**；每次 eval 的原始产物在数据盘
   `/mnt/kai_ckp/yiding/MaxHE/eval_results/_ltlf/<TAG>/` 与 `ltlf_logs/<TAG>/`，不在 git 里（太大）。

---

## 6. 我明早建议先看什么（按顺序）

1. `work/implementation-log.md` 的 **§阶段 5**（主对照与偏差说明）→ 再看 `work/results/stage5_rt_metrics.json`；
2. 单条 ours 成功 episode 的**逐步日志**：
   `/mnt/kai_ckp/yiding/MaxHE/ltlf_logs/s5-g5-ours-rt8/f14e7115-....jsonl`
   （每步：位置、引导点、阶段描述、事件、残余、模型 LAND、门控结果、兜底、动作）；
3. `work/stage0_audit.md`（baseline 真实调用链 + 14 条坑）——后面所有实验都建立在这份审计上；
4. `work/results/stage3_metrics.json`（无害性）；
5. 代码：`ltlf_logic/formula.py`（内核）、`ltlf_logic/eval_loop.py`（与 baseline 的接线，**未改 baseline 一行**）。

---

## 7. git 仓库状态

| 项 | 值 |
|---|---|
| 仓库真身 | 服务器 `/mnt/world_foundational_model/kai/MaxHE/ltlf`（`git` 2.34.1） |
| GitHub | `git@github.com:MaxHE911/aerovla-ltlf.git`（SSH，已推送成功） |
| 分支 `main` | `196c5a4` = **baseline-v0**（原版逐字节快照，不再改动） |
| tag `baseline-v0` | annotated tag `fd4b928` → `196c5a4` |
| 分支 `dev` | 开发分支（本文档写完时 = 最新 commit，见 `git log`） |
| 原版是否被改 | `git diff baseline-v0 -- AeroVLA/` **为空**（逻辑层全部在 `AeroVLA/` 之外） |
| 推送方式 | **服务器无 GitHub 凭据且是共享机**，不放置私钥；由 Windows 侧用已授权的 SSH key 做 mirror push（`git push --mirror gh`）。服务器→GitHub 直连可用（HTTP 200），**不需要代理**，也没有配置任何 git proxy |

---

## 8. 复现命令索引（全部从仓库根目录执行）

```bash
PY=/mnt/kai_ckp/yiding/MaxHE/envs_py/aerovla_py310/bin/python
# 单测（62 例，纯 CPU）
$PY -m unittest discover -s ltlf_tests -t .
# M0 规格
$PY -m ltlf_logic.tasks --dataset-root AeroVLA/dataset_raw \
  --spawn-area-json AeroVLA/data/meta/map_spawnarea_info.json \
  --split-json AeroVLA/data/uav_dataset/seen_valset_splits/ModernCityMap.json --out-dir work/tasks
# L0 回放
$PY ops/run_offline_replay.py --dataset-root AeroVLA/dataset_raw \
  --spawn-area-json AeroVLA/data/meta/map_spawnarea_info.json \
  --split-json AeroVLA/data/uav_dataset/seen_valset_splits/ModernCityMap.json \
  --tasks work/tasks/tasks_ModernCityMap_seen_valset.json --out work/results/offline_replay_ModernCityMap.json
# 闭环各组（TAG/GPU/PORT/SPLIT 按需改）
MODEL_BASE=/dev/shm/openvla-7b MODEL_PATH=/dev/shm/aero_vla \
  TAG=g5 MODE=fixed_order GPU=0 PORT=30000 STEPS=70 SPLIT=work/splits/stage5_rt8.json \
  TASK_JSON=work/tasks/tasks_ModernCityMap_seen_valset.json bash ops/run_ltlf_eval.sh
# 离线指标
$PY ops/analyze_results.py --arm g4=<dir> --arm g5=<dir> \
  --tasks work/tasks/tasks_ModernCityMap_seen_valset.json --out work/results/x.json
```

---

## 9. 更新记录

- 04:30 初版写出（阶段 0–4 + RT 主对照）。
- 04:40 ST 组（G2/G3）跑完并补进 §2.4/§2.5；两组臂都正常结束（8/8 episode）。
- 04:38 六组指标合并成一份 `work/results/all_arms_metrics.json`；抽 1 条成功 episode 的逐步日志
  放 `work/samples/`（不用连服务器就能看）。
- 04:38 复跑校验：`git diff --stat baseline-v0 -- AeroVLA/` **无输出**；62 个单测全过；
  L0 回放复跑 123/123 + 497/497+0 不一致，PASS。
- 最终提交：`dev` = `aa9dd6a`（已推 GitHub）；`main` = `196c5a4` = tag `baseline-v0`。
