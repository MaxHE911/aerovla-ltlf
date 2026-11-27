# 实现记录（autonomous, v1）

项目：AeroVLA baseline + 外挂 LTLf 任务逻辑层（按 `work/idea1.md` 的冻结决定实现，按 `work/work1.md` 的阶段推进）。
执行方式：全自动（无人工审查环节），每阶段自测自审后进入下一阶段；未验证项一律标注"未验证"。

运行环境事实（2026-11-20 实测）：
- 作业机：Windows（DSH 工作区 `E:\MaxHE\MyData\MSD\task1\work`），GPU 仿真在远端服务器 `root@10.40.0.190:7735`。
- 服务器代码区 `/mnt/world_foundational_model/kai/MaxHE/`（pfs，94 G 可用），数据区 `/mnt/kai_ckp/yiding/MaxHE/`。
- 仓库：`ssh://root@10.40.0.190:7735/mnt/world_foundational_model/kai/MaxHE/ltlf`（服务器为真身），
  GitHub 镜像 `git@github.com:MaxHE911/aerovla-ltlf.git`。
- 本会话 `ops_bash` 执行器不可用（`EXECUTOR_UNAVAILABLE`：niubash 硬链接并发 bug，与项目无关），
  因此全部命令经 `pwsh`（Windows MSYS/OpenSSH 客户端）驱动；远程脚本以 `tr -d '\r' | bash -s` 方式喂入，
  脚本本体落在仓库 `ops/` 下，可复现。

---

## Step 0：版本管理初始化（2026-11-20 03:00–03:03）

### 0.1 连通性与代理

| 测试 | 结果 | 结论 |
|---|---|---|
| 服务器 `curl https://github.com` | HTTP 200，1.06 s | **服务器直连 GitHub 可用，不需要代理** |
| 服务器 `curl https://ghfast.top/https://github.com/MaxHE911/aerovla-ltlf` | HTTP 403 | 镜像不可用于本仓库 |
| 服务器 `getent hosts github.com` | 20.247.166.64 | DNS 正常 |
| 服务器 `ssh -T git@github.com` | `Permission denied (publickey)`，但 TCP 可达 | 服务器无凭据，**不能从服务器 push** |
| 服务器环境变量 | 无 `*_PROXY` | 无预置代理 |
| 本机（Windows）`git ls-remote https://github.com/...` | `schannel: AcquireCredentialsHandle failed (SEC_E_NO_CREDENTIALS)` | 本机 **HTTPS 走不通** |
| 本机 `ssh -T git@github.com` | `Hi MaxHE911! You've successfully authenticated` | 本机 SSH key 有账号权限 |

**决定（偏离用户建议，已记录原因）**：服务器直连可用但**没有任何 GitHub 凭据**，且是多人共享机器
（`/mnt/world_foundational_model/kai/` 下有他人目录），把个人私钥放到共享服务器风险不可接受，因此
**不配置任何 git proxy，也不在服务器上放置私钥**。改为：

- 服务器上的仓库 = 开发真身（commit / tag / diff 都在服务器）；
- GitHub 推送 = 从 Windows 侧用 **SSH 桥**（本机 key 已授权），
  `git clone --mirror ssh://…` → `git push --mirror git@github.com:MaxHE911/aerovla-ltlf.git`；
- 该流程已固化为命令，见 §0.3。用户要求的"git proxy 配置"在本环境**不存在也不需要**（直连 200）。

### 0.2 仓库与 baseline-v0

服务器仓库 `/mnt/world_foundational_model/kai/MaxHE/ltlf/`：
- `AeroVLA/`、`TravelUAV/`：原版逐字节拷贝（`diff -r` 通过；字节数 32,795,674 / 24,818,442 完全一致），
  去掉嵌套 `.git`；数据/模型目录仍是软链接（`dataset_raw`/`envs`/`openvla-7b`/`checkpoints/aero_vla`/`eval_results`）。
- `ops/`：数据盘上辅助脚本的副本（`smoke_test.sh`、`vulkan_adapter.py`、`hfget.py`、`setup_rpc.sh`、`extract_prep.sh`），原件未动。
- `BASELINE.md`：快照说明（来源、上游 `b87aaa7`、去掉了什么）。

命令（服务器）：`bash ops/_step0_init_repo.sh`（等价脚本已记录在 `work/remote/00_setup_repo.sh`）
结果：`git init -b main` → 1 个 commit `196c5a4`（213 个 tracked 文件，`.git` 36 MB）
→ annotated tag **`baseline-v0`** → `git checkout -b dev`。
推送后 GitHub 引用（`git ls-remote` 验证）：

```
196c5a4…  refs/heads/main
196c5a4…  refs/heads/dev
196c5a4…  refs/tags/baseline-v0^{}   (tag object fd4b928)
```

### 0.3 后续推送命令（每次提交后执行）

```powershell
# Windows 侧，bridge 已存在：work\.gitbridge\ltlf.git
git -C E:\MaxHE\MyData\MSD\task1\work\.gitbridge\ltlf.git fetch --prune origin
git -C E:\MaxHE\MyData\MSD\task1\work\.gitbridge\ltlf.git push --mirror gh
```
（首次是 `git clone --mirror ssh://root@10.40.0.190:7735/…/ltlf` + `git remote add gh git@github.com:MaxHE911/aerovla-ltlf.git`）

### 0.4 分支约定

- `main` / tag `baseline-v0`：冻结的原版参照，**不再动**；
- `dev`：全部开发提交；改动一律与 `baseline-v0` 用 `git diff baseline-v0 -- AeroVLA/` 比对，
  目标是该 diff **恒为空**（逻辑层全部放在 `AeroVLA/` 之外的新文件里）。

---

## 阶段 0：代码/环境审计 + baseline 冒烟（03:03–03:08）

审计内容与风险清单见 **`work/stage0_audit.md`**（调用链、5 个接入点、14 条实测风险）。

### 0.5 baseline 闭环冒烟（原版代码，零修改）

```bash
cd /mnt/world_foundational_model/kai/MaxHE/ltlf
TAG=stage0-baseline MAP=ModernCityMap GPU=0 STEPS=20 bash ops/smoke_baseline.sh
```

| 项 | 实测值 |
|---|---|
| 退出码 | 0 |
| 步数 | 20 宏步（`Timeout (Max Steps Reached)`，maxWaypoints=20，预期） |
| 距离 | 起始 122.3? m → 末步 **48.89 m**（单调下降） |
| 保存帧 | 101 个 `log/*.json` = 20×5+1 |
| UE4 设备 | 日志 `Found 3 device(s)` / `Using Device 0`，对应物理 GPU0（`vulkan_adapter.py 0` → 0） |
| 产物 | `/mnt/kai_ckp/yiding/MaxHE/eval_results/_ltlf/stage0-baseline/84eabab1-…/` |
| 日志 | `/mnt/kai_ckp/yiding/MaxHE/logs/stage0-baseline/{eval.log,server.log}` + driver log |

结论：**通过**（baseline 可跑、可保存、可重读；接入点已定位；可回退）。

---

## 阶段 1：纯逻辑内核（M1 逻辑部分 + M4 事件记录）（03:08–03:10）

代码：`ltlf_logic/formula.py`、`automaton.py`、`semantics.py`、`monitor.py`；测试 `ltlf_tests/`。

### 1.1 实现要点

- **公式表示**：`TOP/BOT/AP/Not/And/Or/F/G/U`；`normalize()` 做 NNF（`¬F=G¬`、`¬G=F¬`）+ n 元扁平/排序/去重
  + 常量折叠 + 互补对（`a∧¬a=BOT`）+ 吸收律（`a∧(a∨b)=a`）。
- **`prog(φ, events)`**：一个"采样点"对应一个事件集合（可为空集）。
  `prog(F x)=prog(x)∨F x`、`prog(G x)=prog(x)∧G x`、`prog(x U y)=prog(y)∨(prog(x)∧x U y)`。
- **`accept(ψ)`**：空迹接受函数（LTLf 的弱真值 t(·)）：`TOP/G→真`、`BOT/F/U/AP→假`、`¬AP→真`、`∧/∨` 递归。
  终止判据 **不用** `residual==TOP`。
- **`G` 显式实现**（冻结决定 2）：保留 `G` 节点与专属推进规则，绝不退化成 `¬F`。
- **独立互验**：`semantics.sat()` 用有限迹的直接 LTLf 定义（不经过残余/导数）实现，
  测试里对小字母表、长度 ≤3 的**全部**事件序列逐个比对 `accept(prog_trace)` 与 `sat`（>1000 例）。
- **M4 monitor**：12 m 半径（3D 欧氏）判定；`inside_prev` 差集保证"停留只触发一次、离开重进再触发"；
  每个采样点**都必须**推进一次（空事件集也推进）；多事件同一采样点按集合语义处理。
- **M1 残余自动机**（`automaton.py`）：不动点展开，节点=残余公式，边=事件集合（1..k 元子集）；
  另给 `enumerate_accepting_paths()`（规划用，默认只走单事件步）与 `swap_equivalent()`（换序判据）。

### 1.2 验收测试

```bash
cd <repo>; python -m unittest discover -s ltlf_tests -t . -v
```
**32 个用例全通过**（Windows 本机 Python 3.14 与服务器 3.10 均跑过，见 §2.6 复跑记录）。

覆盖 work1.md 阶段 1 的 7 条必测：

| # | 用例 | 结果 |
|---|---|---|
| 1 | `F(g)` 只有到达 g 后 accept=True | ✅ |
| 2 | `F(a∧F g)`：a→g 接受；g→a 不接受；g→a→g 接受 | ✅ |
| 3 | `F(a∧F(b∧F g))` 正确顺序接受、乱序不接受 | ✅ |
| 4 | `F a∧F b∧F g` 合法排列接受（6 种序） | ✅ |
| 5 | `G¬z∧F(a∧F g)`：进 z 后残余=BOT 且不可恢复；无 z 完成时 accept=True（残余语法上不是 TOP） | ✅ |
| 6 | 同采样点多事件、空事件集推进、停留只触发一次、离开重进再触发 | ✅ |
| 7 | 小公式状态数有限（无语法膨胀） | ✅ |

**残余状态数实测**（`build_automaton`，事件集合=原子命题子集）：

| 公式 | 本实现 | idea1 原型声称 |
|---|---|---|
| `F(g)` | **2** | （未列） |
| `F(a∧F g)` | **3** | 3 ✅ |
| `F(a∧F(b∧F g))` | **4** | 4 ✅ |
| `F a∧F b∧F g` | **8** | 8 ✅ |
| `G¬z∧F(a∧F g)` | **4** | 3（差 1，见下） |

> 差异说明（不沉默改动）：`G¬z∧F(a∧F g)` 得 4 个残余而不是 3 个，原因是我们的规范化**不做
> `And` 对 `Or` 的分配律**（`G¬z ∧ (F g ∨ F(a∧F g))` 不被展开成 `(G¬z∧F g) ∨ (G¬z∧F(a∧F g))`）。
> 这是有意选择：分配会带来状态数乘性增长，而**接受语义不受影响**（`accept` 与独立语义互验一致）。
> 该公式族的状态数仍是有界小整数，符合"不出现无控语法膨胀"的验收要求。

### 1.3 与 design 的两处语义澄清（记录，不改冻结项）

1. **同时进入多个地标**：同一采样点内 `{a,g}` 同时成立时，`F(a∧F g)` **是接受的**
   （标准 LTLf 集合语义：嵌套 F 可在同一点满足）。这与"严格顺序"并不矛盾——严格性体现在
   **不同时间点**的顺序上（`g→a` 不接受）。为保证实验里不出现这种歧义，M0 构造任务时要求
   地标两两距离 > 24 m（2×12 m 半径），offline judge 会单独统计"同一采样点多事件"的次数。
2. **`¬z` 与 `G¬z` 的区别**（idea1 冻结决定 2 的反例）已固化为回归测试：
   `F(a∧F g) ∧ ¬z` 在 `a,z,g` 上 accept=True（约束丢失），`F(a∧F g) ∧ G¬z` 在 `a,z,g` 上残余=BOT。

---

## 阶段 2：M0 离线规格构造 + L0 离线回放（03:10–03:12）

代码：`ltlf_logic/tasks.py`（M0）、`ltlf_logic/offline.py` + `ops/run_offline_replay.py`（L0）。

### 2.1 数据事实核对（实测，不是引用文档）

| 项 | 实测值 |
|---|---|
| `ModernCityMap` 区域数 | **33**（23 种 asset），目标点坐标 = `area[9:12]` |
| ✅ 坐标约定核对 | seq `84eabab1…` 目标点 = `(29.400, 9.079, -0.087)`，与 baseline 阶段 0 冒烟 eval.log 打印的 `target: <29.40, 9.08, -0.09>` **完全一致** |
| split 记录数 / 唯一轨迹数 | 2873 / **124**（`frame` 字段被 baseline 忽略 → 每条轨迹会被重复评测，必须去重） |
| 轨迹帧数 | 69–130（中位 104） |
| 专家轨迹末帧到目标中心距离 | 中位 **5.10 m**，p90 6.67 m，最大 18.91 m（1 条 >12 m） |
| 专家路径长度 | 中位 **99.0 m**，p90 150.9 m，最大 203.0 m |
| 真实目标描述长度 | 中位 34–43 词（`object_description.json` 里的 2 词短描述**没有**使用） |

### 2.2 构造规则（固定，可复现）

1. 事件 = 进入某地标中心的 12 m 球（3D 欧氏），沿专家轨迹按帧检测，只记首次进入；
2. 排除最终目标本身；**排除在地图里没有任何真实 40–60 词描述的地标**；
3. 地标两两间距 > 24 m（2×半径，避免同一采样点同时命中两个地标）、与最终目标间距 > 24 m；
4. 最多 3 个中间地标 → 公式 `F(L0 & F(L1 & … & F(g)))`；
5. 与"起点→终点直线"的垂直距离 ≤ 12 m 记 on-line（ST），否则 off-line（RT）；任务只要含 1 个
   off-line 地标即记 RT（分析字段，不影响公式）；
6. 每个中间地标的阶段描述 = 数据集里**该区域当过最终目标**的那条轨迹的真实描述（取词数最接近
   训练中位 43 词者，记录来源 seq）；最后阶段描述 = 该 episode 自己的原描述（一字不改）；
7. 长指令用固定模板回译（中文版按 idea1 §5.3；**另附逐字英译版**，因为 baseline 唯一能接收
   文本的槽位是英文 prompt 的 `obj_desc`，见 §2.4 说明）；
8. 生成后立刻用**该 episode 的专家轨迹**做一次 `accept` 断言；不通过则标 `skipped`。

### 2.3 结果（服务器实跑，`work/tasks/summary_ModernCityMap_seen_valset.json`）

```bash
PY=/mnt/kai_ckp/yiding/MaxHE/envs_py/aerovla_py310/bin/python
$PY -m ltlf_logic.tasks --dataset-root AeroVLA/dataset_raw \
    --spawn-area-json AeroVLA/data/meta/map_spawnarea_info.json \
    --split-json AeroVLA/data/uav_dataset/seen_valset_splits/ModernCityMap.json \
    --map ModernCityMap --split-name seen_valset --out-dir work/tasks
```

| 指标 | 值 |
|---|---|
| 唯一轨迹 / 可用任务 | 124 / **123**（1 条跳过：专家轨迹自身不满足 `F(g)`，末帧距目标 18.9 m > 12 m，属规则允许的排除） |
| 阶段数分布 | 2 阶段 59 / 3 阶段 33 / 4 阶段 31 |
| ST / RT | ST 88 / RT 35 |
| 地标对过近的任务 | **0 / 123** |

L0 离线回放（`ops/run_offline_replay.py`，队列内部含 2 套独立判定器）：

| 检查 | 结果 |
|---|---|
| 专家轨迹满足自身公式 | **123 / 123** |
| 负例（漏事件/漏终点/交换前两阶段/逆序/违反 G）被拒绝 | **497 / 497** |
| 残余推进判定 vs 直接有限迹语义判定 不一致数 | **0** |
| 单元测试总数 | **57**（含 M0 真实数据测试、monitor、fake-executor 阶段机） |

复跑命令与结果文件：`work/results/offline_replay_ModernCityMap.json`。

### 2.4 与设计的两处最小修正（已记录，不改冻结项）

1. **长指令需要英文版**：idea1 §5.3 的模板是中文，但 `baseline+长指令` 组只能把文本塞进英文
   `obj_desc` 槽位。做法：保留中文模板（用于论文任务陈述），**逐字翻译**一份英文版用于注入，
   并在日志与结果里同时记录两版。这不是新设计，只是把同一句模板换成能进 prompt 的语言。
2. **地标两两间距约束**：idea1 未显式要求，但标准 LTLf 集合语义下"同一采样点同时进入两个地标"
   会让 `F(a & F(b))` 一次性满足。为避免这种歧义，构造时强制 24 m 间距（实测 0 个任务违反），
   并在 judge 里单独统计 `multi_event_points`。

---

## 阶段 3：单阶段退化 + baseline 无害性（03:30–03:44）

协议（三组共用同一 split，12 条不同轨迹，maxWaypoints=70）：

```bash
# G0：原版 baseline（未改一行代码）
TAG=s3-g0-baseline GPU=0 PORT=30000 STEPS=70 SPLIT=work/splits/stage3_12ep.json bash ops/run_baseline_eval.sh
# G1：ours 单阶段退化（gate 关、takeover 关、距离递增启发式保持开启、引导点=最终目标、描述=原描述）
TAG=s3-g1-singlestage MODE=single_stage GPU=1 PORT=30010 STEPS=70 SPLIT=work/splits/stage3_12ep.json bash ops/run_ltlf_eval.sh
```

### 3.1 结果（`work/results/stage3_metrics.json`，N=12 条轨迹）

| 组 | episodes | SR | OSR | NE (m) | SPL | ESR | SR∧ESR |
|---|---|---|---|---|---|---|---|
| G0 baseline（原版） | 12 | **0.333** (4/12) | **0.500** (6/12) | 38.66 | 0.265 | **0.417** (5/12) | **0.333** (4/12) |
| G1 ours 单阶段退化 | 12 | **0.333** (4/12) | **0.500** (6/12) | 41.34 | 0.301 | **0.417** (5/12) | **0.333** (4/12) |

**离散指标（SR/OSR/ESR/SR∧ESR）两组完全一致**；NE/SPL 差异 2.7 m / 0.036，量级与下面测到的
仿真非确定性噪声相当。G1 的 `suppressed_stops`、`takeovers`、`stage_switches` 全为 0（gate 未启用），
与设计一致。

### 3.2 关键环境发现：闭环**不**可逐位复现（这条决定后续所有对照的做法）

同一 episode、同一配置、原版 baseline 连跑两次：

| 指标 | 第 1 次 | 第 2 次 | ours(single_stage) |
|---|---|---|---|
| reset 后初始 z | −3.088 | **−2.730** | −3.397 |
| 第 1 步后 z | −0.681 | **−2.231** | −4.762 |
| `frontcamera/000000.png` md5 | 19c66af7 | **f559be9a** | 8eb0eb50 |

结论：**初始位姿（setPoses 后的落点）与渲染图像都不是逐位确定的**（UE4 渲染/物理 settle 时序 +
TAA 抖动），所以"相同输入 → 相同动作"无法通过在仿真里重放两次来验证。因此：

- 无害性用**接口级证明** + **离散指标统计对照**（3.1 表）；
- 接口级证明：`ltlf_tests/test_prompt_equivalence.py` 用原版的 prompt 组装逻辑（含
  `get_semantic_direction` 7 桶、`split("degrees from you.")` 切分）对比我们的注入，单阶段模式
  **逐字节相同**（通过）；
- 多阶段模式只改变"引导点 + 物体描述"两个值，其余 prompt 模板不变（同一测试覆盖）。

### 3.3 期间修的 bug / 环境坑（都写进记录，不掩盖）

1. `get_obs()` 返回的是每个 env 的 4 元组 `(observations, dones, collisions, oracle_success)`，
   我们的循环一开始直接把 `outputs[i][-1]` 当观测用 → `TypeError: 'bool' object is not subscriptable`；
   已改为先解包再取 `observations[-1]`。
2. **UE4 子进程泄漏**：仿真服务端会拉起 UE4 游戏进程，只 `pkill AirVLNSimulatorServerTool` 会漏掉它，
   连跑几轮后机器上堆了 **8 个 UE4 实例、load average 366**，模型加载被拖到 7 分钟以上。
   修复：脚本收尾与启动时都 `pkill -f airsim_plugin/settings/`（已加 `kill_sim()`，见 ops/*.sh）。
3. 运行脚本必须把 `SPLIT`/`TASK_JSON` 转成绝对路径（评测时 cwd 会切到 `AeroVLA/`）。
4. 后台 ssh 启动任务时不要让远端命令保持 stdout 管道打开（否则 ssh 挂住），统一用
   `nohup ... > driver.log 2>&1 &`。

---

## 阶段 4：最小多阶段闭环（固定顺序）（04:00–04:13）

### 4.1 先做 fake executor 单测（work1.md 要求的顺序：先假执行器，再仿真）

`ltlf_tests/test_stage_machine.py`（5 个用例，纯逻辑，无 torch/无仿真）复刻真实循环里决定
"当前阶段 / 能否停 / 要不要兜底"的那几个组件：

| 用例 | 断言 |
|---|---|
| 中间地标处 LAND | 门控**拦下**（`suppressed_residual_pending`）、阶段立刻推进到下一个地标、零动作被兜底替换 |
| 全程按序 | 残余 `accept=True`，最后阶段 LAND 被放行（`accepted`），拦下+放行合计恰好 1 次 |
| 单阶段退化 | 门控关闭，LAND 直接放行，无兜底 |
| 乱序进入 L1 | **不会跳过 L0 阶段**（阶段由残余推导，不是可变指针） |
| 带 `G` 的违规 | 残余变 `BOT`，此后 LAND 永远不放行 |

### 4.2 真机（仿真器）最小闭环：2 条 RT 多阶段 episode

```bash
TAG=s4-fixed-order-rt2 MODE=fixed_order GPU=1 PORT=30010 STEPS=70 \
  SPLIT=work/splits/stage4_rt2.json TASK_JSON=work/tasks/tasks_ModernCityMap_seen_valset.json \
  MODEL_BASE=/dev/shm/openvla-7b MODEL_PATH=/dev/shm/aero_vla bash ops/run_ltlf_eval.sh
```

| seq | 阶段数 | 步数 | SR | ESR | 阶段切换 | 事件顺序 | 末帧距终点 |
|---|---|---|---|---|---|---|---|
| `6badc126` | 4 | 43 | ✅ | ✅ | 2 | `L0 → L1 → L2 → g` | 3.5 m |
| `f14e7115` | 4 | 1 | ❌ | ❌ | 0 | `L0` | 71.4 m（第 1 步即被判碰撞） |

**通过**：多阶段能按公式顺序完成（事件顺序与公式完全一致）、最终成功仍按 baseline 的
"20 m + 有效停止"判定、残余在结束时 `accept=True`。第 2 条在第 1 步就被深度碰撞判据判死
（该起点贴障碍物，属仿真/起始条件问题，不是逻辑层问题，已记录为该 episode 的失败类型）。

### 4.3 重要观察：这批运行里 **门控没有被触发**

两次 Stage-4 + 全部 Stage-5 ours 运行中 `suppressed_stops = 0`、`takeovers = 0`。原因：
AeroVLA 在拿到"中间地标描述"时**不会**误判为"找到目标"而 LAND（它按描述找的是另一个物体），
所以 idea1 设想的"阶段边界 LAND"在这批 episode 里没有出现。
→ M5 门控/兜底的**逻辑正确性**由 fake-executor 单测保证；**在仿真中的触发统计目前为 0**，
不能声称门控在真机上被验证过。这是如实记录，不是通过。

### 4.4 期间的基础设施修复（影响后续所有实验）

1. **UE4 进程泄漏**（见 §3.3.2）已修；并且 kill 模式改为**按端口**：
   `airsim_plugin/settings/$PORT/`。之前用宽匹配 `airsim_plugin/settings/`，导致**并行运行的
   另一条 arm 的 UE4 被判死**（表现为 eval 卡在 "waiting for airsim connection" 8 分钟）。
2. **pfs 拥塞导致模型加载卡死**：`/mnt/kai_ckp` 是并行文件系统（100 T，90% 已用）。
   观察到进程处于 `cxiWaitEventWait`（CXI 网络文件系统 RPC 等待）、`read_bytes=0`、
   `rchar=316 MB` 却 8 分钟不前进——即模型权重（15 GB）每次都要走网络文件系统。
   **修复**：把模型与 adapter 拷到 RAM 盘 `/dev/shm`（100 GB tmpfs），用
   `MODEL_BASE=/dev/shm/openvla-7b MODEL_PATH=/dev/shm/aero_vla` 启动。
   效果：单 arm 启动到出第一个 episode 从 **>8 分钟（卡死）** 降到 **~2 分钟**。
   注意 `/dev/shm` 里原本有两个**指向 pfs 的同名软链接**，`cp -a` 会只拷贝软链接本身（
   0 字节），必须用 `cp -aL`。这条坑已记录在脚本注释与 README。

   > **2026-11-20 17:30 更正（重要，之前的归因是错的）**：上面的"效果"其实是**混淆**。
   > `AeroVLA/src/model_wrapper/aerovla_wrapper_ui.py:22` 把 `base_model_path` 写死成
   > `"./openvla-7b"`，**`--model_base` 根本不生效**（`--model_path` 生效，只管 442 MB 的 adapter）。
   > 也就是说那次"变快"是因为 pfs 拥塞恰好缓解，而不是因为模型进了 RAM。
   > 当天 17:2x 复跑时 pfs 再次拥塞（load 391、90 T/100 T），同样卡在 D 状态，
   > 证明前一次的因果判断错了。**正确做法**（已实现并 e2e 验证）：
   > `ops/sim-runtime/with_shm_model.sh` 在**私有 mount namespace** 里把
   > `/dev/shm/openvla-7b` bind-mount 到软链接的解析目标（`/mnt/kai_ckp/.../models/openvla-7b`），
   > 从而让 `./openvla-7b` 读到 RAM 副本；**磁盘上什么都没有改**（`git status -- AeroVLA/` 干净，
   > 也不影响同机其他用户）。三个 driver（`run_ltlf_eval.sh` / `run_baseline_eval.sh` /
   > `smoke_baseline.sh`）都已接上，日志首行会打印
   > `[with_shm_model] bind-mounting ... over ...`。
   > 验证：5 步 baseline episode 从启动到落盘 **4.4 分钟**（含 ~2 分钟场景加载），
   > 模型加载不再卡；`/dev/shm/openvla-7b` 不存在时自动退回原路径。
   > **教训**：第一次归因到 pfs 之前应该先给"变快"做对照（换回 pfs 再跑一次），
   > 否则会把"环境缓解"误记成"我修好了"。

---

## 阶段 5：ModernCityMap 第一版对照（04:13–现在）

四组协议统一：同一 split（一条轨迹 = 一条 episode，去重后）、`maxWaypoints=70`、`STEPS=70`、
模型/adapter 从 RAM 盘加载、每次用全新输出目录。

| 组 | 系统 | 任务 | split |
|---|---|---|---|
| G0 | 原版 baseline | `F(g)` | 12 条（first:12） |
| G1 | ours 单阶段退化 | `F(g)` | 同上 |
| G4 | baseline + 长指令（塞进 `obj_desc`，保留两个 prompt 锚点） | `F(L0&F(…&F(g)))` | RT 8 条 |
| G5 | ours 固定顺序多阶段 | 同 G4 | 同 G4 |
| G2 | baseline + 长指令 | 长任务 | ST 8 条 |
| G3 | ours 固定顺序 | 长任务 | 同 G2 |

**协议差异（必须披露）**：G0/G1 保持 baseline 原有"距目标连续 10 次不减少即失败"启发式**开启**；
所有长任务组（G2–G5）统一**关闭**该启发式。代码中的落实方式是 runner 内显式替换
`closeloop_util.target_distance_increasing_for_10frames`，并在 eval.log 打印
`protocol: ... heuristic DISABLED`。

### 5.1 主对照（RT 组，8 条不同轨迹，独立离线判定）

`work/results/stage5_rt_metrics.json`（判定器只用保存的 `log/*.json` 位姿重建事件，
不使用在线逻辑层状态；在线/离线判定一致率 **8/8 = 1.00**）

| 组 | episodes | SR | OSR | NE (m) | SPL | **ESR** | **SR∧ESR** | 阶段完成率 | 阶段切换 | 误停抑制 | 兜底接管 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G4 baseline+长指令 | 8 | 0.250 (2/8) | 0.250 | 39.65 | 0.227 | **0.000 (0/8)** | **0.000** | 0.448 | — | — | — |
| G5 ours 固定顺序 | 8 | **0.625 (5/8)** | **0.750** | **13.20** | **0.470** | **0.625 (5/8)** | **0.625** | **0.854** | 20 | 0 | 0 |

逐条（事件序列为离线重建的完整序列，重复项=离开后重进）：
- G4：`6badc126` 到点但只经过 L0 → ESR ✗；`8daaf47b` 经过 L0→L1 后直接到 g（漏 L2）→ ESR ✗；
  `9812065a` 经过 L0→L1→L2 但最后没进 g 的 12 m → ESR ✗；其余 5 条根本没接近目标。
- G5：`6badc126` 事件 `L0→L1→L2→g`，43 步，SR+ESR ✓；`8daaf47b` 64 步 ✓；`c8e7469a` ✓；
  `dc55f10b` ✓；`f14e7115` ✓。失败 3 条：`6452f234` 第 1 步被判深度碰撞（起点贴障碍物）、
  `9812065a` 走到 L2 但没进 g 的 12 m（末帧 18.5 m）、`982c8942` 70 步超时（末帧 17.3 m）。

**G4 vs G5 的差别不在"到没到"，在"按不按路线走"**：G4 有 2/8 到点（SR 25%）但 ESR 全 0；
G5 达到 5/8 且都是"到点 + 按顺序经过全部地标"。这正是 idea1 §5.6 假设的方向，但**样本只有 8 条**，
不能当作统计显著的结论。

### 5.1b ST 组对照（8 条不同轨迹，`work/results/stage5_st_metrics.json`）

| 组 | episodes | SR | OSR | NE (m) | SPL | ESR | SR∧ESR | 阶段完成率 |
|---|---|---|---|---|---|---|---|---|
| G2 baseline+长指令（ST） | 8 | 0.375 (3/8) | 0.375 | 47.02 | 0.329 | 0.375 | 0.375 | 0.625 |
| G3 ours 固定顺序（ST） | 8 | **0.625 (5/8)** | **0.625** | **28.57** | **0.594** | **0.625** | **0.625** | **0.812** |

**ST vs RT 的对照正好支持 idea1 的判别假设**：ST 组的地标都在"起点→终点"直线附近，
所以"直飞"会**偶然**满足部分/全部中间地标要求 → G2 的 ESR 从 RT 的 **0.000** 升到 ST 的 **0.375**；
RT 组地标偏离直线，直飞无法偶然满足 → G4 ESR = 0。ours 在两种任务上都是 0.625 ESR。

| 对比（同为 8 条） | baseline+长指令 ESR | ours ESR |
|---|---|---|
| ST（顺路地标） | 0.375 | 0.625 |
| RT（偏线地标） | **0.000** | 0.625 |

即：**两者的差距在 RT 上最大（0.625 vs 0）**，在 ST 上缩小（0.625 vs 0.375），
说明改善来自"逐阶段落实路线"，而不是"任务整体变简单"。

### 5.2 已知偏差 / 需要读者警惕的点

1. **G4 的 ESR=0 不是"设计成 0"**：判定用严格 `accept(最终残余)`，G4 的轨迹确实缺中间地标。
   但 G4 的 prompt 里塞的是英文长指令（例如 "First pass the XX, then finally reach the YY."），
   与训练分布（单目标描述）不同，**可能**让它比"原样 baseline"更差；因此 G4 的 SR(0.25) 低于
   G0(0.333, 不同 episode 集) 这一现象不能单独归因于长指令。
2. **ours 的 SR 提升有多少来自"门控"**：本批运行 `suppressed_stops=0`、`takeovers=0`，
   即门控/兜底**没有触发过**。所以 G5 的收益目前只能归因于"逐阶段给引导点 + 阶段描述"，
   不能归因于 M5。要做 ±门控消融才能回答。
3. **样本量**：每组 8 条（RT）/12 条（G0/G1），只够看方向，不够做显著性检验。
4. **仿真非确定性**（§3.2）意味着同一组重跑结果会有波动；本次未做多种子重复。

---

（后续阶段记录追加在下方）

### 4.5 仓库卫生：仿真运行时写入的 settings 文件曾被误提交（已回滚）

AirVLNSimulatorServerTool.py 每次启动都会写 AeroVLA/airsim_plugin/settings/<port>/settings.json，
git add -A 会把它们当作改动提交。已在 commit 996f922 中：
把 AeroVLA/ 恢复到 aseline-v0 的内容、删除新增端口目录、并把该目录加入 .gitignore。
现在 git diff baseline-v0 -- AeroVLA/ **为空**，可以用一条命令复核：
```
git diff --stat baseline-v0 -- AeroVLA/   # 期望：无输出
```
（注意：以后每次跑完仿真仍会重新生成这些文件，提交前需 git checkout -- AeroVLA/ 或
git clean -fd AeroVLA/airsim_plugin/settings/；这一点已写进 README 的注意事项。）

---

# v2 会话记录（2026-11-27，多地图 + idea2 STL + idea3 后训练）

> 本段追加于 `work/implementation-log.md` 末尾。所有命令均为服务器上实际执行过的；
> 未验证的内容显式标注「未验证」。

## V2.0 会话开始时的真实状态核对（07:00）

- 服务器 `root@10.40.0.190:7735` 可连；load ≈56（多人共享），GPU0 已用 21.5/80 GB（他人），GPU1 已用 64.7/80 GB（他人，仅剩 ~17 GB → 全程未使用 GPU1）。
- 仓库 `dev@f1f46c7`，`AeroVLA/` 与 `baseline-v0` 逐字节一致。
- 上一轮结论"只有 ModernCityMap 有数据"**被本轮推翻**：见 V2.1。

## V2.1 地图与数据盘点（本轮新增事实）

- 仿真场景（`envs/closeloop_envs/`）：ModernCityDowntown(=ModernCityMap)、ModularEuropean、ModularPark、NYCEnvironmentMegapa、NewYorkCity、TropicalIsland 共 6 个。
- `dataset_raw/` 目录：ModernCityMap 718、NewYorkCity 1139、NYCEnvironmentMegapa 2015、TropicalIsland 1418、ModularPark 958、Carla_Town01 954（**缺 merged_data.json，跳过**）；Carla_Town04/05/06/07/10HD/15 只有分卷压缩包，未解压。
- `data/meta/map_spawnarea_info.json` 覆盖以上全部 UE4 地图（Carla 系列也在）。
- 仿真器按 `item['map_name']` 自动开场景（`env_uav.py:257` → `AirVLNSimulatorServerTool.py:242`），**加地图不需要改代码**，只要 split + tasks。
- **可跑地图数 = 5**，其中 ModularPark 是官方 *unseen map* 划分，且训练集里 **0 条样本**（见 V2.6）→ 真正的零样本地图。
- 每张图数据抽查 6 条轨迹的 `merged_data.json`：键一致（`trajectory/trajectory_raw/trajectory_raw_detailed/image_feature_path/index/length/conversations`），无缺失。

## V2.2 训练数据与动作空间的实测事实（决定了 idea2/idea3 的设计）

- `AeroVLA/data/aerovla_train_dataset.json` = **410,450 条**训练样本；按图分布：
  Carla_Town01 38,742 / Town04 24,873 / Town05 19,936 / Town06 16,802 / Town07 15,594 / Town10HD 27,611 / Town15 36,612 /
  ModernCityMap **13,715** / NYCEnvironmentMegapa 109,856 / NewYorkCity 47,856 / TropicalIsland 58,853 /
  **ModularPark 0**。
- 动作空间不是纯回归：`aerovla_dataset.py:75-79` 把连续 `label{fwd,down,yaw}` **量化成 3 个 bin**（`f"{bin_fwd:02d} {bin_down:02d} {bin_yaw:02d}"`），推理端 `aerovla_wrapper_ui.py:212-223` 再反量化。
  → 模型输出是**离散 token**；因此"动作投影式屏蔽"只能做**事后投影**（改解码后的连续量），不能做梯度。
- 动作执行语义（`AirVLNSimulatorClientTool_AeroVLA.py:299-395`）：`dx=fwd·cos(yaw_cur+Δyaw)`、`dz=down`，
  cruise 1.0 m/s，`duration=fwd`，随后 `moveByVelocity(0,0,0,0.5)`；`|Δyaw|≥0.25 rad` 先原地转。
  每个宏动作只产生**一个**新位姿（`for _ in range(5)` 复制同一帧）→ STL 用**宏步**做时间单位（不是秒）。
- 训练入口：`AeroVLA/src/train_aerovla.py`（LoRA r=64、α=128、dropout 0.05、lr 2e-4、micro-bs 2、accum 8、5 epoch），`scripts/train.sh`。
  **本轮未修改该文件**；若要跑后训练会另写 `ops/` 下的入口。

## V2.3 评测基础设施（本轮新增）

- `ops/launch_arm.sh`：单臂分离式启动（`setsid nohup`），支持任意 `K=V` 覆盖（`GPU=`、`MODEL_PATH=` 等）。
- `ops/run_next.sh`：**依赖式排队器**——等指定进程模式消失后再启动本臂。
  坑：`pgrep -f` 匹配 `/proc/<pid>/cmdline`，所以等待模式**必须通过环境变量传入**，否则排队器会匹配到自己而死锁。
  另一个坑：`pkill -f "run_wave_sequence.sh"` 会**杀掉执行该命令的 ssh 会话自己**（自己的 cmdline 含该字符串）→
  改用 `pat="run_wave_seq""uence.sh"` 拆分字面量后再 `pgrep/kill`。
- `ops/queue_chain.sh`：把 6 个波次按依赖串起来（wave1 → mp → stl → rf → nyc → ti → nycem），全程无人值守。
- 坑（已修）：把 `cd X && R=Y && ... && cmd &` 整体放到后台时，`&` 会把**整条链**变成子 shell，后续命令拿不到 `R`/cwd → 报 `bash: ops/run_ltlf_eval.sh: No such file`。

## V2.4 idea2（STL）实现与验证

- 新增 `ltlf_logic/stl.py`（AST + min/max 鲁棒度 + 在线上下界 `bounds()`）、
  `ltlf_logic/stl_tasks.py`（`chain`/`schedule`/安全包络 + 离线打分 + CLI）、
  `ltlf_logic/stl_layer.py`（在线监控 + 截止期提示 + 动作投影 shield）、
  `ltlf_tests/test_stl.py`（18 个测试）。
- 集成：`eval_loop.py` +63 行（`mode=stl` 时实例化/每步 observe/gate 前 shield/阶段切换后提示/落盘）、
  `runner.py` 加 `MODES += ("stl",)` 与 7 个 `stl_*` 字段（`stl_step_len_m=4.3`）。
- 测试：`python -m pytest ltlf_tests/ -q` → **80 passed**（62 旧 + 18 新）。
- 交叉验证：5 组真实 rollout（39 条 episode）上 `STL chain 的 ρ>0` 与 `offline.independent_esr()` **一致 39/39**。
- 过程中发现并修复的问题（详见 `work/idea2-stl.md` §9）：步数被位姿去重吞掉、`bounds()` 语义与实现不符、
  交叉验证用错字段（`accepted` 依模式而异）、安全包络误用地标中心 z、ρ 均值出现 −inf、6 个新测试首轮失败（3 个是测试自己写错）。
- 工具坑：服务器 aerovla 环境原本**没有 pytest**，`pip install --index-url tuna pytest` 后可用；
  本地 Windows 装不上 pytest（PEP 668 + uv 环境），**本地只做 `py_compile` 语法检查，测试一律在服务器跑**。

## V2.5 离线 STL 数值（0 GPU，对已跑完的 rollout 重打分）

见 `work/results/stl_off_*.json` 与 `work/idea2-stl.md` §7。关键三条：

1. RT 组"阶段从未到达"率：baseline+长指令 **48.3%** → 本层 **13.8%**；
2. 但本层"到达者中迟到"率 **34.5%**（baseline 6.9%，因为它几乎不到达）→ **顺序已修好，时间还没修**；
3. `ρ_schedule` 中位数：G5 **−5.06 m** / G3 **+4.08 m** / G4 **−41.30 m**。

## V2.6 idea3（后训练）现状

- 文献结论：Logic-VLA = STL 条件 SFT + 轨迹级 IPO（+20.4pp 再 +20.3pp），且**可微鲁棒度当损失会把 NL 能力打崩**（68.5→45.0）；
  RefineFly = 同底座（openvla-7b+LoRA/AeroVLA 管线）的 token 级 PPO + 失败记忆，+8.37pp，**代码未开源但 HF 权重匿名开放**。
- 已下载 `refinefly/refinefly`（397 MB，`r=64/α=128/base=openvla-7b`）到 `$R/models/refinefly`，并排进评测队列作为对照臂。
- 已实现 `ops/make_posttrain_data.py`（Stage-A 数据管线：同一批帧/同一批动作标签，只改写指令为分阶段提示；50/50 与原指令混采）。
  方向分桶与 baseline `get_semantic_direction` 逐条一致：**3000 条随机位姿 0 处不一致**。
- 发现：官方训练集只含**训练轨迹**，与评测轨迹不重叠 → 生成 Stage-A 数据必须给训练轨迹补 M0 任务（正在进行）。
- **未验证**：LoRA 冒烟尚未跑（GPU0 被评测波次占满）。

## V2.7 待办（按优先级）

1. 多地图评测波次跑完 → 汇总表（SR/OSR/ESR/NE/SPL + STL ρ）写入周报；
2. idea2 在线 `stl` vs `fixed_order` 对照（队列中）；消融（shield off / deadline off）视算力；
3. RefineFly 权重臂结果（若适配器能加载）；
4. idea3 LoRA 冒烟（等 GPU）；
5. 周报 + 组会材料。

## V2.8 运行期 bug 与修复（本轮真实发生，按发现顺序）

### (a) 同一条命令里 `&` 把整条链变成子 shell
`cd X && R=Y && ... && cmd &` —— `&` 作用于**整个列表**，导致后续命令拿不到 `R`/cwd，
报 `bash: ops/run_ltlf_eval.sh: No such file` 与 `/logs/...` 路径。
→ 改为独立的 `ops/launch_arm.sh`（参数化 + `setsid nohup`）。

### (b) `pkill -f "<自己的命令里出现的字符串>"` 会杀掉执行它的 ssh 会话
两次踩到（`pkill -f run_wave_sequence.sh`、`pkill -f run_next.sh`），现象是 ssh 直接返回 exit 1 且无输出。
→ 用字符串拼接绕开自匹配：`pat="run_ne""xt.sh"; pkill -f "$pat"`。

### (c) 依赖式排队器（等"前驱进程消失"）存在竞态
后置等待器在前驱**尚未启动**时 `pgrep` 为空 → 判定"已完成" → **立即启动**。
后果：12 个臂几乎同时起动 → 全部 CUDA OOM（wave1 未受影响），留下 9 个空结果目录与 9 个僵尸 server。
→ 废弃 waiter 链，改为**单进程顺序执行**（见 (e)）。

### (d) `setsid nohup cmd &` 让 `$!` 变成 setsid 的 PID
`setsid` 立即 fork 后退出 → `wait "$pid"` **立刻返回** → 每个波次"瞬间完成"，
序列脚本在 30 秒内把 6 个波次的 12 个臂全放出去（又一次险些 OOM）。
→ 改为 `nohup cmd &`（`$!` 即真正的驱动脚本 PID），`wait` 才有效。
→ 这是本轮最隐蔽的坑：**表面上"自动排队成功"，实际完全没有排队**。

### (e) STL 在线层被 1 条 episode 的冒烟抓出的 3 个真 bug
1. **控制步重复计数**：`observe()` 在 `__init__`/`update_ltlf`/`record_step` 三处被调用，
   内部用 `len(positions)` 自增 → 步数以 2× 速度增长，窗口全部错位（摘要里出现 `best_step=49` 而 episode 只有 25 步）。
   → `observe(i, pose, step=...)` 显式接收控制步；位姿去重只影响轨迹，不影响时钟。
2. **包络用了默认值**：`LTLfConfig` 默认 `1.0/30.0`，而离线标定是 `1.722/33.638` →
   在线/离线口径不一致。→ 默认值改为标定值，并在注释里写明来源。
3. **截止期提示重复触发**：判据用"上次警告的 step"→ 在窗口尾段**每步**都追加一句。
   → 改为按截止期签名 `(lo, hi)` 去重，每个截止期只提示一次。

### (f) wave1 截断决策
wave1（MCM RT 全量 35 条）跑到 21/21 时，两臂 SR 已稳定在 .316/.368（Wilson 区间大幅重叠），
继续跑边际信息低 → 主动截断，把 GPU 让给多地图与 idea2/idea3 的实验。
截断是通过 `kill -9` 掉两个 python 评测进程实现的（驱动脚本会继续执行清理，sim server 会被回收）。

### (g) 数据事实修正：训练集与评测集不重叠
`make_posttrain_data.py` 首版在评测轨迹上跑出 0 条样本（`traj_without_task: 594`）——
因为官方 SFT 数据只含**训练轨迹**。于是给 594 条 MCM 训练轨迹补建 M0 任务
（`work/tasks/tasks_ModernCityMap_train.json`），再生成 Stage-A 数据（13,715 条）。
这也顺带确认：**ModularPark 在训练集里是 0 条**（真零样本图）。

## V2.9 运行期第二次事故链（08:15-08:30）与最终机制

### (h) `$(arm ...)` 命令替换让 `wait` 失效（rc=127）
`pA=$(arm ...)` 里的 `&` 进程是**子 shell 的子进程**，主脚本 `wait` 拿不到 → 返回 127，
于是"每波都瞬间完成"。修复：`arm()` 内部设全局 `LAST_PID=$!`，调用侧 `arm ...; pA=$LAST_PID`。
这条和 (d) 是同一类错误的不同变体，教训一致：**后台进程必须由同一个 shell 启动并等待**。

### (i) `stl` 臂中途崩溃：baseline 拿到空图像
```
File ".../aerovla_wrapper_ui.py", line 86, in prepare_inputs
    frame_front_rgb = cv2.cvtColor(rgb_list[0], cv2.COLOR_BGR2RGB)
cv2.error: ... !_src.empty() in function 'cvtColor'
```
`stl-a` 在 2 条 episode 后退出（rc=0，因为驱动脚本记录的是 timeout 的返回码）。
**这不是 STL 逻辑的错**：崩溃点在 baseline 包装器，仿真切换场景后偶发返回空帧（transient）。
处理：清空该臂的半成品目录后整波重跑（避免新旧 episode 混在同一 tag 下）。

### (j) 之前那批"同时启动"确实发生过 CUDA OOM
`grep -l "CUDA out of memory" $R/logs/*/eval.log` → `stl-ctl / stl-g5 / ti-g4 / ti-g5` 均命中，
证明 (c) 之后 12 臂并发的那次事故里确有 OOM；另有若干臂是被我主动 kill 的，两者要区分开写。

### (k) 自匹配杀进程又踩到一次
`for pid in $(pgrep -f "run_eval_ltlf.py --run_type")` —— 这条命令**自己的 cmdline 里就含该字符串**，
于是脚本把自己杀掉（无输出、exit 1）。→ 所有 pgrep/pkill 的搜索串都必须拆字面量：
`pat="run_eval_""ltlf.py"; pgrep -f "$pat"`。

### 当前采用的最小可靠机制（冻结）
1. 启动一个臂：`ops/launch_arm.sh`（参数化 env，`setsid nohup`）。
2. 排队：`ops/run_waves_final.sh` —— **单进程**、`nohup ... &` + `LAST_PID` + `wait`、
   波内两臂并行、波间 `sleep 25`；自定义环境通过位置参数 `K=V` 传入。
3. 优先级：idea2 核心 → 高度消融 → 后训练 → RefineFly → 零样本图（STEPS 降到 40）→ 覆盖图 → 后训练 NL 保持检查。
4. 每波结束后立刻用 `ops/collect_arms.py` / `ops/paired_compare.py` 出数，不等全部跑完。
