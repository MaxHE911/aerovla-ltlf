# 阶段 0 审计：baseline 真实调用链、接入点与风险清单

> 依据：2026-11-20 03:0x 逐行阅读 `/mnt/world_foundational_model/kai/MaxHE/AeroVLA`（= 仓库 `AeroVLA/`，上游 `b87aaa7`），
> 并以本机 `work/baseline_ref/` 镜像（同一份代码）为准。行号为当前代码实测，不照抄文档。

## 1. 真实调用链（每个控制步）

```
eval_aerovla.py:97 __main__
 ├─ setup()                                        # 随机种子 / 分布式初始化
 ├─ initialize_env_eval(dataset_path, save_path, eval_json_path)
 │   └─ AirVLNENV.__init__                          # env_uav.py:78
 │       ├─ prepare_object_map()  ← --map_spawn_area_json_path
 │       └─ load_my_datasets()                      # env_uav.py:114
 │           ├─ split json 逐条 {json: "Map/<seq>/merged_data.json", frame: N}
 │           ├─ mark.json → object_name / target.position
 │           ├─ find_closest_area(target)           # env_uav.py:61，用 area[0]+1,area[1]+1,area[2]+0.5
 │           └─ object_position = [area[9], area[10], area[11]]   ← 全仓库唯一"最终目标"赋值（env_uav.py:139）
 │              trajectory = merged_data['trajectory_raw_detailed']（实测 69 帧）
 │              instruction = merged_data['conversations'][0]['value']
 ├─ AerialVLAWrapper(model_args, data_args)         # OpenVLA-7B + PEFT adapter
 ├─ Assist(always_help=False, use_gt=False)         # 仅深度碰撞/卡死监测，不给导航提示
 └─ eval()                                          # eval_aerovla.py:31
     ├─ next_minibatch() → batch（batchSize 条）
     ├─ EvalBatchState(...)                         # closeloop_util.py:113
     │   ├─ target_positions = [b['object_position']]      ← 固定最终目标
     │   ├─ instructions     = [b['instruction']]
     │   └─ _initialize_batch_data()                # env.reset() → changeToNewTrajectorys()
     │        ├─ _changeEnv(): 起/复用 UE4 场景（AirVLNSimulatorClientTool）
     │        ├─ _setTrajectorys(): setPoses ×3 + getSensorInfo
     │        ├─ _setObjects(): 放置目标物体
     │        └─ observations = get_obs()；episodes[i].append(observations[i][-1])
     ├─ inputs = model_wrapper.prepare_inputs(episodes, target_positions, instructions)
     │    per ep（aerovla_wrapper_ui.py:78-151）：
     │      dir_text = get_semantic_direction(state, target_pos)   # 7 桶方位词，每帧重算
     │      obj_desc = instructions[i].split("degrees from you.")[1].split(" Please control")[0].strip()
     │      prompt   = "<image>\nFly {dir_text}and find the target. {obj_desc}\nAction: "
     └─ for t in range(maxWaypoints+1):                             # eval_aerovla.py:60
         ├─ check_batch_termination(t)              # closeloop_util.py:218（写盘 + 全部 skip 则退出）
         ├─ batch_actions, model_stops = model_wrapper.run(...)      # LAND = 文本含 LAND 或三个动作全 0
         ├─ eval_env.makeActions(batch_actions)     # env_uav.py:416 → airsim_plugin move_path_by_actions
         │    ├─ fwd = 沿 (当前yaw+pred_yaw) 的位移(米)，down = Δz，|yaw|>=0.25 rad 时先原地转到目标 yaw
         │    ├─ 返回 5 个 state（实测是同一个 state 的 5 份拷贝，见风险 1）
         │    ├─ oracle_success：本步末尾距 object_position < 20 m
         │    └─ sim_states.trajectory.extend(5 个 state)；update_measurements()
         ├─ outputs = eval_env.get_obs()
         ├─ batch_state.update_from_env_output(outputs)              # closeloop_util.py:162
         │    ├─ Assist.check_collision_by_depth：任一深度图 >10% 像素 <=1 → collisions/dones
         │    ├─ 卡死：xy 位移 < 0.05 m 连续 >15 步 → collisions=True, dones=True
         │    ├─ episodes[i] 追加全部 5 个观测
         │    ├─ distance_to_ends.append(距最终目标的距离)
         │    └─ 连续 10 次距离不减 → collisions=True, dones=True    ← 长任务需关闭的启发式
         ├─ batch_state.predict_dones = model_stops                  ← LAND 门控注入点
         ├─ batch_state.update_metric()             # closeloop_util.py:201
         │    LAND 且 距最终目标<=20 且未 early_end → success；>20 → early_end
         │    oracle_success 且 early_end → dones；success → dones
         └─ inputs = prepare_inputs(...)            # 每帧重算方位词，目标点不变
```

评测产物（`save_to_dataset_eval`，closeloop_util.py:41）：每个 episode 一个目录
`<eval_save_path>/[success_|oracle_]<seq>/`，内含 `log/000000.json…`（每宏步 5 份 + 初始 1 份）、
5 路 rgb/depth png、`object_description.json`、`ori_info.json`（含 `ori_traj_dir` 相对路径）。

指标（`utils/metric.py`）：SR/OSR 由目录名前缀统计；NE = 末帧位置 vs 专家末帧位置；
SPL = (专家路径长-20)/max(专家路径长-20, 预测路径长)，失败记 0；**必须在仓库根目录执行**（相对路径）。

## 2. 逻辑层接入点（已核对，均为"加参数/加外层"而非改模板）

| 注入 | 位置 | baseline 取值 | 我们取值 | 是否改 baseline 文件 |
|---|---|---|---|---|
| 引导点 p* | `prepare_inputs(target_positions=...)` 入参（baseline 在 closeloop_util.py:120 赋值） | 固定最终目标 | 每阶段地标坐标 | 否（runner 传自己的列表） |
| 物体描述 c | `instructions` 入参（aerovla_wrapper_ui.py:93 从指令切出） | 原指令切出 | 每阶段构造带两个锚点的指令串 | 否（runner 传合成指令） |
| LAND 门控 | `batch_state.predict_dones = model_stops`（eval_aerovla.py:79） | 直接赋值 | 残余未接受时置 False（并记录） | 否（runner 自己写循环） |
| 距离递增启发式 | `closeloop_util.target_distance_increasing_for_10frames` | 开启 | 长任务组关闭（显式配置 + 记录） | 否（模块级替换，runner 内显式做） |
| 事件检测 | `get_obs()` 返回的 `outputs`（每 env 5 份观测） | — | 用 `sensors.state.position` 判 12 m 进入 | 否 |

结论：**原版 `AeroVLA/` 一个文件都不改**；逻辑层用独立 runner 复用 baseline 的
`AirVLNENV / EvalBatchState / AerialVLAWrapper`，只在循环层面注入上述 4 个值。
这样 `git diff baseline-v0 -- AeroVLA/` 恒为空，baseline 组可直接用原 `eval_aerovla.py` 跑，对照最干净。

## 3. 风险/坑清单（都是实测发现，不是文档复述）

1. **"每个宏步 5 个子位姿"没有额外信息**：`move_path_by_actions` 用 `for _ in range(5)` 把同一个
   `state_sensor.retrieve()` 结果追加 5 次（airims_plugin:388-391）。所以一个宏步只有 1 个真实位置，
   离线回放要去重连续相同位置，否则会给逻辑层喂 5 倍冗余时间点。
2. **split 的 `frame` 字段被忽略**：`load_my_datasets` 只用 `merged_data.json` 整条轨迹；
   ModernCityMap 的 split 有 2873 条 frame 记录但只有 **124 条唯一轨迹** → 同一 episode 会被重复评测。
   本项目的实验必须按轨迹去重（否则 SR 分母虚高且重复计权）。
3. **绝对 `--dataset_path` 会丢前导 `/`**（env_uav.py:146 `'/'.join(path_parts[:-1])`）：
   必须 `cd` 到仓库根目录并用相对路径 `dataset_raw/`；`utils/metric.py` 同理。
4. **`eval_save_path` 有断点续跑语义**：`load_my_datasets` 会把 save_path 下已存在的目录对应的轨迹跳过，
   所以每次实验必须用全新输出目录。
5. **LAND 判据含"零动作"分支**：`has_land = "LAND" in text or (fwd<0.01 and down<0.01 and yaw<0.01)`。
   解析失败也会退化成 0,0,0 → 误判 LAND。门控必须能区分"真到点"和"解析失败/悬停"。
6. **零动作=卡死**：xy 不动连续 >15 步直接判 collision+done。所以"只拦 LAND 继续发 0 动作"必然失败，
   必须做保底推进（idea1 M5 的 L2）。
7. **连续 10 次距离不减→失败**（closeloop_util.py:194）会写公共 collisions/dones；绕行中间地标时
   距最终目标必然变远，长任务组必须统一关闭（baseline+长指令 与 ours 同设置）。
8. **一次 `EvalBatchState` 只覆盖一个 minibatch**：循环在外层 `next_minibatch()`。
9. **距离/成功全部对"最终目标"计算**：逻辑层必须另存 `guide_positions` 与最终目标分离，
   否则阶段切换会污染 NE/SPL/SR。保存日志要同时留最终目标与当步引导点。
10. **batchSize=1 时** `_changeEnv` 的复用逻辑（`this_scene_used_cnt`）依赖同一场景；换图会重启 UE4。
11. **UE4 Vulkan 适配器索引 ≠ nvidia-smi 索引**，必须用 `vulkan_adapter.py` 推导；
    起 Xvfb 时不能预加载 `libfakerootuid.so`。
12. **两个 GPU 常被别人占**：启动前看 `memory.used`，显式 `CUDA_VISIBLE_DEVICES`。
13. **评测 split 只覆盖 ModernCityMap（124 条轨迹）**，其它地图缺 `dataset_raw` → 第一轮只跑 ModernCityMap。
14. **`assist.check_collision_by_depth` 只看深度图**（>10% 像素 ≤1 m 判碰撞），新逻辑层不改它。

## 4. 阶段 0 冒烟（原版 baseline，未改任何代码）

命令（服务器，仓库副本）：

```bash
cd /mnt/world_foundational_model/kai/MaxHE/ltlf
TAG=stage0-baseline MAP=ModernCityMap GPU=0 STEPS=20 bash ops/smoke_baseline.sh
```

结果：退出码 0；20 个宏步；距最终目标 122.3? m → **48.89 m**（线性下降）；终止原因
`Timeout (Max Steps Reached)`（maxWaypoints=20 时预期）；保存 101 个 log 帧（=20×5+1，与风险 1 一致）；
UE4 日志 `Using Device 0`（物理 GPU0 → Vulkan 适配器 0，与 `vulkan_adapter.py 0` 一致）；
产物目录 `/mnt/kai_ckp/yiding/MaxHE/eval_results/_ltlf/stage0-baseline/84eabab1-…/`。
日志：`/mnt/kai_ckp/yiding/MaxHE/logs/stage0-baseline/`（`eval.log` / `server.log` / driver log）。

结论：baseline 闭环可跑、结果可保存可重读，逻辑层接入点已定位，原版可回退（tag `baseline-v0`）。
