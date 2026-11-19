#!/bin/bash
# 单 episode 闭环冒烟：Xvfb + 仿真服务端 + eval
set -u
R=/mnt/kai_ckp/yiding/MaxHE
C=/mnt/world_foundational_model/kai/MaxHE/AeroVLA
V=$R/envs_py/aerovla_py310
PY=$V/bin/python
MAP=${MAP:-ModernCityMap}
GPU=${GPU:-1}          # 物理 GPU
PORT=${PORT:-30000}
STEPS=${STEPS:-30}
L=$R/logs/smoke-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$L"

export HOME=$R
export XDG_RUNTIME_DIR=$R/tmp/xdg-smoke
export XDG_CACHE_HOME=$R/hf_cache
export HF_HOME=$R/hf_cache
export TMPDIR=$R/tmp
export TORCH_HOME=$R/tmp/torch
export TRITON_CACHE_DIR=$R/tmp/triton
export DISPLAY=:99
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$XDG_RUNTIME_DIR" "$TMPDIR"; chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null
log(){ echo "[$(date '+%F %T')] $*"; }

log "LOG 目录: $L"
log "地图=$MAP 物理GPU=$GPU 端口=$PORT"

# ---- 0. 造 1 条 episode 的 split ----
SPL=$C/data/uav_dataset/seen_valset_splits/$MAP.json
SMOKE_SPL=$C/data/uav_dataset/_smoke_splits/${MAP}_1ep.json
mkdir -p "$(dirname "$SMOKE_SPL")"
"$PY" - "$SPL" "$SMOKE_SPL" <<'PYEOF'
import json, sys, collections
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
traj = d[0]["json"].split("/")[1]
keep = [e for e in d if e["json"].split("/")[1] == traj]
json.dump(keep, open(dst, "w"))
print("  split: 总 %d 条 frame 记录，取第 1 条轨迹 %s 的 %d 条 -> %s" % (len(d), traj, len(keep), dst))
PYEOF

# ---- 1. Xvfb ----
unset LD_PRELOAD
rm -f /tmp/.X99-lock
pgrep -f "Xvfb :99" >/dev/null || { nohup Xvfb :99 -screen 0 1280x720x24 -nolisten tcp > "$L/xvfb.log" 2>&1 & sleep 4; }
DISPLAY=:99 xdpyinfo >/dev/null 2>&1 && log "Xvfb OK" || { log "Xvfb 失败"; tail -5 "$L/xvfb.log"; exit 1; }

# ---- 2. Vulkan 适配器索引 ----
ADAPTER=$("$R/envs_py/downloader/bin/python" "$R/vulkan_adapter.py" "$GPU" 2>&1)
log "物理 GPU$GPU -> Vulkan 适配器 $ADAPTER"

# ---- 3. 仿真服务端 ----
pkill -9 -f "AirVLNSimulatorServerTool.py --port $PORT" 2>/dev/null
pkill -9 -f "$C/airsim_plugin/settings/" 2>/dev/null
sleep 2
cd "$C/airsim_plugin" || exit 1
LD_PRELOAD=$R/sim-runtime/lib/libfakerootuid.so nohup "$PY" -u AirVLNSimulatorServerTool.py \
    --port "$PORT" --gpus "$ADAPTER" --root_path "$R/envs" > "$L/server.log" 2>&1 &
SPID=$!
log "服务端 pid=$SPID"
port_open(){ "$R/envs_py/downloader/bin/python" -c "
import socket,sys
s=socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null; }
OK=0
for i in $(seq 1 30); do
  sleep 3
  if port_open; then OK=1; break; fi
  kill -0 $SPID 2>/dev/null || { log "服务端退出"; tail -15 "$L/server.log"; break; }
done
[ "$OK" = 1 ] && log "服务端已监听 $PORT" || { log "服务端没起来"; exit 1; }

# ---- 4. 跑评测 ----
cd "$C" || exit 1
SAVE=$R/eval_results/_smoke/seen_valset/$MAP
mkdir -p "$SAVE"
log "开始跑单 episode ..."
CUDA_VISIBLE_DEVICES=$GPU timeout 1800 "$PY" -u src/vlnce_src/eval_aerovla.py \
    --run_type eval --name AeroVLA_Smoke \
    --gpu_id "$ADAPTER" \
    --simulator_tool_port "$PORT" \
    --DDP_MASTER_PORT 29600 \
    --batchSize 1 \
    --maxWaypoints "$STEPS" \
    --dataset_path "dataset_raw/" \
    --eval_save_path "$SAVE" \
    --model_path "$C/checkpoints/aero_vla" \
    --model_base "$C/openvla-7b" \
    --eval_json_path "$SMOKE_SPL" \
    --map_spawn_area_json_path "$C/data/meta/map_spawnarea_info.json" \
    --object_name_json_path "$C/data/meta/object_description.json" \
    > "$L/eval.log" 2>&1
RC=$?
log "评测退出码 $RC"
log "--- eval.log 尾部 25 行 ---"; tail -25 "$L/eval.log"
log "--- 产物 ---"; find "$SAVE" -maxdepth 1 -type d | head

# ---- 5. UE4 是否用了正确的设备 ----
UELOG=$(find "$R/envs" -path "*Saved/Logs/*.log" -newermt "-30 minutes" 2>/dev/null | head -1)
if [ -n "$UELOG" ]; then
  log "UE4 日志: $UELOG"
  grep -m2 -E "Using Device|Found [0-9]+ device" "$UELOG" 2>/dev/null
else
  log "没找到 UE4 日志"
fi

# ---- 6. 收尾 ----
log "清理仿真进程"
kill -9 $SPID 2>/dev/null
pkill -9 -f "AirVLNSimulatorServerTool.py --port $PORT" 2>/dev/null
pkill -9 -f "$C/airsim_plugin/settings/" 2>/dev/null
log "完成。日志在 $L"
