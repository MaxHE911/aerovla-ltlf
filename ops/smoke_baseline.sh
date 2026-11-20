#!/usr/bin/env bash
# Baseline closed-loop smoke test, run from the project repository copy of AeroVLA.
#
# Usage (on the server):
#   TAG=stage0-baseline MAP=ModernCityMap GPU=0 STEPS=20 bash ops/smoke_baseline.sh
#
# Env:
#   TAG     label for the output directory (default: smoke-<utc timestamp>)
#   MAP     map name; must exist in data/uav_dataset/seen_valset_splits/<MAP>.json
#   GPU     physical GPU index (nvidia-smi order)
#   STEPS   maxWaypoints (macro steps)
#   PORT    simulator socket port (default 30000)
#   SPLIT_EP  how many trajectories from the split file to keep (default: all frames of the 1st trajectory)
set -u

R=/mnt/kai_ckp/yiding/MaxHE
REPO=$(cd "$(dirname "$0")/.." && pwd)
C=$REPO/AeroVLA
V=$R/envs_py/aerovla_py310
PY=$V/bin/python
DONWLOADER_PY=$R/envs_py/downloader/bin/python

MAP=${MAP:-ModernCityMap}
GPU=${GPU:-0}
PORT=${PORT:-30000}
STEPS=${STEPS:-20}
TAG=${TAG:-smoke-$(date -u +%Y%m%dT%H%M%SZ)}

L=$R/logs/$TAG
SAVE=$R/eval_results/_ltlf/$TAG
mkdir -p "$L" "$SAVE"

export HOME=$R
export XDG_RUNTIME_DIR=$R/tmp/xdg-$TAG
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

log "REPO=$REPO"
log "LOG=$L  SAVE=$SAVE"
log "MAP=$MAP GPU=$GPU PORT=$PORT STEPS=$STEPS TAG=$TAG"

# ---- 0. build a small split from the first trajectory of the map's split file ----
SPL=$C/data/uav_dataset/seen_valset_splits/$MAP.json
if [ ! -f "$SPL" ]; then
  echo "MISSING SPLIT: $SPL" >&2; exit 2
fi
SMOKE_SPL=$C/data/uav_dataset/_smoke_splits/${MAP}_1ep.json
mkdir -p "$(dirname "$SMOKE_SPL")"
"$PY" - "$SPL" "$SMOKE_SPL" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
traj = d[0]["json"].split("/")[1]
keep = [e for e in d if e["json"].split("/")[1] == traj]
json.dump(keep, open(dst, "w"))
print("  split: %d frame records total, keeping %d of trajectory %s -> %s" % (len(d), len(keep), traj, dst))
PYEOF

# ---- 1. Xvfb ----
# UE4 refuses to run as root (it checks geteuid()==0 inside its Linux binary), so the simulator
# server must be started with a shim that reports an unprivileged uid. Built from
# ops/sim-runtime/fakerootuid.c; falls back to the prebuilt copy on the data disk.
PRELOAD=${PRELOAD:-$REPO/ops/sim-runtime/lib/libfakerootuid.so}
[ -f "$PRELOAD" ] || PRELOAD=$R/sim-runtime/lib/libfakerootuid.so
[ -f "$PRELOAD" ] || { echo "fakeroot shim not found (UE4 will refuse to run as root)" >&2; exit 2; }
log "fakeroot shim: $PRELOAD"
# The wrapper hardcodes ./openvla-7b (ignores --model_base), so when the parallel filesystem is
# saturated we expose the RAM copy at that path inside a private mount namespace (see
# ops/sim-runtime/with_shm_model.sh). Nothing on disk is modified.
SHMW=$REPO/ops/sim-runtime/with_shm_model.sh
RAM_MODEL=${RAM_MODEL:-/dev/shm/openvla-7b}
unset LD_PRELOAD
rm -f /tmp/.X99-lock
pgrep -f "Xvfb :99" >/dev/null || { nohup Xvfb :99 -screen 0 1280x720x24 -nolisten tcp > "$L/xvfb.log" 2>&1 & sleep 4; }
DISPLAY=:99 xdpyinfo >/dev/null 2>&1 && log "Xvfb OK" || { log "Xvfb FAILED"; tail -5 "$L/xvfb.log"; exit 1; }

# ---- 2. Vulkan adapter index for the physical GPU ----
ADAPTER=$("$DONWLOADER_PY" "$R/vulkan_adapter.py" "$GPU" 2>&1)
log "physical GPU$GPU -> Vulkan adapter $ADAPTER"

# ---- 3. simulator server ----
pkill -9 -f "AirVLNSimulatorServerTool.py --port $PORT" 2>/dev/null
sleep 2
cd "$C/airsim_plugin" || exit 1
LD_PRELOAD=$PRELOAD nohup "$PY" -u AirVLNSimulatorServerTool.py \
    --port "$PORT" --gpus "$ADAPTER" --root_path "$R/envs" > "$L/server.log" 2>&1 &
SPID=$!
log "server pid=$SPID"
port_open(){ "$DONWLOADER_PY" -c "
import socket,sys
s=socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null; }
OK=0
for i in $(seq 1 40); do
  sleep 3
  if port_open; then OK=1; break; fi
  kill -0 $SPID 2>/dev/null || { log "server exited"; tail -20 "$L/server.log"; break; }
done
if [ "$OK" != 1 ]; then log "server did not listen on $PORT"; kill -9 $SPID 2>/dev/null; exit 1; fi
log "server listening on $PORT"

# ---- 4. evaluation ----
cd "$C" || exit 1
log "starting eval ..."
CUDA_VISIBLE_DEVICES=$GPU "$SHMW" "$C/openvla-7b" "$RAM_MODEL" timeout 3600 "$PY" -u src/vlnce_src/eval_aerovla.py \
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
log "eval exit code $RC"
log "--- eval.log tail 30 ---"; tail -30 "$L/eval.log"
log "--- artifacts ---"; find "$SAVE" -maxdepth 1 -mindepth 1 | head -20
log "--- step log count ---"; find "$SAVE" -path "*/log/*.json" | wc -l

# ---- 5. which UE4 device was used ----
UELOG=$(find "$R/envs" -path "*Saved/Logs/*.log" -newermt "-40 minutes" 2>/dev/null | head -1)
if [ -n "$UELOG" ]; then
  log "UE4 log: $UELOG"; grep -m2 -E "Using Device|Found [0-9]+ device" "$UELOG" 2>/dev/null
else
  log "no UE4 log found"
fi

# ---- 6. cleanup ----
log "stopping simulator"
kill -9 $SPID 2>/dev/null
pkill -9 -f "AirVLNSimulatorServerTool.py --port $PORT" 2>/dev/null
log "done. logs in $L ; results in $SAVE"
