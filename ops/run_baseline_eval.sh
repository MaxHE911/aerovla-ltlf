#!/usr/bin/env bash
# G0 arm: the original, untouched baseline evaluation (src/vlnce_src/eval_aerovla.py).
#
# Usage (server, repo root):
#   TAG=g0-baseline GPU=0 PORT=30000 STEPS=80 SPLIT=work/splits/stage3_12ep.json bash ops/run_baseline_eval.sh
set -u

R=/mnt/kai_ckp/yiding/MaxHE
REPO=$(cd "$(dirname "$0")/.." && pwd)
C=$REPO/AeroVLA
V=$R/envs_py/aerovla_py310
PY=$V/bin/python
DLPY=$R/envs_py/downloader/bin/python

TAG=${TAG:?TAG is required}
GPU=${GPU:-0}
PORT=${PORT:-30000}
STEPS=${STEPS:-80}
BATCH=${BATCH:-1}
SPLIT=${SPLIT:?SPLIT is required}
SAVE_ROOT=${SAVE_ROOT:-$R/eval_results/_ltlf}
SKIP_SERVER=${SKIP_SERVER:-0}
MODEL_BASE=${MODEL_BASE:-$C/openvla-7b}
MODEL_PATH=${MODEL_PATH:-$C/checkpoints/aero_vla}

abspath(){ case "$1" in /*) echo "$1";; *) echo "$(cd "$(dirname "$1")" 2>/dev/null && pwd)/$(basename "$1")";; esac; }
SPLIT=$(abspath "$SPLIT")
[ -f "$SPLIT" ] || { echo "SPLIT not found: $SPLIT" >&2; exit 2; }

L=$R/logs/$TAG
SAVE=$SAVE_ROOT/$TAG
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

log "REPO=$REPO TAG=$TAG GPU=$GPU PORT=$PORT STEPS=$STEPS SPLIT=$SPLIT"
log "SAVE=$SAVE LOGS=$L"
log "n_episodes_in_split=$("$PY" -c "import json;print(len(json.load(open('$SPLIT'))))")"
log "gpu_mem_before=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

unset LD_PRELOAD
if ! DISPLAY=:99 xdpyinfo >/dev/null 2>&1; then
  rm -f /tmp/.X99-lock
  nohup Xvfb :99 -screen 0 1280x720x24 -nolisten tcp > "$L/xvfb.log" 2>&1 &
  sleep 4
fi
DISPLAY=:99 xdpyinfo >/dev/null 2>&1 && log "Xvfb OK" || { log "Xvfb FAILED"; exit 1; }

ADAPTER=$("$DLPY" "$R/vulkan_adapter.py" "$GPU" 2>&1)
log "physical GPU$GPU -> Vulkan adapter $ADAPTER"
SPID=""
# the simulator server spawns UE4 game processes; both must be killed or they pile up and
# saturate the machine (measured: 8 leaked UE4 instances -> load average 366)
kill_sim(){ pkill -9 -f "AirVLNSimulatorServerTool.py --port $PORT" 2>/dev/null
            pkill -9 -f "$C/airsim_plugin/settings/$PORT/" 2>/dev/null; }
kill_sim
if [ "$SKIP_SERVER" = "1" ]; then
  log "SKIP_SERVER=1: reusing simulator on $PORT"
else
  pkill -9 -f "AirVLNSimulatorServerTool.py --port $PORT" 2>/dev/null
  sleep 2
  cd "$C/airsim_plugin" || exit 1
  LD_PRELOAD=$R/sim-runtime/lib/libfakerootuid.so nohup "$PY" -u AirVLNSimulatorServerTool.py \
      --port "$PORT" --gpus "$ADAPTER" --root_path "$R/envs" > "$L/server.log" 2>&1 &
  SPID=$!
  log "server pid=$SPID"
  port_open(){ "$DLPY" -c "
import socket,sys
s=socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null; }
  OK=0
  for i in $(seq 1 40); do
    sleep 3
    if port_open; then OK=1; break; fi
    kill -0 $SPID 2>/dev/null || { log "server exited"; tail -20 "$L/server.log"; break; }
  done
  [ "$OK" = 1 ] || { log "server did not listen on $PORT"; kill -9 $SPID 2>/dev/null; exit 1; }
  log "server listening on $PORT"
fi

cd "$C" || exit 1
log "starting baseline eval ..."
CUDA_VISIBLE_DEVICES=$GPU timeout 14400 "$PY" -u src/vlnce_src/eval_aerovla.py \
    --run_type eval --name AeroVLA_Baseline \
    --gpu_id "$ADAPTER" \
    --simulator_tool_port "$PORT" \
    --DDP_MASTER_PORT $((29600 + PORT - 30000)) \
    --batchSize "$BATCH" \
    --maxWaypoints "$STEPS" \
    --dataset_path "dataset_raw/" \
    --eval_save_path "$SAVE" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --eval_json_path "$SPLIT" \
    --map_spawn_area_json_path "$C/data/meta/map_spawnarea_info.json" \
    --object_name_json_path "$C/data/meta/object_description.json" \
    > "$L/eval.log" 2>&1
RC=$?
log "baseline eval exit code $RC"
log "--- eval.log tail 20 ---"; tail -20 "$L/eval.log"
log "--- result dirs ---"; find "$SAVE" -maxdepth 1 -mindepth 1 | head -30; find "$SAVE" -maxdepth 1 -mindepth 1 | wc -l

if [ -n "$SPID" ]; then
  log "stopping simulator + UE4"
  kill -9 $SPID 2>/dev/null
  kill_sim
fi
log "done. logs in $L ; results in $SAVE"
