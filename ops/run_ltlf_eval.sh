#!/usr/bin/env bash
# One evaluation arm with the LTLf logic layer: Xvfb + UE4 simulator + run_eval_ltlf.py.
#
# Usage (on the server, from the repository root):
#   TAG=s3-ours MODE=fixed_order GPU=0 PORT=30000 STEPS=80 \
#   SPLIT=work/splits/stage3_12.json TASK_JSON=work/tasks/tasks_ModernCityMap_seen_valset.json \
#   bash ops/run_ltlf_eval.sh
#
# Env knobs (all optional except TAG/MODE/SPLIT):
#   TAG          output label, used for log + result directories
#   MODE         original | single_stage | long_instruction | fixed_order
#   GPU          physical GPU index (nvidia-smi order)
#   PORT         simulator socket port
#   STEPS        maxWaypoints
#   BATCH        batch size (default 1)
#   SPLIT        eval split json (built by ops/make_split.py)
#   TASK_JSON    M0 tasks json (required for fixed_order)
#   GATE         1/0 override the mode default
#   TAKEOVER     1/0 override the mode default
#   DISABLE_DIH  1/0 override "distance increasing for 10 frames" heuristic
#   SAVE_ROOT    result root (default <data>/eval_results/_ltlf)
#   SKIP_SERVER  1 = reuse an already running simulator on $PORT
set -u

R=/mnt/kai_ckp/yiding/MaxHE
REPO=$(cd "$(dirname "$0")/.." && pwd)
C=$REPO/AeroVLA
V=$R/envs_py/aerovla_py310
PY=$V/bin/python
DLPY=$R/envs_py/downloader/bin/python

TAG=${TAG:?TAG is required}
MODE=${MODE:-single_stage}
GPU=${GPU:-0}
PORT=${PORT:-30000}
STEPS=${STEPS:-80}
BATCH=${BATCH:-1}
SPLIT=${SPLIT:?SPLIT is required}
TASK_JSON=${TASK_JSON:-}
GATE=${GATE:-}
TAKEOVER=${TAKEOVER:-}
DISABLE_DIH=${DISABLE_DIH:-}
SAVE_ROOT=${SAVE_ROOT:-$R/eval_results/_ltlf}
SKIP_SERVER=${SKIP_SERVER:-0}

# the evaluation runs with cwd=AeroVLA, so every path we pass must be absolute
abspath(){ case "$1" in /*) echo "$1";; *) echo "$(cd "$(dirname "$1")" 2>/dev/null && pwd)/$(basename "$1")";; esac; }
SPLIT=$(abspath "$SPLIT")
if [ ! -f "$SPLIT" ]; then echo "SPLIT not found: $SPLIT" >&2; exit 2; fi
if [ -n "$TASK_JSON" ]; then
  TASK_JSON=$(abspath "$TASK_JSON")
  if [ ! -f "$TASK_JSON" ]; then echo "TASK_JSON not found: $TASK_JSON" >&2; exit 2; fi
fi

L=$R/logs/$TAG
SAVE=$SAVE_ROOT/$TAG
LOGD=$R/ltlf_logs/$TAG
mkdir -p "$L" "$SAVE" "$LOGD"

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

log "REPO=$REPO TAG=$TAG MODE=$MODE GPU=$GPU PORT=$PORT STEPS=$STEPS BATCH=$BATCH"
log "SPLIT=$SPLIT"
log "TASK_JSON=${TASK_JSON:-<none>} GATE=${GATE:-<mode default>} TAKEOVER=${TAKEOVER:-<mode default>} DISABLE_DIH=${DISABLE_DIH:-<mode default>}"
log "SAVE=$SAVE LOGS=$LOGD"
log "n_episodes_in_split=$("$PY" -c "import json,sys;print(len(json.load(open('$SPLIT'))))")"
log "gpu_mem_before=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

# ---- Xvfb ----
unset LD_PRELOAD
if ! DISPLAY=:99 xdpyinfo >/dev/null 2>&1; then
  rm -f /tmp/.X99-lock
  nohup Xvfb :99 -screen 0 1280x720x24 -nolisten tcp > "$L/xvfb.log" 2>&1 &
  sleep 4
fi
DISPLAY=:99 xdpyinfo >/dev/null 2>&1 && log "Xvfb OK" || { log "Xvfb FAILED"; exit 1; }

# ---- simulator ----
ADAPTER=$("$DLPY" "$R/vulkan_adapter.py" "$GPU" 2>&1)
log "physical GPU$GPU -> Vulkan adapter $ADAPTER"
SPID=""
# the simulator server spawns UE4 game processes; both must be killed or they pile up and
# saturate the machine (measured: 8 leaked UE4 instances -> load average 366)
kill_sim(){ pkill -9 -f "AirVLNSimulatorServerTool.py --port $PORT" 2>/dev/null
            pkill -9 -f "$C/airsim_plugin/settings/$PORT/" 2>/dev/null; }
kill_sim
if [ "$SKIP_SERVER" = "1" ]; then
  log "SKIP_SERVER=1: reusing the simulator already listening on $PORT"
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
  if [ "$OK" != 1 ]; then log "server did not listen on $PORT"; kill -9 $SPID 2>/dev/null; exit 1; fi
  log "server listening on $PORT"
fi

# ---- evaluation ----
cd "$C" || exit 1
EXTRA=()
[ -n "$GATE" ] && EXTRA+=(--ltlf-gate "$GATE")
[ -n "$TAKEOVER" ] && EXTRA+=(--ltlf-takeover "$TAKEOVER")
[ -n "$DISABLE_DIH" ] && EXTRA+=(--ltlf-disable-distance-heuristic "$DISABLE_DIH")
[ -n "$TASK_JSON" ] && EXTRA+=(--ltlf-task-json "$TASK_JSON")

log "starting eval (mode=$MODE) ..."
CUDA_VISIBLE_DEVICES=$GPU timeout 14400 "$PY" -u "$REPO/ops/run_eval_ltlf.py" \
    --run_type eval --name "ltlf_$MODE" \
    --gpu_id "$ADAPTER" \
    --simulator_tool_port "$PORT" \
    --DDP_MASTER_PORT $((29600 + PORT - 30000)) \
    --batchSize "$BATCH" \
    --maxWaypoints "$STEPS" \
    --dataset_path "dataset_raw/" \
    --eval_save_path "$SAVE" \
    --model_path "$C/checkpoints/aero_vla" \
    --model_base "$C/openvla-7b" \
    --eval_json_path "$SPLIT" \
    --map_spawn_area_json_path "$C/data/meta/map_spawnarea_info.json" \
    --object_name_json_path "$C/data/meta/object_description.json" \
    --ltlf-mode "$MODE" \
    --ltlf-log-dir "$LOGD" \
    --ltlf-run-summary "$LOGD/run_summary.json" \
    "${EXTRA[@]}" \
    > "$L/eval.log" 2>&1
RC=$?
log "eval exit code $RC"
log "--- eval.log tail 40 ---"; tail -40 "$L/eval.log"
log "--- ltlf log dir ---"; ls "$LOGD" | head -5; echo "..."; ls "$LOGD" | wc -l
log "--- result dirs ---"; find "$SAVE" -maxdepth 1 -mindepth 1 | head -5; find "$SAVE" -maxdepth 1 -mindepth 1 | wc -l

UELOG=$(find "$R/envs" -path "*Saved/Logs/*.log" -newermt "-40 minutes" 2>/dev/null | head -1)
[ -n "$UELOG" ] && { log "UE4 log: $UELOG"; grep -m2 -E "Using Device|Found [0-9]+ device" "$UELOG" 2>/dev/null; }

if [ -n "$SPID" ]; then
  log "stopping simulator + UE4"
  kill -9 $SPID 2>/dev/null
  kill_sim
else
  log "leaving the reused simulator running"
fi
log "done. logs in $L ; results in $SAVE ; ltlf logs in $LOGD"
