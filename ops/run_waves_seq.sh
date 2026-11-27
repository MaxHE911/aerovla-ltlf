#!/usr/bin/env bash
# Sequential multi-wave driver: one process, explicit `wait` on the two arms of each wave.
#
# A dependency-based waiter chain was tried first (ops/run_next.sh) and failed: a waiter that polls
# for a pattern which does not exist *yet* concludes "already finished" and starts immediately, so
# nine arms launched at once and every one of them died of CUDA OOM.  A single sequential process
# cannot have that race, so this is the mechanism of record.
#
#   TAG=<unique prefix> bash ops/run_waves_seq.sh &
#
# Each wave runs two arms in parallel on GPU0 (two distinct simulator ports) and blocks until both
# have exited; then the next wave starts.
set -u
R=/mnt/kai_ckp/yiding/MaxHE
REPO=/mnt/world_foundational_model/kai/MaxHE/ltlf
cd "$REPO" || exit 1
mkdir -p "$R/logs"
TSK=work/tasks/tasks_ModernCityMap_seen_valset.json
LOGF=$R/logs/waves_seq.log
WAIT_FOR=${WAIT_FOR:-}

log(){ echo "[$(date '+%F %T')] $*" >> "$LOGF"; }

# wait until the given patterns are really gone (arms of the previous session)
for pat in $WAIT_FOR; do
  log "waiting for: $pat"
  while pgrep -f "$pat" > /dev/null 2>&1; do sleep 20; done
  log "clear: $pat"
done

arm(){ # arm <tag> <mode> <port> <split> <tasks> [extra env]
  local tag=$1 mode=$2 port=$3 split=$4 tasks=$5 extra=${6:-}
  # shellcheck disable=SC2086
  setsid nohup env TAG="$tag" MODE="$mode" PORT="$port" SPLIT="$split" TASK_JSON="$tasks" \
      GPU=0 STEPS=80 $extra \
      bash ops/run_ltlf_eval.sh > "$R/logs/$tag.driver.log" 2>&1 < /dev/null &
  echo $!
}

wave(){ # wave <label> <portA> <portB> <split> <tasks> <modeA> <modeB> [extraA] [extraB]
  local label=$1 pa=$2 pb=$3 split=$4 tasks=$5 ma=$6 mb=$7 ea=${8:-} eb=${9:-}
  log "=== wave $label START split=$split modes=$ma/$mb ==="
  local tA="$label-a" tB="$label-b"
  local pA pB
  pA=$(arm "$tA" "$ma" "$pa" "$split" "$tasks" "$ea")
  sleep 6
  pB=$(arm "$tB" "$mb" "$pb" "$split" "$tasks" "$eb")
  log "wave $label pids: A=$pA($tA) B=$pB($tB)"
  wait "$pA" 2>/dev/null; log "wave $label: $tA done"
  wait "$pB" 2>/dev/null; log "wave $label: $tB done"
  sleep 20
}

# ---- 1. ModularPark: the official unseen-map split (zero training samples for this map) ----
wave mp 30200 30210 work/splits/mp_rt14.json work/tasks/tasks_ModularPark_unseen_map_valset.json \
     fixed_order long_instruction

# ---- 2. idea 2: STL layer vs idea 1 on identical episodes ----
wave stl 30300 30310 work/splits/rt12.json "$TSK" stl fixed_order

# ---- 3. RefineFly post-trained adapter vs the AeroVLA SFT checkpoint (same mode) ----
wave rf 30400 30410 work/splits/rt12.json - original original \
     "MODEL_PATH=$R/models/refinefly" "MODEL_PATH=AeroVLA/checkpoints/aero_vla"

# ---- 4. NewYorkCity ----
wave nyc 30500 30510 work/splits/NewYorkCity_rt12.json work/tasks/tasks_NewYorkCity_seen_valset.json \
     fixed_order long_instruction

# ---- 5. TropicalIsland (this map's tasks are almost all short-horizon) ----
wave ti 30600 30610 work/splits/TropicalIsland_st12.json work/tasks/tasks_TropicalIsland_seen_valset.json \
     fixed_order long_instruction

# ---- 6. NYCEnvironmentMegapa (largest map) ----
wave nycem 30700 30710 work/splits/NYCEnvironmentMegapa_rt12.json work/tasks/tasks_NYCEnvironmentMegapa_seen_valset.json \
     fixed_order long_instruction

log "=== all waves finished ==="
