#!/usr/bin/env bash
# Second wave sequence: the safety-envelope altitude ablation + the remaining seen maps.
#
# Waits for the FIRST sequence process to disappear, so this one never runs concurrently with it.
# (A dependency chain built from per-arm waiters was abandoned: a waiter polling for a pattern that
# does not exist yet concludes "already finished" and starts immediately - nine arms then OOM'ed.)
set -u
R=/mnt/kai_ckp/yiding/MaxHE
REPO=/mnt/world_foundational_model/kai/MaxHE/ltlf
cd "$REPO" || exit 1
mkdir -p "$R/logs"
TSK=work/tasks/tasks_ModernCityMap_seen_valset.json
LOGF=$R/logs/waves_seq2.log

log(){ echo "[$(date '+%F %T')] $*" >> "$LOGF"; }

for pat in ${WAIT_FOR:-}; do
  log "waiting for: $pat"
  while pgrep -f "$pat" > /dev/null 2>&1; do sleep 20; done
  log "clear: $pat"
done

arm(){
  local tag=$1 mode=$2 port=$3 split=$4 tasks=$5 extra=${6:-}
  # shellcheck disable=SC2086
  setsid nohup env TAG="$tag" MODE="$mode" PORT="$port" SPLIT="$split" TASK_JSON="$tasks" \
      GPU=0 STEPS=80 $extra \
      bash ops/run_ltlf_eval.sh > "$R/logs/$tag.driver.log" 2>&1 < /dev/null &
  echo $!
}

wave(){
  local label=$1 pa=$2 pb=$3 split=$4 tasks=$5 ma=$6 mb=$7 ea=${8:-} eb=${9:-}
  log "=== wave $label START split=$split modes=$ma/$mb ==="
  local tA="$label-a" tB="$label-b" pA pB
  pA=$(arm "$tA" "$ma" "$pa" "$split" "$tasks" "$ea")
  sleep 6
  pB=$(arm "$tB" "$mb" "$pb" "$split" "$tasks" "$eb")
  log "wave $label pids A=$pA B=$pB"
  wait "$pA" 2>/dev/null; log "wave $label: $tA done"
  wait "$pB" 2>/dev/null; log "wave $label: $tB done"
  sleep 20
}

# ---- 1. safety-envelope ablation: raise the altitude floor from the expert p01 (1.72 m) to 6 m ----
# Motivation (measured, see work/idea2-stl.md): episodes that collide cruise at 4.2-4.7 m mean
# altitude, episodes that succeed at 8.4-8.5 m.  If the shield keeps the drone at >= 6 m, the
# collision rate should drop -- this arm tests exactly that with the same STL layer.
wave stlalt 30810 30820 work/splits/rt12.json "$TSK" stl stl \
     "LTLF_EXTRA_ARGS=--ltlf-config-stl_alt_min_m 6.0" \
     "LTLF_EXTRA_ARGS=--ltlf-config-stl_alt_min_m 1.722"

# ---- 2..4. remaining seen maps (idea 1 vs baseline+long instruction) ----
wave nyc 30500 30510 work/splits/NewYorkCity_rt8.json work/tasks/tasks_NewYorkCity_seen_valset.json \
     fixed_order long_instruction
wave ti 30600 30610 work/splits/TropicalIsland_st8.json work/tasks/tasks_TropicalIsland_seen_valset.json \
     fixed_order long_instruction
wave nycem 30700 30710 work/splits/NYCEnvironmentMegapa_rt6.json work/tasks/tasks_NYCEnvironmentMegapa_seen_valset.json \
     fixed_order long_instruction

log "=== seq2 finished ==="
