#!/usr/bin/env bash
# Final sequential wave driver.  Two bugs are fixed here, both observed in this session:
#
#  1. Dependency waiters that poll for a pattern which does not exist *yet* conclude "already done"
#     and launch immediately (nine arms started at once and OOM'ed).  -> one sequential process.
#  2. `setsid nohup cmd &` makes `$!` the PID of `setsid`, which exits immediately, so `wait "$pid"`
#     returns at once and every wave is skipped.  -> plain `nohup cmd &`, whose `$!` is the driver
#     script that actually runs the evaluation.
#
# Waves are ordered by value: idea-2 comparison, RefineFly comparison, the altitude ablation, then
# map coverage, then the post-training evaluation.
set -u
R=/mnt/kai_ckp/yiding/MaxHE
REPO=/mnt/world_foundational_model/kai/MaxHE/ltlf
cd "$REPO" || exit 1
mkdir -p "$R/logs"
TSK=work/tasks/tasks_ModernCityMap_seen_valset.json
CKPT=$REPO/work/posttrain/ckpt_stageA
BASE=$REPO/AeroVLA/checkpoints/aero_vla
LOGF=$R/logs/waves_final.log

log(){ echo "[$(date '+%F %T')] $*" >> "$LOGF"; }

for pat in ${WAIT_FOR:-}; do
  log "waiting for: $pat"
  while pgrep -f "$pat" > /dev/null 2>&1; do sleep 20; done
  log "clear: $pat"
done

arm(){ # arm <tag> <mode> <port> <split> <tasks> [K=V ...]
  local tag=$1 mode=$2 port=$3 split=$4 tasks=$5
  shift 5
  # shellcheck disable=SC2086
  nohup env TAG="$tag" MODE="$mode" PORT="$port" SPLIT="$split" TASK_JSON="$tasks" \
      GPU=0 STEPS=80 "$@" \
      bash ops/run_ltlf_eval.sh > "$R/logs/$tag.driver.log" 2>&1 < /dev/null &
  echo $!
}

wave(){ # wave <label> <portA> <portB> <split> <tasks> <modeA> <modeB> [extraA] [extraB]
  local label=$1 pa=$2 pb=$3 split=$4 tasks=$5 ma=$6 mb=$7 ea=${8:-} eb=${9:-}
  log "=== wave $label START modeA=$ma modeB=$mb split=$split ==="
  local tA="$label-a" tB="$label-b" pA pB
  # shellcheck disable=SC2086
  pA=$(arm "$tA" "$ma" "$pa" "$split" "$tasks" $ea)
  sleep 8
  # shellcheck disable=SC2086
  pB=$(arm "$tB" "$mb" "$pb" "$split" "$tasks" $eb)
  log "wave $label pids A=$pA($tA) B=$pB($tB)"
  wait "$pA"; log "wave $label: $tA exited rc=$?"
  wait "$pB"; log "wave $label: $tB exited rc=$?"
  sleep 25
}

# 1. idea 2 (STL: monitoring + deadline pressure + safety shield) vs idea 1 (same stage machine)
wave stl 30300 30310 work/splits/rt12.json "$TSK" stl fixed_order

# 2. RefineFly post-trained adapter vs the AeroVLA SFT checkpoint, identical mode and episodes
wave rf 30400 30410 work/splits/rt12.json "$TSK" original original \
     "MODEL_PATH=$R/models/refinefly" "MODEL_PATH=$BASE"

# 3. safety-envelope ablation: altitude floor 6.0 m vs the expert p01 of 1.722 m
wave stlalt 30810 30820 work/splits/rt12.json "$TSK" stl stl \
     "LTLF_EXTRA_ARGS=--ltlf-config-stl_alt_min_m 6.0" \
     "LTLF_EXTRA_ARGS=--ltlf-config-stl_alt_min_m 1.722"

# 4-6. remaining seen maps (idea 1 vs baseline + long instruction)
wave nyc 30500 30510 work/splits/NewYorkCity_rt8.json work/tasks/tasks_NewYorkCity_seen_valset.json \
     fixed_order long_instruction
wave ti 30600 30610 work/splits/TropicalIsland_st8.json work/tasks/tasks_TropicalIsland_seen_valset.json \
     fixed_order long_instruction
wave nycem 30700 30710 work/splits/NYCEnvironmentMegapa_rt6.json work/tasks/tasks_NYCEnvironmentMegapa_seen_valset.json \
     fixed_order long_instruction

# 7-8. idea 3: the Stage-A logic-conditioned SFT checkpoint vs the original checkpoint
if [ -f "$CKPT/adapter_model.safetensors" ]; then
  wave sa_order 30910 30920 work/splits/rt12.json "$TSK" fixed_order fixed_order \
       "MODEL_PATH=$CKPT" "MODEL_PATH=$BASE"
  wave sa_single 30930 30940 work/splits/rt8.json "$TSK" single_stage single_stage \
       "MODEL_PATH=$CKPT" "MODEL_PATH=$BASE"
else
  log "WARNING: Stage-A checkpoint missing at $CKPT, skipping waves 7-8"
fi

log "=== final sequence finished ==="
