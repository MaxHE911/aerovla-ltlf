#!/usr/bin/env bash
# Third wave sequence: evaluate the Stage-A logic-conditioned SFT checkpoint against the original
# AeroVLA SFT checkpoint, in the same session, on the same episodes.
#
# Wave 1 (sa_order):  the staged-prompt arm -- does fine-tuning on per-stage instructions help when
#                     the runtime actually injects per-stage instructions?
# Wave 2 (sa_single): the single-goal arm -- does the fine-tune damage the natural-language ability
#                     (the failure mode Logic-VLA reports for Stage 1)?
set -u
R=/mnt/kai_ckp/yiding/MaxHE
REPO=/mnt/world_foundational_model/kai/MaxHE/ltlf
CKPT=$REPO/work/posttrain/ckpt_stageA
BASE=$REPO/AeroVLA/checkpoints/aero_vla
cd "$REPO" || exit 1
mkdir -p "$R/logs"
TSK=work/tasks/tasks_ModernCityMap_seen_valset.json
LOGF=$R/logs/waves_seq3.log

log(){ echo "[$(date '+%F %T')] $*" >> "$LOGF"; }

for pat in ${WAIT_FOR:-}; do
  log "waiting for: $pat"
  while pgrep -f "$pat" > /dev/null 2>&1; do sleep 20; done
  log "clear: $pat"
done

if [ ! -f "$CKPT/adapter_model.safetensors" ]; then
  log "Stage-A checkpoint missing at $CKPT -- nothing to evaluate, exiting"
  exit 1
fi

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
  log "=== wave $label START modes=$ma/$mb ==="
  local tA="$label-a" tB="$label-b" pA pB
  pA=$(arm "$tA" "$ma" "$pa" "$split" "$tasks" "$ea")
  sleep 6
  pB=$(arm "$tB" "$mb" "$pb" "$split" "$tasks" "$eb")
  log "wave $label pids A=$pA B=$pB"
  wait "$pA" 2>/dev/null; log "wave $label: $tA done"
  wait "$pB" 2>/dev/null; log "wave $label: $tB done"
  sleep 20
}

# ---- 1. staged-prompt arm: Stage-A weights vs original weights ----
wave sa_order 30910 30920 work/splits/rt12.json "$TSK" fixed_order fixed_order \
     "MODEL_PATH=$CKPT" "MODEL_PATH=$BASE"

# ---- 2. single-goal arm: did the fine-tune hurt the natural-language ability? ----
wave sa_single 30930 30940 work/splits/rt8.json "$TSK" single_stage single_stage \
     "MODEL_PATH=$CKPT" "MODEL_PATH=$BASE"

log "=== seq3 finished ==="
