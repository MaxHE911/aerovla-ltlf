#!/usr/bin/env bash
# Run a queue of multi-map experiment waves, two arms in parallel on GPU0.
# Each wave = (fixed_order = ours) x (long_instruction = baseline+long) on the same split.
# The script waits for both arms of a wave to exit before starting the next wave, so the GPU is
# never oversubscribed and UE4 processes cannot pile up.
#
#   nohup bash ops/run_wave_sequence.sh mp:work/splits/mp_rt14.json:work/tasks/tasks_ModularPark_unseen_map_valset.json ... &
#
# Wave spec format:  <tag>:<split>:<tasks_json>
set -u
R=/mnt/kai_ckp/yiding/MaxHE
REPO=/mnt/world_foundational_model/kai/MaxHE/ltlf
cd "$REPO" || exit 1
mkdir -p "$R/logs"

log(){ echo "[$(date '+%F %T')] $*"; }

# optional: wait until other arms (matched by pgrep -f) are gone, so GPU0 is not oversubscribed
for pat in ${WAIT_FOR:-}; do
  log "waiting for existing arm(s) matching: $pat"
  while pgrep -f "$pat" > /dev/null 2>&1; do sleep 20; done
  log "clear: $pat"
done

launch(){
  # launch <logname> <ENV=VAL ...>
  local name=$1; shift
  setsid nohup env "$@" bash ops/run_ltlf_eval.sh > "$R/logs/$name.driver.log" 2>&1 < /dev/null &
  echo $!
}

run_wave(){
  local tag=$1 split=$2 tasks=$3 p1=$4 p2=$5
  log "=== wave $tag start (split=$split) ==="
  local a b
  a=$(launch "${tag}-g5" TAG=${tag}-g5 MODE=fixed_order PORT=$p1 SPLIT="$split" TASK_JSON="$tasks" GPU=0 STEPS=80)
  sleep 6
  b=$(launch "${tag}-g4" TAG=${tag}-g4 MODE=long_instruction PORT=$p2 SPLIT="$split" TASK_JSON="$tasks" GPU=0 STEPS=80)
  log "wave $tag arms: g5 pid=$a g4 pid=$b"
  wait "$a" 2>/dev/null
  log "wave $tag: g5 done"
  wait "$b" 2>/dev/null
  log "wave $tag: g4 done"
  sleep 20   # let UE4/socket release
}

for spec in "$@"; do
  IFS=':' read -r tag split tasks <<< "$spec"
  run_wave "$tag" "$split" "$tasks" 30100 30110
  # second pair of ports, in case a wave left a socket in TIME_WAIT
  [ "${tag#*b}" = "$tag" ] || true
done
log "=== wave sequence finished ==="
