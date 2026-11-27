#!/usr/bin/env bash
# Wait until the given process patterns disappear, then launch ONE detection arm detached.
#
# All parameters come from the environment on purpose: ``pgrep -f`` matches /proc/<pid>/cmdline, so
# a wait pattern passed as an *argument* would match the waiter itself and deadlock.  Environment
# variables are not in cmdline, hence this looks odd but is required.
#
#   WAIT="mp-g5 mp-g4" TAG=stl-g5 MODE=stl PORT=30300 SPLIT=work/splits/rt12.json \
#   TASK_JSON=work/tasks/tasks_ModernCityMap_seen_valset.json \
#   setsid nohup env WAIT="mp-g5 mp-g4" TAG=stl-g5 ... bash ops/run_next.sh &
#
# Extra per-arm env: pass as EXTRA_ENV="MODEL_PATH=/path k2=v2".
set -u
: "${TAG:?TAG}"; : "${MODE:?MODE}"; : "${PORT:?PORT}"; : "${SPLIT:?SPLIT}"
R=/mnt/kai_ckp/yiding/MaxHE
REPO=/mnt/world_foundational_model/kai/MaxHE/ltlf
WAIT=${WAIT:-}
TASK_JSON=${TASK_JSON:-}
GPU=${GPU:-0}
STEPS=${STEPS:-80}
EXTRA_ENV=${EXTRA_ENV:-}
mkdir -p "$R/logs"
LOG=$R/logs/waiter-$TAG.log
{
  echo "[$(date '+%F %T')] waiter for TAG=$TAG waits on: ${WAIT:-<none>}"
  for pat in $WAIT; do
    while pgrep -f "$pat" > /dev/null 2>&1; do sleep 15; done
    echo "[$(date '+%F %T')] clear: $pat"
  done
  sleep 8
  cd "$REPO" || exit 1
  # shellcheck disable=SC2086
  setsid nohup env TAG="$TAG" MODE="$MODE" PORT="$PORT" SPLIT="$SPLIT" TASK_JSON="$TASK_JSON" \
      GPU="$GPU" STEPS="$STEPS" $EXTRA_ENV \
      bash ops/run_ltlf_eval.sh > "$R/logs/$TAG.driver.log" 2>&1 < /dev/null &
  echo "[$(date '+%F %T')] launched TAG=$TAG MODE=$MODE PORT=$PORT pid=$!"
} >> "$LOG" 2>&1
