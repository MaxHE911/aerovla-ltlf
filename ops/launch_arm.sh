#!/usr/bin/env bash
# Launch ONE detached evaluation arm. Survives the ssh session ending.
#   bash ops/launch_arm.sh TAG MODE PORT SPLIT [TASK_JSON] [EXTRA_KV ...]
# EXTRA_KV examples: GPU=1 STEPS=60 DISABLE_DIH=1 GATE=0 TAKEOVER=0 DATASET=dataset_raw/
set -u
TAG=${1:?TAG}
MODE=${2:?MODE}
PORT=${3:?PORT}
SPLIT=${4:?SPLIT}
TASK_JSON=${5:-}
shift 5 2>/dev/null || shift 4
R=/mnt/kai_ckp/yiding/MaxHE
REPO=/mnt/world_foundational_model/kai/MaxHE/ltlf
mkdir -p "$R/logs"
ENVV="TAG=$TAG MODE=$MODE PORT=$PORT SPLIT=$SPLIT"
if [ -n "${TASK_JSON:-}" ] && [ "$TASK_JSON" != "-" ]; then ENVV="$ENVV TASK_JSON=$TASK_JSON"; fi
for kv in "$@"; do ENVV="$ENVV $kv"; done
cd "$REPO" || exit 1
setsid nohup env $ENVV bash ops/run_ltlf_eval.sh > "$R/logs/$TAG.driver.log" 2>&1 < /dev/null &
sleep 1
echo "launched TAG=$TAG MODE=$MODE PORT=$PORT pid=$!"
echo "log: $R/logs/$TAG.driver.log"
