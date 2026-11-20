#!/usr/bin/env bash
# Re-exec a command in a private mount namespace where the model directory resolves to a RAM copy.
#
# Why this is needed
# ------------------
# The baseline wrapper hardcodes `base_model_path = "./openvla-7b"`
# (AeroVLA/src/model_wrapper/aerovla_wrapper_ui.py:22), so `--model_base` has **no effect**.
# `AeroVLA/openvla-7b` is a symlink into the parallel filesystem (/mnt/kai_ckp/...), and when other
# jobs saturate that filesystem a single model load can stall for many minutes (observed: >8 min,
# process sitting in cxiWaitEventWait).
#
# This helper bind-mounts a RAM copy (/dev/shm/...) **over the symlink's resolved target** inside a
# private mount namespace.  Nothing on disk is created, deleted or modified: only mounts are
# namespaced, so the working tree stays clean (verified with `git status`) and no other user of the
# machine is affected.
#
# Usage:
#   with_shm_model.sh <model-path-as-seen-by-the-code> <ram-copy-path> <command> [args...]
#   e.g. with_shm_model.sh "$C/openvla-7b" /dev/shm/openvla-7b python -u src/vlnce_src/eval_aerovla.py ...
#
# If <ram-copy-path> is missing, or the resolved target cannot be resolved, the command runs unchanged.
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <model-path> <ram-copy-path> <command> [args...]" >&2
  exit 2
fi

MODEL_PATH=$1
SOURCE=$2
shift 2

if [ ! -d "$SOURCE" ]; then
  exec "$@"
fi

RESOLVED=$(readlink -f "$MODEL_PATH" 2>/dev/null || true)
if [ -z "$RESOLVED" ] || [ ! -d "$RESOLVED" ]; then
  echo "[with_shm_model] cannot resolve $MODEL_PATH; running without the RAM model" >&2
  exec "$@"
fi

echo "[with_shm_model] bind-mounting $SOURCE over $RESOLVED (private mount namespace)" >&2
exec unshare -m bash -c '
  set -euo pipefail
  source=$1; resolved=$2; shift 2
  mount --bind "$source" "$resolved"
  exec "$@"
' _ "$SOURCE" "$RESOLVED" "$@"
