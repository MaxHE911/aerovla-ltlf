#!/usr/bin/env python3
"""Run one evaluation arm with the LTLf task-logic layer (M3+M5).

The script keeps the baseline CLI untouched: everything after ``--`` style ``--ltlf-*`` options
is stripped from ``sys.argv`` before the baseline's own HfArgumentParser runs, then the baseline
modules are imported and driven exactly like ``src/vlnce_src/eval_aerovla.py``.

Example (from the repository root, inside an Xvfb + simulator session):

    python ops/run_eval_ltlf.py \
        --run_type eval --name ltlf_fixed_order --gpu_id 0 --simulator_tool_port 30000 \
        --DDP_MASTER_PORT 29600 --batchSize 1 --maxWaypoints 80 \
        --dataset_path dataset_raw/ --eval_save_path /data/eval_results/x \
        --model_path checkpoints/aero_vla --model_base ./openvla-7b \
        --eval_json_path /data/splits/x.json \
        --map_spawn_area_json_path data/meta/map_spawnarea_info.json \
        --object_name_json_path data/meta/object_description.json \
        --ltlf-mode fixed_order --ltlf-task-json /data/tasks.json \
        --ltlf-log-dir /data/ltlf_logs/x --ltlf-run-summary /data/ltlf_logs/x/run_summary.json \
        --ltlf-disable-distance-heuristic 1 --ltlf-gate 1 --ltlf-takeover 1
"""
from __future__ import annotations

import json
import os
import sys
import traceback


def _split_ltlf_argv(argv):
    """Separate our own ``--ltlf-*`` options from the baseline CLI (handles --k v and --k=v)."""
    ltlf = {}
    rest = [argv[0]]
    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--ltlf-"):
            if "=" in a:
                key, val = a.split("=", 1)
            else:
                key = a
                if i + 1 >= len(argv):
                    raise SystemExit("missing value for %s" % a)
                val = argv[i + 1]
                i += 1
            ltlf[key[len("--ltlf-"):].replace("-", "_")] = val
        else:
            rest.append(a)
        i += 1
    return ltlf, rest


def _as_bool(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    ltlf_args, rest = _split_ltlf_argv(sys.argv)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aero = os.path.join(repo, "AeroVLA")
    if not os.path.isdir(aero):
        raise SystemExit("AeroVLA baseline directory not found at %s" % aero)

    os.chdir(aero)  # the baseline resolves ./openvla-7b and relative dataset paths from here
    sys.path.insert(0, aero)
    sys.path.insert(0, repo)
    sys.argv = rest

    from src.common.param import args, data_args, model_args  # noqa: E402  (parses sys.argv)
    from src.model_wrapper.aerovla_wrapper_ui import AerialVLAWrapper  # noqa: E402
    from src.vlnce_src.assist import Assist  # noqa: E402
    from src.vlnce_src.closeloop_util import (  # noqa: E402
        CheckPort,
        initialize_env_eval,
        is_dist_avail_and_initialized,
        setup,
    )
    from ltlf_logic.eval_loop import run_eval_ltlf  # noqa: E402
    from ltlf_logic.runner import LTLfConfig  # noqa: E402

    import torch  # noqa: E402

    mode = ltlf_args.get("mode", "single_stage")
    overrides = {}
    if "task_json" in ltlf_args:
        overrides["task_json"] = ltlf_args["task_json"]
    if "gate" in ltlf_args:
        overrides["gate_enabled"] = _as_bool(ltlf_args["gate"])
    if "takeover" in ltlf_args:
        overrides["takeover_enabled"] = _as_bool(ltlf_args["takeover"])
    if "disable_distance_heuristic" in ltlf_args:
        overrides["disable_distance_increasing"] = _as_bool(ltlf_args["disable_distance_heuristic"])
    if "event_radius" in ltlf_args:
        overrides["event_radius_m"] = float(ltlf_args["event_radius"])
    if "takeover_fwd" in ltlf_args:
        overrides["takeover_fwd_m"] = float(ltlf_args["takeover_fwd"])
    config = LTLfConfig.for_mode(mode, **overrides)

    log_dir = ltlf_args.get("log_dir") or os.path.join(os.path.dirname(os.path.abspath(args.eval_save_path)), "ltlf_logs")
    summary_path = ltlf_args.get("run_summary") or os.path.join(log_dir, "run_summary.json")
    os.makedirs(log_dir, exist_ok=True)

    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path
    os.makedirs(eval_save_path, exist_ok=True)

    print("[ltlf] repo=%s aero=%s" % (repo, aero))
    print("[ltlf] mode=%s config=%s" % (mode, json.dumps(config.to_dict())))
    print("[ltlf] eval_json_path=%s eval_save_path=%s dataset_path=%s" % (eval_json_path, eval_save_path, dataset_path))
    print("[ltlf] cuda_visible=%s" % os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[ltlf] torch.cuda.is_available()=%s" % torch.cuda.is_available())

    setup()
    assert CheckPort(), "error port"
    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)
    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()
    args.DistributedDataParallel = False

    model_wrapper = AerialVLAWrapper(model_args=model_args, data_args=data_args)
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)

    rc = 0
    summary = {}
    try:
        summary = run_eval_ltlf(
            model_wrapper=model_wrapper,
            assist=assist,
            eval_env=eval_env,
            eval_save_dir=eval_save_path,
            config=config,
            log_dir=log_dir,
        )
        summary["baseline_args"] = {
            "maxWaypoints": int(args.maxWaypoints),
            "batchSize": int(args.batchSize),
            "gpu_id": int(args.gpu_id),
            "simulator_tool_port": int(args.simulator_tool_port),
            "dataset_path": str(dataset_path),
            "eval_json_path": str(eval_json_path),
            "model_path": str(model_args.model_path),
        }
    except Exception:
        traceback.print_exc()
        summary = {"error": traceback.format_exc(), "mode": mode, "config": config.to_dict()}
        rc = 1
    finally:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=1)
        print("[ltlf] run summary -> %s" % summary_path)
        try:
            eval_env.delete_VectorEnvUtil()
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
