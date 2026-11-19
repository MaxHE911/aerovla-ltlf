#!/usr/bin/env python3
"""Independent offline metrics for one or more evaluation arms.

The judge never trusts the logic layer's own state: it re-reads the *saved episode logs*
(``log/NNNNNN.json`` positions), rebuilds the event trace, and recomputes

    SR / OSR / NE / SPL        - baseline definitions (utils/metric.py semantics)
    ESR                        - accept(final residual) with the direct finite-trace semantics
    SR & ESR                   - the main metric of idea1 §5.5
    stage completion / ND      - from the task spec + rebuilt event sequence
    failure types              - from the arm's eval log terminal messages, if present

Usage (repo root):
    python ops/analyze_results.py \
        --arm g0-baseline=/data/eval_results/_ltlf/s3-g0-baseline \
        --arm g1-ours=/data/eval_results/_ltlf/s3-g1-singlestage \
        --tasks work/tasks/tasks_ModernCityMap_seen_valset.json \
        --ltlf-logs g1-ours=/data/ltlf_logs/s3-g1-singlestage \
        --out work/results/stage3_metrics.json

Notes
-----
* episode directories are named ``[success_|oracle_]<seq>`` exactly like the baseline writes them;
* expert reference (NE / SPL) is read from each episode's ``ori_info.json`` -> merged_data.json;
* the script is read-only with respect to the experiment outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AERO_DIR = os.path.join(REPO_DIR, "AeroVLA")

from ltlf_logic.formula import accept, normalize, parse, prog, to_str  # noqa: E402
from ltlf_logic.offline import events_from_positions, independent_esr  # noqa: E402
from ltlf_logic.tasks import split_instruction  # noqa: E402

TERMINAL_PATTERNS = [
    ("success", "Success!"),
    ("oracle", "Oracle Success!"),
    ("collision", "Collision (Crashed)"),
    ("early_end", "Early End"),
    ("timeout", "Timeout (Max Steps Reached)"),
]


# --------------------------------------------------------------------------------------
# episode level
# --------------------------------------------------------------------------------------
def _sort_key(name: str) -> int:
    m = re.search(r"\d+", name)
    return int(m.group()) if m else -1


def read_episode_positions(ep_dir: str) -> List[Tuple[float, float, float]]:
    log_dir = os.path.join(ep_dir, "log")
    files = sorted(os.listdir(log_dir), key=_sort_key)
    out: List[Tuple[float, float, float]] = []
    for name in files:
        with open(os.path.join(log_dir, name)) as f:
            data = json.load(f)
        p = data["sensors"]["state"]["position"]
        out.append((float(p[0]), float(p[1]), float(p[2])))
    return out


def read_expert_reference(ep_dir: str) -> Optional[dict]:
    ori_path = os.path.join(ep_dir, "ori_info.json")
    if not os.path.exists(ori_path):
        return None
    with open(ori_path) as f:
        ori = json.load(f)
    # ori_traj_dir is relative to the AeroVLA working directory used during evaluation
    candidates = [
        ori["ori_traj_dir"],
        os.path.join(AERO_DIR, ori["ori_traj_dir"]),
        os.path.join(REPO_DIR, ori["ori_traj_dir"]),
    ]
    merged = None
    for cand in candidates:
        p = os.path.join(cand, "merged_data.json")
        if os.path.exists(p):
            merged = p
            break
    if merged is None:
        return None
    with open(merged) as f:
        data = json.load(f)
    frames = data["trajectory_raw_detailed"]
    pts = [(float(fr["position"][0]), float(fr["position"][1]), float(fr["position"][2])) for fr in frames]
    length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    return {"last": pts[-1], "length": length, "instruction": data["conversations"][0]["value"]}


def path_length(positions: Sequence[Sequence[float]]) -> float:
    return sum(math.dist(positions[i], positions[i + 1]) for i in range(len(positions) - 1))


def parse_terminal_events(eval_log: str) -> Dict[str, int]:
    out = {k: 0 for k, _ in TERMINAL_PATTERNS}
    if not os.path.exists(eval_log):
        return out
    with open(eval_log, "r", errors="ignore") as f:
        for line in f:
            for key, needle in TERMINAL_PATTERNS:
                if needle in line:
                    out[key] += 1
    return out


def task_for(seq: str, tasks_by_seq: Dict[str, dict], single_goal: bool) -> dict:
    """Task record for judging; falls back to the minimal single-goal spec."""
    if seq in tasks_by_seq and not single_goal:
        return tasks_by_seq[seq]
    t = tasks_by_seq.get(seq)
    if t is None:
        return {
            "task_id": seq,
            "formula": "F(g)",
            "formula_single_goal": "F(g)",
            "event_order": ["g"],
            "landmarks": [],
            "n_stages": 1,
            "final_target": {"ap": "g", "center": None, "description": ""},
        }
    if single_goal:
        return dict(
            t,
            formula=t["formula_single_goal"],
            landmarks=[],
            event_order=["g"],
            n_stages=1,
        )
    return t


def judge_arm(arm_dir: str, tasks_by_seq: Dict[str, dict], *, single_goal: bool, ltlf_log_dir: Optional[str] = None) -> dict:
    episodes: List[dict] = []
    for name in sorted(os.listdir(arm_dir)):
        ep_dir = os.path.join(arm_dir, name)
        if not os.path.isdir(ep_dir):
            continue
        if not os.path.isdir(os.path.join(ep_dir, "log")):
            continue
        success = name.startswith("success_")
        oracle = success or name.startswith("oracle_")
        seq = name.replace("success_", "").replace("oracle_", "")
        positions = read_episode_positions(ep_dir)
        expert = read_expert_reference(ep_dir)
        rec = {
            "seq": seq,
            "dir": name,
            "success": success,
            "oracle_success": oracle,
            "steps_logged": len(positions),
            "final_position": list(positions[-1]) if positions else None,
        }
        if expert:
            rec["ne"] = math.dist(expert["last"], positions[-1])
            pred_len = path_length(positions)
            ref = max(expert["length"] - 20.0, 1e-6)
            rec["pred_length"] = pred_len
            rec["expert_length"] = expert["length"]
            rec["spl"] = (ref / max(ref, pred_len)) if success else 0.0
            rec["spl"] = max(min(rec["spl"], 1.0), 0.0)
        # ESR from the *saved log*, judged independently of the online logic layer
        task = task_for(seq, tasks_by_seq, single_goal)
        if task.get("final_target", {}).get("center") is None and expert:
            # minimal spec: target = the episode's own target position is unknown here; use the
            # dataset mark.json of the episode instead
            pass
        try:
            esr = independent_esr(task, positions) if task.get("final_target", {}).get("center") else None
        except Exception as exc:  # pragma: no cover
            esr = {"accepted": None, "error": str(exc)}
        rec["esr_accepted"] = None if esr is None else esr["accepted"]
        rec["events_order"] = [] if esr is None else esr["events_order"]
        rec["multi_event_points"] = 0 if esr is None else esr.get("multi_event_points", 0)
        rec["n_stages"] = task.get("n_stages", 1)
        if esr is not None and rec["esr_accepted"] is not None:
            # per-stage completion: which spec atoms were entered, in order
            want = list(task.get("event_order", []))
            got = rec["events_order"]
            completed = 0
            remaining = list(want)
            for ap in got:
                if remaining and ap == remaining[0]:
                    completed += 1
                    remaining.pop(0)
            rec["stages_completed"] = completed
            rec["stage_completion_rate"] = completed / max(1, len(want))
        # cross-check against the online logic layer's own residual (if the arm logged it)
        if ltlf_log_dir:
            jl = os.path.join(ltlf_log_dir, seq + ".jsonl")
            if os.path.exists(jl):
                rows = [json.loads(line) for line in open(jl) if line.strip()]
                if rows:
                    rec["online_final_residual"] = rows[-1]["residual"]
                    rec["online_accepted"] = rows[-1]["accepted"]
                    rec["online_suppressed_stops"] = sum(
                        1 for r in rows if r.get("model_stop") and r.get("gated_stop") is False
                    )
                    rec["online_takeovers"] = sum(1 for r in rows if r.get("takeover"))
                    rec["online_stage_switches"] = sum(1 for r in rows if r.get("stage_switch"))
                    rec["online_agrees_with_offline"] = (
                        None if rec["esr_accepted"] is None else bool(rec["online_accepted"]) == bool(rec["esr_accepted"])
                    )
        episodes.append(rec)

    n = len(episodes)
    esr_known = [e for e in episodes if e.get("esr_accepted") is not None]
    agg = {
        "arm": os.path.basename(os.path.normpath(arm_dir)),
        "episodes": n,
        "sr": _mean([1.0 if e["success"] else 0.0 for e in episodes]),
        "osr": _mean([1.0 if e["oracle_success"] else 0.0 for e in episodes]),
        "ne": _mean([e["ne"] for e in episodes if "ne" in e]),
        "spl": _mean([e["spl"] for e in episodes if "spl" in e]),
        "esr": _mean([1.0 if e["esr_accepted"] else 0.0 for e in esr_known]) if esr_known else None,
        "sr_and_esr": _mean([1.0 if (e["success"] and e["esr_accepted"]) else 0.0 for e in esr_known]) if esr_known else None,
        "episodes_with_esr": len(esr_known),
        "stage_completion_rate": _mean([e["stage_completion_rate"] for e in episodes if "stage_completion_rate" in e]),
        "mean_nd": _mean([e["n_stages"] for e in episodes]) if episodes else None,
        "suppressed_stops": sum(e.get("online_suppressed_stops", 0) for e in episodes) or None,
        "takeovers": sum(e.get("online_takeovers", 0) for e in episodes) or None,
        "stage_switches": sum(e.get("online_stage_switches", 0) for e in episodes) or None,
        "online_offline_agreement": _mean(
            [1.0 if e["online_agrees_with_offline"] else 0.0 for e in episodes if e.get("online_agrees_with_offline") is not None]
        ),
        "multi_event_points": sum(e.get("multi_event_points", 0) for e in episodes),
        "episodes_detail": episodes,
    }
    return agg


def _mean(items: Sequence[float]) -> Optional[float]:
    items = list(items)
    if not items:
        return None
    return sum(items) / len(items)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="label=result_dir")
    ap.add_argument("--ltlf-logs", action="append", default=[], help="label=ltlf_log_dir")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--single-goal-arms", default="", help="comma separated labels judged as F(g)")
    ap.add_argument("--eval-logs", action="append", default=[], help="label=eval.log path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    tasks_by_seq: Dict[str, dict] = {}
    if args.tasks and os.path.exists(args.tasks):
        with open(args.tasks) as f:
            for t in json.load(f):
                tasks_by_seq[t["seq"]] = t

    log_map = dict(x.split("=", 1) for x in args.ltlf_logs)
    eval_map = dict(x.split("=", 1) for x in args.eval_logs)
    single_goal = set(x.strip() for x in args.single_goal_arms.split(",") if x.strip())

    report = {"arms": {}, "tasks_json": args.tasks}
    for spec in args.arm:
        label, arm_dir = spec.split("=", 1)
        agg = judge_arm(
            arm_dir,
            tasks_by_seq,
            single_goal=(label in single_goal or not tasks_by_seq),
            ltlf_log_dir=log_map.get(label),
        )
        agg["label"] = label
        agg["dir"] = arm_dir
        if label in eval_map:
            agg["failure_types"] = parse_terminal_events(eval_map[label])
        report["arms"][label] = agg
        print("== %s ==" % label)
        print(json.dumps({k: v for k, v in agg.items() if k != "episodes_detail"}, indent=1))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)

    # compact comparison table
    keys = ["episodes", "sr", "osr", "ne", "spl", "esr", "sr_and_esr", "stage_completion_rate"]
    print("\n| arm | " + " | ".join(keys) + " |")
    print("|" + "---|" * (len(keys) + 1))
    for label, agg in report["arms"].items():
        cells = []
        for k in keys:
            v = agg.get(k)
            cells.append("n/a" if v is None else ("%.3f" % v if isinstance(v, float) else str(v)))
        print("| %s | %s |" % (label, " | ".join(cells)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
