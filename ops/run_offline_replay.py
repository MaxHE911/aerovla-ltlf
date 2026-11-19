#!/usr/bin/env python3
"""L0 offline replay: assert every task is accepted by its own expert trajectory, and that the
mutated (missing / swapped / reversed / G-violating) traces are rejected.

CPU only, no simulator, no VLA.  Run from the repository root:

    python ops/run_offline_replay.py \
        --dataset-root dataset_raw --map ModernCityMap \
        --spawn-area-json AeroVLA/data/meta/map_spawnarea_info.json \
        --split-json AeroVLA/data/uav_dataset/seen_valset_splits/ModernCityMap.json \
        --tasks <tasks_ModernCityMap_seen_valset.json>   # optional: rebuild if omitted
        --out work/results/offline_replay_ModernCityMap.json

Exit code 0 only if every usable task replays accepted and every negative case is rejected.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ltlf_logic.offline import check_task_replay  # noqa: E402
from ltlf_logic.tasks import (  # noqa: E402
    build_task,
    build_tasks_for_split,
    build_description_index,
    load_episode,
    load_map_areas,
    split_instruction,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--spawn-area-json", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--map", default="ModernCityMap")
    ap.add_argument("--split-name", default="seen_valset")
    ap.add_argument("--tasks", default=None, help="existing tasks json (skip M0 rebuild)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-landmarks", type=int, default=3)
    args = ap.parse_args(argv)

    areas = load_map_areas(args.spawn_area_json, args.map)

    if args.tasks:
        with open(args.tasks) as f:
            tasks = json.load(f)
        if args.limit:
            tasks = tasks[: args.limit]
    else:
        tasks, _ = build_tasks_for_split(
            args.dataset_root, args.spawn_area_json, args.split_json, args.map,
            split_name=args.split_name, max_landmarks=args.max_landmarks, limit=args.limit,
        )

    results: List[dict] = []
    n_ok = n_bad = n_expert_fail = n_judge_disagree = 0
    for i, task in enumerate(tasks):
        if task.get("skipped"):
            results.append({"task_id": task.get("task_id"), "skipped": True, "reason": task.get("skip_reason")})
            continue
        episode = load_episode(args.dataset_root, task["map"], task["seq"], areas)
        rep = check_task_replay(task, episode.positions)
        results.append(rep)
        if rep["expert_accepted_online"] and rep["experts_agree"]:
            pass
        else:
            n_expert_fail += 1
            print("  [EXPERT-FAIL] %s  formula=%s  events=%s  residual=%s"
                  % (task["task_id"], task["formula"], rep["events_order"], rep.get("residual", "")))
        if not rep["negatives_all_ok"]:
            n_bad += 1
            bad = [n for n in rep["negatives"] if not n["ok"]]
            print("  [NEGATIVE-FAIL] %s  %s" % (task["task_id"], bad))
        if not rep["judges_agree_everywhere"]:
            n_judge_disagree += 1
            print("  [JUDGE-DISAGREE] %s" % task["task_id"])
        if rep["expert_accepted_online"] and rep["negatives_all_ok"] and rep["judges_agree_everywhere"]:
            n_ok += 1
        if (i + 1) % 20 == 0:
            print("  replayed %d/%d  ok=%d bad-negatives=%d expert-fail=%d judge-disagree=%d"
                  % (i + 1, len(tasks), n_ok, n_bad, n_expert_fail, n_judge_disagree))

    usable = [r for r in results if not r.get("skipped")]
    summary = {
        "map": args.map,
        "split": args.split_name,
        "tasks_total": len(tasks),
        "tasks_usable": len(usable),
        "tasks_skipped": len(tasks) - len(usable),
        "expert_accepted": sum(1 for r in usable if r["expert_accepted_online"]),
        "expert_accepted_independent_semantics": sum(1 for r in usable if r["expert_accepted_independent"]),
        "negative_cases_total": sum(len(r["negatives"]) for r in usable),
        "negative_cases_passed": sum(1 for r in usable for n in r["negatives"] if n["ok"]),
        "tasks_all_checks_ok": n_ok,
        "tasks_with_failing_negative": n_bad,
        "tasks_expert_failed": n_expert_fail,
        "judge_disagreements": n_judge_disagree,
        "stage_histogram": _hist([str(r["n_stages"]) for r in usable]),
        "st_rt_histogram": _hist([r["st_rt"] for r in usable]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    print(json.dumps(summary, indent=1))
    print("wrote", args.out)
    ok = (
        summary["tasks_expert_failed"] == 0
        and summary["tasks_with_failing_negative"] == 0
        and summary["judge_disagreements"] == 0
        and summary["tasks_all_checks_ok"] == len(usable)
    )
    print("L0 RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _hist(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


if __name__ == "__main__":
    raise SystemExit(main())
