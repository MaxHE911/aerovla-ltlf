#!/usr/bin/env python3
"""Build a de-duplicated, optionally filtered evaluation split for one experiment arm.

The shipped split files list the same trajectory once per start frame (2873 entries but only
124 unique trajectories for ModernCityMap), and ``load_my_datasets`` ignores the ``frame``
field, so every duplicate record would be scored as an independent episode.  All arms of this
project therefore use one entry per trajectory, built here.

Usage (repo root):
    python ops/make_split.py --split AeroVLA/data/uav_dataset/seen_valset_splits/ModernCityMap.json \
        --tasks work/tasks/tasks_ModernCityMap_seen_valset.json \
        --select first:12 --out work/splits/stage3_12.json

--select grammar:
    all                 every unique trajectory
    first:N             first N unique trajectories in split order
    st:N / rt:N         filtered by the task's ST/RT label
    stages2:N stages3:N stages4:N   filtered by number of stages
    seq:<seq>,<seq>     explicit trajectory ids
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--tasks", default=None, help="M0 tasks json (needed for st/rt/stages filters)")
    ap.add_argument("--select", default="all")
    ap.add_argument("--map", default=None, help="keep only entries of this map")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with open(args.split) as f:
        entries = json.load(f)

    # first occurrence per trajectory, preserving split order
    order: List[str] = []
    first_entry: Dict[str, dict] = {}
    for e in entries:
        parts = e["json"].strip("/").split("/")
        seq = parts[-2]
        if args.map and parts[-3] != args.map:
            continue
        if seq not in first_entry:
            first_entry[seq] = e
            order.append(seq)

    task_by_seq: Dict[str, dict] = {}
    if args.tasks and os.path.exists(args.tasks):
        with open(args.tasks) as f:
            for t in json.load(f):
                task_by_seq[t["seq"]] = t

    sel = args.select.strip()
    if sel == "all":
        chosen = list(order)
    elif sel.startswith("seq:"):
        wanted = [s.strip() for s in sel[len("seq:"):].split(",") if s.strip()]
        missing = [s for s in wanted if s not in first_entry]
        if missing:
            raise SystemExit("unknown seq(s): %s" % missing)
        chosen = wanted
    elif sel.startswith("first:"):
        chosen = order[: int(sel.split(":", 1)[1])]
    elif sel.split(":")[0] in ("st", "rt"):
        label, n = sel.split(":")
        want = "ST" if label == "st" else "RT"
        chosen = [s for s in order if task_by_seq.get(s, {}).get("st_rt") == want][: int(n)]
    elif sel.startswith("stages"):
        n_stages, n = sel.split(":")
        want_stages = int(n_stages.replace("stages", ""))
        chosen = [s for s in order if task_by_seq.get(s, {}).get("n_stages") == want_stages][: int(n)]
    else:
        raise SystemExit("unknown --select %r" % sel)

    if not chosen:
        raise SystemExit("selection %r produced 0 episodes" % sel)
    out = [first_entry[s] for s in chosen]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("select=%s -> %d episodes (%s)" % (sel, len(out), ", ".join(chosen[:6]) + (" ..." if len(chosen) > 6 else "")))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
