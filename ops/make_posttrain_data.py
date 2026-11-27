"""Build the Stage-A training set for idea 3 (logic-conditioned SFT) from the existing SFT data.

The point of Stage A is to remove the distribution shift that idea 1/2 leave behind: at run time the
policy is given a *per-stage* instruction (one landmark description + the direction of that landmark),
but the model was trained only on single-goal instructions.  Here we regenerate the instruction text
for the **same expert frames and the same action labels**: no new data collection, no new annotation.

For every training frame ``f`` of a trajectory that has an M0 task we

1. find the stage the expert is currently flying towards: the first landmark whose expert arrival frame
   is ``> f`` (falling back to the final target),
2. compute the direction bucket with a *faithful copy* of the baseline's
   ``get_semantic_direction`` (``AeroVLA/src/model_wrapper/aerovla_wrapper_ui.py:46-79``, 7 buckets),
3. emit ``<image>\\nFly {dir_text}and find the target. {desc}`` with ``desc`` = that landmark's 40-60
   word scene description, i.e. exactly what ``build_stage_instruction`` / the runtime injection uses,
4. keep the original action label untouched.

To avoid eroding the natural-language capability (the failure mode Logic-VLA reports when the logic
signal is over-weighted), a fraction of samples keeps the **original** instruction.  The choice is
hash-based, so the mix is reproducible without a RNG seed.

Output schema is identical to ``AeroVLA/data/aerovla_train_dataset.json`` (plus two audit fields), so
``AeroVLA/src/aerovla_dataset.py`` reads it unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["bucket_direction", "build_stageA", "main"]

# --------------------------------------------------------------------------------------
# baseline-faithful direction bucket (measured copy, see module docstring)
# --------------------------------------------------------------------------------------
BUCKETS = (
    (-15.0, 15.0, "straight ahead "),
    (15.0, 60.0, "forward-right "),
    (60.0, 120.0, "to your right "),
    (120.0, 180.0, "to your right rear "),
    (-60.0, -15.0, "forward-left "),
    (-120.0, -60.0, "to your left "),
    (-180.0, -120.0, "to your left rear "),
)


def _rotate_inverse(q: Sequence[float], v: Sequence[float]) -> Tuple[float, float, float]:
    """Rotate ``v`` by ``q^-1`` (q = x, y, z, w), i.e. world -> body."""
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    # inverse rotation quaternion is (-x, -y, -z, w)
    ax, ay, az, aw = -x, -y, -z, w
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    # t = 2 * (a_vec x v); v' = v + aw * t + a_vec x t
    tx = 2.0 * (ay * vz - az * vy)
    ty = 2.0 * (az * vx - ax * vz)
    tz = 2.0 * (ax * vy - ay * vx)
    rx = vx + aw * tx + (ay * tz - az * ty)
    ry = vy + aw * ty + (az * tx - ax * tz)
    rz = vz + aw * tz + (ax * ty - ay * tx)
    return rx, ry, rz


def bucket_direction(position: Sequence[float], quaternion: Sequence[float], target: Sequence[float]) -> str:
    """Copy of the baseline's 7-bucket semantic direction (returns '' when nearly on target)."""
    vx = float(target[0]) - float(position[0])
    vy = float(target[1]) - float(position[1])
    vz = float(target[2]) - float(position[2])
    if math.hypot(vx, vy) < 0.01:
        return ""
    bx, by, _ = _rotate_inverse(quaternion, (vx, vy, vz))
    angle = math.degrees(math.atan2(by, bx))
    for lo, hi, text in BUCKETS:
        if lo <= angle <= hi:
            return text
    return ""


# --------------------------------------------------------------------------------------
# stage assignment
# --------------------------------------------------------------------------------------
def _stage_table(task: dict) -> Tuple[List[Tuple[int, str, List[float], str]], int]:
    """``[(arrival_frame, ap, center, description), ...]`` in expert order + final target frame."""
    lms = list(task.get("landmarks") or [])
    by_area: Dict[int, int] = {}
    for lm in task.get("all_landmarks_along_route") or []:
        f = lm.get("frame")
        if f is not None:
            by_area[int(lm.get("area_index", -1))] = int(f)
    table: List[Tuple[int, str, List[float], str]] = []
    for lm in lms:
        fr = by_area.get(int(lm.get("area_index", -1)), 10 ** 9)
        table.append((fr, str(lm["ap"]), [float(c) for c in lm["center"]], str(lm.get("description") or "")))
    table.sort(key=lambda r: r[0])
    nf = int(task.get("expert_frames") or 0)
    final = task.get("final_target") or {}
    table.append((nf - 1 if nf > 0 else 10 ** 9, "g", [float(c) for c in final["center"]], str(final.get("description") or "")))
    return table, nf


def _target_for_frame(table, f: int):
    for fr, ap, center, desc in table:
        if f <= fr:
            return ap, center, desc
    fr, ap, center, desc = table[-1]
    return ap, center, desc


# --------------------------------------------------------------------------------------
# main builder
# --------------------------------------------------------------------------------------
def build_stageA(
    train_json: str,
    tasks_json: str,
    dataset_root: str,
    out_path: str,
    *,
    mix_keep_original: float = 0.5,
    limit: Optional[int] = None,
    maps: Optional[Sequence[str]] = None,
) -> dict:
    with open(tasks_json) as f:
        tasks = json.load(f)
    by_key = {}
    for t in tasks:
        if t.get("skipped"):
            continue
        by_key[(str(t["map"]), str(t["seq"]))] = t

    with open(train_json) as f:
        samples = json.load(f)
    if limit:
        samples = samples[:limit]

    # group by trajectory so each merged_data.json is read once
    per_traj: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        rel = str(s.get("traj_rel_dir") or "").strip("/")
        per_traj[rel].append(s)

    traj_cache: Dict[str, dict] = {}
    out: List[dict] = []
    stats = Counter()
    stage_counter = Counter()
    skipped_traj = 0

    for rel, group in per_traj.items():
        parts = rel.split("/")
        if len(parts) < 2:
            skipped_traj += 1
            continue
        map_name, seq = parts[0], parts[1]
        if maps and map_name not in maps:
            continue
        task = by_key.get((map_name, seq))
        if task is None:
            stats["traj_without_task"] += 1
            skipped_traj += 1
            continue
        traj_dir = os.path.join(dataset_root, map_name, seq)
        merged = traj_cache.get(traj_dir)
        if merged is None:
            path = os.path.join(traj_dir, "merged_data.json")
            if not os.path.exists(path):
                stats["traj_missing_data"] += 1
                skipped_traj += 1
                continue
            with open(path) as f:
                merged = json.load(f)
            traj_cache[traj_dir] = merged
        frames = merged.get("trajectory_raw_detailed") or []
        if not frames:
            stats["traj_no_frames"] += 1
            skipped_traj += 1
            continue
        table, nf = _stage_table(task)
        for s in group:
            img = str(s.get("img_name") or "")
            try:
                idx = int("".join(ch for ch in img if ch.isdigit()))
            except ValueError:
                stats["bad_img_name"] += 1
                continue
            if idx >= len(frames):
                stats["frame_out_of_range"] += 1
                continue
            fr = frames[idx]
            pos = [float(v) for v in fr["position"][:3]]
            quat = [float(v) for v in (fr.get("orientation") or [0, 0, 0, 1])[:4]]
            ap, center, desc = _target_for_frame(table, idx)
            stage_counter[ap] += 1
            dir_text = bucket_direction(pos, quat, center)
            staged = "<image>\nFly %sand find the target. %s" % (dir_text, desc)
            # deterministic mix so the split is reproducible without a seed
            h = hashlib.sha1(("%s|%s" % (rel, img)).encode()).digest()
            keep_original = (h[0] / 255.0) < mix_keep_original
            rec = {
                "traj_rel_dir": rel,
                "img_name": img,
                "instruction": s["instruction"] if keep_original else staged,
                "label": s["label"],
                "is_last_step": bool(s.get("is_last_step")),
                "is_penultimate": bool(s.get("is_penultimate")),
                "ltlf_stage": None if keep_original else ap,
                "ltlf_source": "original" if keep_original else "staged",
            }
            out.append(rec)
            stats["staged" if not keep_original else "original"] += 1

        if len(out) and len(out) % 5000 == 0:
            print("[make_posttrain_data] %d samples ..." % len(out), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    summary = {
        "out": out_path,
        "train_json": train_json,
        "tasks_json": tasks_json,
        "n_samples": len(out),
        "n_staged": stats["staged"],
        "n_original": stats["original"],
        "mix_keep_original": mix_keep_original,
        "trajectories_used": len(traj_cache),
        "trajectories_skipped": skipped_traj,
        "stage_histogram": dict(stage_counter),
        "skips": {k: v for k, v in stats.items() if k.startswith(("traj_", "bad_", "frame_"))},
    }
    with open(out_path + ".summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="build the logic-conditioned SFT set of idea 3")
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix-keep-original", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--map", action="append", default=None)
    args = ap.parse_args(argv)
    s = build_stageA(
        args.train_json,
        args.tasks,
        args.dataset_root,
        args.out,
        mix_keep_original=args.mix_keep_original,
        limit=args.limit,
        maps=args.map,
    )
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
