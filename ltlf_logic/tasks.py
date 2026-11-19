"""M0: offline task/spec construction from episode data (no VLA, no simulator).

For every evaluation episode we build

* the original single-goal spec ``F(g)`` (no trajectory needed), and
* a long-horizon spec ``F(L1 & F(L2 & ... & F(g)))`` derived from the landmark order that the
  *expert* trajectory passes through, plus a human readable long instruction and one real
  40-60 word scene description per landmark.

Coordinate convention (frozen, see ``work/idea1.md`` §2/M0): a landmark's position is
``area[9:12]`` of ``map_spawnarea_info.json`` - exactly the coordinate the baseline uses as the
final target (``env_uav.py:139``), *not* the ``area[0:3]+1`` variant used only to pick the
closest area.  Event radius is 12 m, measured in 3-D.

All outputs go to a project-owned derived-data directory; ``merged_data.json``, the shared
splits and the metadata files are only read.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .formula import AP, F, Formula, mk_and, mk_f, normalize, to_str
from .monitor import DEFAULT_EVENT_RADIUS_M, Landmark

__all__ = [
    "MapArea",
    "load_map_areas",
    "find_closest_area",
    "load_dataset_root",
    "load_episode",
    "detect_landmarks",
    "build_description_index",
    "build_task",
    "build_tasks_for_split",
    "GENERATOR_VERSION",
]

GENERATOR_VERSION = "v1.0.0"

# pairwise separation below 2*radius would allow one sampling point to touch two landmarks
MIN_PAIRWISE_SEPARATION_M = 2 * DEFAULT_EVENT_RADIUS_M
# a landmark counts as "off the straight line" when its centre is farther than this
OFF_LINE_THRESHOLD_M = DEFAULT_EVENT_RADIUS_M
# target description length in the training distribution (median 43 words, idea1 §1.4)
TARGET_DESC_WORDS = 43


# --------------------------------------------------------------------------------------
# map metadata
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MapArea:
    index: int
    center: Tuple[float, float, float]
    asset: str
    raw: Tuple = field(default=(), compare=False)

    def to_dict(self) -> dict:
        return {"index": self.index, "center": list(self.center), "asset": self.asset}


def load_map_areas(spawn_area_json: str, map_name: str) -> List[MapArea]:
    """Parse ``map_spawnarea_info.json`` into :class:`MapArea` records (area[9:12] centres)."""
    with open(spawn_area_json, "r") as f:
        raw = json.load(f)
    if map_name not in raw:
        raise KeyError("map %r not in %s (available: %s)" % (map_name, spawn_area_json, sorted(raw)[:5]))
    areas: List[MapArea] = []
    for i, a in enumerate(raw[map_name]):
        if len(a) < 18:
            continue
        areas.append(MapArea(index=i, center=(float(a[9]), float(a[10]), float(a[11])), asset=str(a[16]), raw=tuple(a)))
    return areas


def _euclid(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def find_closest_area(coord: Sequence[float], areas: Sequence[MapArea]) -> MapArea:
    """Faithful copy of ``env_uav.find_closest_area`` (distance on ``area[0:3]+ (1,1,0.5)``)."""
    best: Optional[MapArea] = None
    best_d = float("inf")
    for area in areas:
        raw = area.raw
        true_area = (raw[0] + 1, raw[1] + 1, raw[2] + 0.5)
        d = _euclid(coord, true_area)
        if d < best_d:
            best_d = d
            best = area
    if best is None:
        raise ValueError("no area found for %r" % (coord,))
    return best


# --------------------------------------------------------------------------------------
# episode data
# --------------------------------------------------------------------------------------
@dataclass
class Episode:
    map_name: str
    seq: str
    instruction: str
    obj_desc: str
    start_position: Tuple[float, float, float]
    target_position_raw: Tuple[float, float, float]
    target_area: MapArea
    positions: List[Tuple[float, float, float]]
    traj_dir: str

    @property
    def start_target_description(self) -> str:
        return self.obj_desc


def load_dataset_root(dataset_root: str, map_name: str) -> str:
    """Path of the per-map trajectory directory (``<dataset_root>/<map>``)."""
    path = os.path.join(dataset_root, map_name)
    if not os.path.isdir(path):
        raise FileNotFoundError("no trajectory directory: %s" % path)
    return path


def split_instruction(instruction: str) -> str:
    """Extract ``obj_desc`` exactly like ``aerovla_wrapper_ui.py:93``."""
    try:
        return instruction.split("degrees from you.")[1].split(" Please control")[0].strip()
    except IndexError:
        raise ValueError("instruction does not contain the two prompt anchors: %r" % instruction[:120])


def load_episode(dataset_root: str, map_name: str, seq: str, areas: Sequence[MapArea]) -> Episode:
    traj_dir = os.path.join(load_dataset_root(dataset_root, map_name), seq)
    with open(os.path.join(traj_dir, "merged_data.json"), "r") as f:
        merged = json.load(f)
    with open(os.path.join(traj_dir, "mark.json"), "r") as f:
        mark = json.load(f)
    instruction = merged["conversations"][0]["value"]
    frames = merged["trajectory_raw_detailed"]
    positions = [tuple(float(x) for x in fr["position"][:3]) for fr in frames]
    target_raw = tuple(float(x) for x in mark["target"]["position"][:3])
    target_area = find_closest_area(target_raw, areas)
    return Episode(
        map_name=map_name,
        seq=seq,
        instruction=instruction,
        obj_desc=split_instruction(instruction),
        start_position=positions[0],
        target_position_raw=target_raw,
        target_area=target_area,
        positions=positions,
        traj_dir=traj_dir,
    )


# --------------------------------------------------------------------------------------
# landmark extraction along the expert trajectory
# --------------------------------------------------------------------------------------
@dataclass
class LandmarkHit:
    area: MapArea
    frame: int
    position: Tuple[float, float, float]

    def to_dict(self) -> dict:
        return {"area_index": self.area.index, "asset": self.area.asset, "frame": self.frame,
                "center": list(self.area.center)}


def detect_landmarks(
    positions: Sequence[Sequence[float]],
    areas: Sequence[MapArea],
    *,
    radius: float = DEFAULT_EVENT_RADIUS_M,
    exclude_area_indices: Iterable[int] = (),
) -> List[LandmarkHit]:
    """Ordered first-entry list of landmark areas along a pose sequence.

    A landmark triggers when the pose is within ``radius`` of its centre; repeated frames inside
    the same area trigger only once (first entry).  Areas in ``exclude_area_indices`` (the final
    target) are ignored.
    """
    excluded = set(exclude_area_indices)
    inside: set = set()
    hits: List[LandmarkHit] = []
    for t, p in enumerate(positions):
        now = {a.index for a in areas if a.index not in excluded and _euclid(p, a.center) < radius}
        for idx in sorted(now - inside):
            area = areas[idx]
            hits.append(LandmarkHit(area=area, frame=t, position=tuple(float(x) for x in p[:3])))
        inside = now
    return hits


def _point_segment_distance(p: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    """Distance from point ``p`` to segment ``ab`` (3-D)."""
    ax, ay, az = (float(a[0]), float(a[1]), float(a[2]))
    bx, by, bz = (float(b[0]), float(b[1]), float(b[2]))
    px, py, pz = (float(p[0]), float(p[1]), float(p[2]))
    vx, vy, vz = bx - ax, by - ay, bz - az
    wx, wy, wz = px - ax, py - ay, pz - az
    vv = vx * vx + vy * vy + vz * vz
    if vv <= 1e-12:
        return math.sqrt(wx * wx + wy * wy + wz * wz)
    t = max(0.0, min(1.0, (wx * vx + wy * vy + wz * vz) / vv))
    cx, cy, cz = ax + t * vx, ay + t * vy, az + t * vz
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)


def route_length(positions: Sequence[Sequence[float]]) -> float:
    return sum(_euclid(positions[i], positions[i + 1]) for i in range(len(positions) - 1))


# --------------------------------------------------------------------------------------
# per-landmark real descriptions (taken verbatim from the dataset, never invented)
# --------------------------------------------------------------------------------------
def build_description_index(
    dataset_root: str, map_name: str, areas: Sequence[MapArea], limit: Optional[int] = None
) -> Dict[int, List[dict]]:
    """``area_index -> [{seq, description, words}]`` from trajectories whose *final target* is it.

    The dataset's ``merged_data.json`` prompt carries the real 40-60 word description of the
    target object; a landmark that was never a target of any trajectory has no description and is
    reported as such by the caller.
    """
    root = load_dataset_root(dataset_root, map_name)
    index: Dict[int, List[dict]] = {}
    seqs = sorted(os.listdir(root))
    if limit:
        seqs = seqs[:limit]
    for seq in seqs:
        traj_dir = os.path.join(root, seq)
        mark_path = os.path.join(traj_dir, "mark.json")
        merged_path = os.path.join(traj_dir, "merged_data.json")
        if not (os.path.isfile(mark_path) and os.path.isfile(merged_path)):
            continue
        try:
            with open(mark_path, "r") as f:
                mark = json.load(f)
            with open(merged_path, "r") as f:
                merged = json.load(f)
            area = find_closest_area(mark["target"]["position"], areas)
            desc = split_instruction(merged["conversations"][0]["value"])
        except Exception:
            continue
        index.setdefault(area.index, []).append(
            {"seq": seq, "description": desc, "words": len(desc.split())}
        )
    return index


def pick_description(candidates: Sequence[dict], target_words: int = TARGET_DESC_WORDS) -> dict:
    """Deterministically pick the description closest to the training-set median length."""
    return sorted(candidates, key=lambda c: (abs(c["words"] - target_words), c["seq"]))[0]


# --------------------------------------------------------------------------------------
# long instruction templates (fixed templates, no LLM)
# --------------------------------------------------------------------------------------
def long_instruction(
    landmark_assets: Sequence[str], final_asset: str, *, lang: str = "zh", unordered: bool = False
) -> str:
    """Render the long instruction with the frozen template family (idea1 §5.3).

    ``lang='zh'`` is the template exactly as designed (used for the paper's task statement);
    ``lang='en'`` is its literal translation, needed because the only slot the baseline has for
    text is the English ``obj_desc`` field of the prompt (baseline+long-instruction arm).
    """
    assets = [a for a in landmark_assets]
    if lang == "zh":
        if not assets:
            return "最后到达%s" % final_asset
        if unordered:
            names = "和".join(assets)
            return "经过%s，先后顺序不限，最后到达%s" % (names, final_asset)
        if len(assets) == 1:
            return "先经过%s，最后到达%s" % (assets[0], final_asset)
        return "先经过%s，最后到达%s" % ("，再经过".join(assets), final_asset)
    if not assets:
        return "Finally reach the %s." % final_asset
    if unordered:
        return "Visit the %s in any order, then finally reach the %s." % (" and the ".join(assets), final_asset)
    if len(assets) == 1:
        return "First pass the %s, then finally reach the %s." % (assets[0], final_asset)
    return "First pass the %s, then finally reach the %s." % (", then the ".join(assets), final_asset)


# --------------------------------------------------------------------------------------
# task builder
# --------------------------------------------------------------------------------------
def _formula_for(ap_names: Sequence[str]) -> Formula:
    """``F(L0 & F(L1 & ... & F(g)))`` built inside-out."""
    if not ap_names:
        raise ValueError("need at least the final goal atom")
    node: Formula = mk_f(AP(ap_names[-1]))
    for name in reversed(ap_names[:-1]):
        node = mk_f(mk_and([AP(name), node]))
    return normalize(node)


def build_task(
    episode: Episode,
    areas: Sequence[MapArea],
    description_index: Dict[int, List[dict]],
    *,
    map_name: str,
    split_name: str,
    radius: float = DEFAULT_EVENT_RADIUS_M,
    max_landmarks: int = 3,
    off_line_threshold: float = OFF_LINE_THRESHOLD_M,
    min_pairwise: float = MIN_PAIRWISE_SEPARATION_M,
    skip_reasons: Optional[List[str]] = None,
) -> dict:
    """Build one task record (single-goal + long-horizon spec) for one episode."""
    skip_reasons = skip_reasons if skip_reasons is not None else []
    target_area = episode.target_area
    hits = detect_landmarks(
        episode.positions, areas, radius=radius, exclude_area_indices=[target_area.index]
    )

    # first-entry order, one hit per area (detect_landmarks already guarantees that)
    start = episode.start_position
    end = target_area.center
    enriched: List[dict] = []
    for hit in hits:
        off_line = _point_segment_distance(hit.area.center, start, end)
        cands = description_index.get(hit.area.index, [])
        chosen = pick_description(cands) if cands else None
        enriched.append(
            {
                "area_index": hit.area.index,
                "asset": hit.area.asset,
                "center": [float(x) for x in hit.area.center],
                "frame": hit.frame,
                "off_line_distance_m": round(off_line, 3),
                "on_line": off_line <= off_line_threshold,
                "description_candidates": len(cands),
                "description": chosen["description"] if chosen else None,
                "description_source_seq": chosen["seq"] if chosen else None,
            }
        )

    # ordered picks, enforcing: has a real description, pairwise separation, off-line preference
    picked: List[dict] = []
    dropped: List[dict] = []
    for item in enriched:
        if item["description"] is None:
            dropped.append({"area_index": item["area_index"], "reason": "no_real_description"})
            continue
        if len(picked) >= max_landmarks:
            dropped.append({"area_index": item["area_index"], "reason": "beyond_max_landmarks"})
            continue
        if any(_euclid(item["center"], p["center"]) <= min_pairwise for p in picked):
            dropped.append({"area_index": item["area_index"], "reason": "too_close_to_previous_landmark"})
            continue
        if _euclid(item["center"], end) <= min_pairwise:
            dropped.append({"area_index": item["area_index"], "reason": "too_close_to_final_target"})
            continue
        picked.append(item)

    n_off_line = sum(1 for p in picked if not p["on_line"])
    st_rt = "RT" if n_off_line > 0 else "ST"

    ap_names = ["L%d" % i for i in range(len(picked))] + ["g"]
    formula = _formula_for(ap_names) if picked else normalize(F(AP("g")))

    final_asset = target_area.asset
    # the dataset's own description of the episode target is used verbatim for the last stage
    final_desc = episode.obj_desc

    task = {
        "task_id": "%s/%s" % (map_name, episode.seq),
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "map": map_name,
        "split": split_name,
        "seq": episode.seq,
        "traj_dir": episode.traj_dir.replace(os.sep, "/"),
        "event_radius_m": radius,
        "start_position": [float(x) for x in episode.start_position],
        "expert_frames": len(episode.positions),
        "expert_route_length_m": round(route_length(episode.positions), 3),
        "final_target": {
            "ap": "g",
            "area_index": target_area.index,
            "center": [float(x) for x in target_area.center],
            "asset": final_asset,
            "description": final_desc,
            "description_source": "own_episode_instruction",
        },
        "landmarks": [
            {
                "ap": ap_names[i],
                "area_index": item["area_index"],
                "center": item["center"],
                "asset": item["asset"],
                "description": item["description"],
                "description_source_seq": item["description_source_seq"],
                "first_entry_frame": item["frame"],
                "off_line_distance_m": item["off_line_distance_m"],
                "on_line": item["on_line"],
            }
            for i, item in enumerate(picked)
        ],
        "n_landmarks": len(picked),
        "n_stages": len(picked) + 1,
        "st_rt": st_rt,
        "n_off_line_landmarks": n_off_line,
        "event_order": ap_names,
        "formula": to_str(formula),
        "formula_ast": _ast_to_json(formula),
        "formula_single_goal": to_str(normalize(F(AP("g")))),
        "long_instruction_zh": long_instruction([p["asset"] for p in picked], final_asset, lang="zh"),
        "long_instruction_en": long_instruction([p["asset"] for p in picked], final_asset, lang="en"),
        "dropped_landmarks": dropped,
        "all_landmarks_along_route": enriched,
        "skipped": False,
        "skip_reason": None,
    }

    # a task is only usable if the *expert* trajectory satisfies its own long spec
    ok, residual = replay_check(task, episode, areas, radius=radius)
    if not ok:
        task["skipped"] = True
        task["skip_reason"] = "expert_trajectory_does_not_accept"
        task["expert_residual"] = residual
        skip_reasons.append("expert_trajectory_does_not_accept")
    return task


def _ast_to_json(phi: Formula) -> dict:
    from .formula import And, G, Not, Or, U

    if phi.__class__.__name__ in ("_True", "_False"):
        return {"op": "top" if phi.__class__.__name__ == "_True" else "bot"}
    if isinstance(phi, AP):
        return {"op": "ap", "name": phi.name}
    if isinstance(phi, Not):
        return {"op": "not", "arg": _ast_to_json(phi.operand)}
    if isinstance(phi, And):
        return {"op": "and", "args": [_ast_to_json(x) for x in phi.operands]}
    if isinstance(phi, Or):
        return {"op": "or", "args": [_ast_to_json(x) for x in phi.operands]}
    if isinstance(phi, F):
        return {"op": "F", "arg": _ast_to_json(phi.operand)}
    if isinstance(phi, G):
        return {"op": "G", "arg": _ast_to_json(phi.operand)}
    if isinstance(phi, U):
        return {"op": "U", "left": _ast_to_json(phi.left), "right": _ast_to_json(phi.right)}
    raise TypeError("unknown node %r" % (phi,))


def replay_check(task: dict, episode: Episode, areas: Sequence[MapArea], *, radius: float) -> Tuple[bool, str]:
    """Progress the task formula through the expert trajectory; must be accepted."""
    from .formula import accept, parse, prog, to_str as fstr

    phi = parse(task["formula"])
    landmarks = _landmarks_from_task(task)
    excluded = {task["final_target"]["area_index"]}
    hits = detect_landmarks(episode.positions, areas, radius=radius, exclude_area_indices=excluded)
    inside_prev: set = set()
    residual = normalize(phi)
    for t, p in enumerate(episode.positions):
        now = {L.ap for L in landmarks if L.is_inside(p)}
        # the final target is also an event source ("g")
        g_center = task["final_target"]["center"]
        if _euclid(p, g_center) < radius:
            now.add("g")
        new = now - inside_prev
        residual = prog(residual, new)
        inside_prev = now
    return accept(residual), fstr(residual)


def _landmarks_from_task(task: dict) -> List[Landmark]:
    out: List[Landmark] = []
    for item in task["landmarks"]:
        out.append(
            Landmark(
                ap=item["ap"],
                center=tuple(item["center"]),
                asset=item["asset"],
                description=item["description"] or "",
                area_index=item["area_index"],
            )
        )
    return out


def task_landmarks(task: dict) -> List[Landmark]:
    """All event sources of a task, including the final goal atom ``g``."""
    out = _landmarks_from_task(task)
    ft = task["final_target"]
    out.append(
        Landmark(
            ap=ft["ap"],
            center=tuple(ft["center"]),
            asset=ft["asset"],
            description=ft["description"],
            area_index=ft["area_index"],
        )
    )
    return out


# --------------------------------------------------------------------------------------
# split level driver
# --------------------------------------------------------------------------------------
def load_split(split_json: str) -> List[dict]:
    with open(split_json, "r") as f:
        return json.load(f)


def unique_sequences(split_entries: Sequence[dict]) -> List[str]:
    """Split files list the same trajectory once per start frame; keep the first occurrence."""
    seen: List[str] = []
    done = set()
    for e in split_entries:
        parts = e["json"].strip("/").split("/")
        seq = parts[-2]
        if seq in done:
            continue
        done.add(seq)
        seen.append(seq)
    return seen


def build_tasks_for_split(
    dataset_root: str,
    spawn_area_json: str,
    split_json: str,
    map_name: str,
    *,
    split_name: str = "seen_valset",
    max_landmarks: int = 3,
    limit: Optional[int] = None,
    description_limit: Optional[int] = None,
    log=print,
) -> Tuple[List[dict], dict]:
    areas = load_map_areas(spawn_area_json, map_name)
    entries = load_split(split_json)
    seqs = unique_sequences(entries)
    if limit:
        seqs = seqs[:limit]
    log("map=%s areas=%d split_entries=%d unique_trajectories=%d (building %d)"
        % (map_name, len(areas), len(entries), len(unique_sequences(entries)), len(seqs)))
    desc_index = build_description_index(dataset_root, map_name, areas, limit=description_limit)
    log("description index: %d areas have a real description" % len(desc_index))

    tasks: List[dict] = []
    reasons: List[str] = []
    for i, seq in enumerate(seqs):
        try:
            episode = load_episode(dataset_root, map_name, seq, areas)
        except Exception as exc:  # missing files etc.
            reasons.append("load_failed:%s" % type(exc).__name__)
            tasks.append({"task_id": "%s/%s" % (map_name, seq), "seq": seq, "skipped": True,
                          "skip_reason": "load_failed", "error": str(exc)})
            continue
        task = build_task(
            episode, areas, desc_index,
            map_name=map_name, split_name=split_name,
            max_landmarks=max_landmarks, skip_reasons=reasons,
        )
        tasks.append(task)
        if (i + 1) % 10 == 0 or i + 1 == len(seqs):
            log("  built %d/%d" % (i + 1, len(seqs)))

    summary = {
        "map": map_name,
        "split": split_name,
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "split_entries": len(entries),
        "unique_trajectories": len(unique_sequences(entries)),
        "tasks_built": len(tasks),
        "tasks_usable": sum(1 for t in tasks if not t.get("skipped")),
        "skipped": sum(1 for t in tasks if t.get("skipped")),
        "skip_reasons": _count(reasons),
        "stage_histogram": _count(str(t.get("n_stages")) for t in tasks if not t.get("skipped")),
        "st_rt_histogram": _count(t.get("st_rt") for t in tasks if not t.get("skipped")),
        "off_line_only_histogram": _count(
            "%d/%d off-line" % (t.get("n_off_line_landmarks", 0), t.get("n_landmarks", 0))
            for t in tasks if not t.get("skipped")
        ),
        "landmark_separation_check": _separation_report(tasks),
    }
    return tasks, summary


def _count(items: Iterable) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in items:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items()))


def _separation_report(tasks: Sequence[dict]) -> dict:
    """How many tasks have landmark pairs closer than 2*radius (ambiguity risk)."""
    close = 0
    total = 0
    for t in tasks:
        if t.get("skipped"):
            continue
        total += 1
        ours = [tuple(l["center"]) for l in t["landmarks"]] + [tuple(t["final_target"]["center"])]
        for i in range(len(ours)):
            for j in range(i + 1, len(ours)):
                if _euclid(ours[i], ours[j]) <= MIN_PAIRWISE_SEPARATION_M:
                    close += 1
                    break
    return {"tasks": total, "tasks_with_close_pair": close}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="M0 offline task construction (LTLf specs from episodes)")
    ap.add_argument("--dataset-root", required=True, help="e.g. dataset_raw/ (contains <map>/<seq>/...)")
    ap.add_argument("--spawn-area-json", required=True, help="data/meta/map_spawnarea_info.json")
    ap.add_argument("--split-json", required=True, help="data/uav_dataset/seen_valset_splits/<map>.json")
    ap.add_argument("--map", default="ModernCityMap")
    ap.add_argument("--split-name", default="seen_valset")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-landmarks", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    tasks, summary = build_tasks_for_split(
        dataset_root=args.dataset_root,
        spawn_area_json=args.spawn_area_json,
        split_json=args.split_json,
        map_name=args.map,
        split_name=args.split_name,
        max_landmarks=args.max_landmarks,
        limit=args.limit,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    tasks_path = os.path.join(args.out_dir, "tasks_%s_%s.json" % (args.map, args.split_name))
    summary_path = os.path.join(args.out_dir, "summary_%s_%s.json" % (args.map, args.split_name))
    with open(tasks_path, "w") as f:
        json.dump(tasks, f, indent=1)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print("wrote", tasks_path)
    print("wrote", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
