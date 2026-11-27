"""M3/M5 helpers: phase plans, prompt injection, landing gate, fallback takeover.

Everything in this module is pure (no torch / no simulator / no baseline imports), so it can be
unit-tested on a laptop.  The loop that actually talks to the simulator lives in
:mod:`ltlf_logic.eval_loop`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .automaton import build_automaton, enumerate_accepting_paths
from .formula import Formula, accept, atoms, normalize, parse, to_str
from .monitor import DEFAULT_EVENT_RADIUS_M, Landmark, Monitor

__all__ = [
    "LTLfConfig",
    "TaskPlan",
    "build_stage_instruction",
    "next_target_atom",
    "takeover_action",
    "should_gate_stop",
    "StageRecord",
    "EpisodeLogger",
    "STEP_LOG_FIELDS",
]

# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
MODES = ("original", "single_stage", "long_instruction", "fixed_order", "stl")


@dataclass
class LTLfConfig:
    """All knobs of the logic layer.  Every field is recorded in the run summary."""

    mode: str = "single_stage"
    task_json: Optional[str] = None
    gate_enabled: bool = False
    takeover_enabled: bool = False
    takeover_fwd_m: float = 3.0
    takeover_min_fwd_m: float = 2.0
    disable_distance_increasing: bool = False
    event_radius_m: float = DEFAULT_EVENT_RADIUS_M
    max_stages: int = 4
    log_every_step: bool = True
    seed_note: str = ""
    # ---- idea 2: STL layer (only used when mode == "stl") ----
    stl_enabled: bool = False
    stl_shield: bool = True
    stl_deadline: bool = True
    # safety envelope calibrated on the ModernCityMap expert flights (p01/p99 of altitude above
    # ground, see work/idea2-stl.md §3.2).  Per-map calibration is a TODO; the STL arm currently
    # runs on ModernCityMap only.
    stl_alt_min_m: float = 1.722
    stl_alt_max_m: float = 33.638
    stl_fwd_max_m: float = 8.0
    stl_step_len_m: float = 4.3

    @classmethod
    def for_mode(cls, mode: str, **overrides) -> "LTLfConfig":
        base = cls(mode=mode)
        if mode in ("original", "single_stage"):
            base.gate_enabled = False
            base.takeover_enabled = False
            base.disable_distance_increasing = False
        elif mode == "long_instruction":
            base.gate_enabled = False
            base.takeover_enabled = False
            base.disable_distance_increasing = True
        elif mode == "fixed_order":
            base.gate_enabled = True
            base.takeover_enabled = True
            base.disable_distance_increasing = True
        elif mode == "stl":
            # idea 2: the same stage machine as fixed_order, plus STL monitoring, deadline
            # pressure prompting and the safety-envelope action shield
            base.gate_enabled = True
            base.takeover_enabled = True
            base.disable_distance_increasing = True
            base.stl_enabled = True
        else:
            raise ValueError("unknown mode %r (expected one of %s)" % (mode, MODES))
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# prompt injection (only the two values the baseline reads)
# --------------------------------------------------------------------------------------
def build_stage_instruction(description: str, *, direction: str = "front", yaw_deg: float = 0.0) -> str:
    """Build an instruction string that the baseline splits into ``obj_desc`` verbatim.

    ``aerovla_wrapper_ui.py:93`` does

        instructions[i].split("degrees from you.")[1].split(" Please control")[0].strip()

    so the two anchors ``degrees from you.`` and `` Please control`` must be present.  The
    direction / yaw prefix is cosmetic (it is discarded by the baseline) - we keep the dataset's
    wording so the prompt stays inside the training format.
    """
    if "degrees from you." in description or " Please control" in description:
        raise ValueError("stage description must not contain the prompt anchors")
    return (
        "<image>\nThere is a target in the %s of uav. Using your front as the x-axis and "
        "your right as the y-axis, The target is at a yaw angle of %.1f degrees from you. %s "
        "Please control the drone and find the target." % (direction, yaw_deg, description)
    )


# --------------------------------------------------------------------------------------
# task plan: formula + landmarks + per-stage descriptions
# --------------------------------------------------------------------------------------
@dataclass
class StageRecord:
    stage_idx: int
    ap: str
    asset: str
    center: List[float]
    description: str
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    entered: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskPlan:
    """One episode's formula, landmarks and phase bookkeeping."""

    task_id: str
    formula: Formula
    landmarks: List[Landmark]
    descriptions: Dict[str, str]
    assets: Dict[str, str]
    mode: str
    st_rt: str = ""
    n_stages: int = 1
    single_goal: bool = False
    stages: List[StageRecord] = field(default_factory=list)
    stage_switches: int = 0
    plan_switches: int = 0

    # -- construction ----------------------------------------------------------
    @classmethod
    def from_task_dict(cls, task: dict, *, mode: str, single_goal: bool = False) -> "TaskPlan":
        if single_goal:
            phi = normalize(parse(task["formula_single_goal"]))
            ft = task["final_target"]
            landmarks = [
                Landmark(
                    ap="g",
                    center=tuple(ft["center"]),
                    asset=ft["asset"],
                    description=ft["description"],
                    area_index=ft["area_index"],
                )
            ]
        else:
            phi = normalize(parse(task["formula"]))
            landmarks = []
            for item in task["landmarks"]:
                landmarks.append(
                    Landmark(
                        ap=item["ap"],
                        center=tuple(item["center"]),
                        asset=item["asset"],
                        description=item["description"] or "",
                        area_index=item["area_index"],
                    )
                )
            ft = task["final_target"]
            landmarks.append(
                Landmark(
                    ap=ft["ap"],
                    center=tuple(ft["center"]),
                    asset=ft["asset"],
                    description=ft["description"],
                    area_index=ft["area_index"],
                )
            )
        descriptions = {L.ap: L.description for L in landmarks}
        assets = {L.ap: L.asset for L in landmarks}
        plan = cls(
            task_id=task["task_id"],
            formula=phi,
            landmarks=landmarks,
            descriptions=descriptions,
            assets=assets,
            mode=mode,
            st_rt=task.get("st_rt", ""),
            n_stages=len(landmarks),
            single_goal=single_goal,
        )
        plan.reset_stages()
        return plan

    def reset_stages(self) -> None:
        self.stages = []
        for i, L in enumerate(self.landmarks):
            self.stages.append(
                StageRecord(
                    stage_idx=i,
                    ap=L.ap,
                    asset=L.asset,
                    center=[float(c) for c in L.center],
                    description=self.descriptions.get(L.ap, ""),
                )
            )

    # -- queries ---------------------------------------------------------------
    def landmark(self, ap: str) -> Landmark:
        for L in self.landmarks:
            if L.ap == ap:
                return L
        raise KeyError(ap)

    def stage_index(self, ap: str) -> int:
        for i, L in enumerate(self.landmarks):
            if L.ap == ap:
                return i
        raise KeyError(ap)

    def final_landmark(self) -> Landmark:
        return self.landmarks[-1]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "formula": to_str(self.formula),
            "mode": self.mode,
            "single_goal": self.single_goal,
            "st_rt": self.st_rt,
            "n_stages": self.n_stages,
            "landmarks": [L.to_dict() for L in self.landmarks],
            "stage_switches": self.stage_switches,
            "plan_switches": self.plan_switches,
        }


def next_target_atom(
    residual: Formula,
    plan: TaskPlan,
    *,
    preset_atoms: Optional[Sequence[str]] = None,
    max_len: int = 8,
) -> Optional[str]:
    """Which atom should the drone fly to next, according to the *residual* formula?

    Recomputing the next phase from the residual (instead of keeping a mutable pointer) is what
    makes the plan re-derivable after any event order; v1 always flies the shortest accepting
    order, which for ``F(L0 & F(L1 & ... F(g)))`` is the fixed order of the task.
    """
    if accept(residual):
        return None
    atom_list = list(preset_atoms) if preset_atoms else sorted(atoms(plan.formula))
    aut = build_automaton(residual, atom_list=atom_list, subsets=False)
    paths = enumerate_accepting_paths(aut, singleton_only=True, max_len=max_len)
    if not paths:
        return None
    first = next(iter(paths[0][0]))
    return first


def should_gate_stop(model_stop: bool, residual: Formula, gate_enabled: bool) -> Tuple[bool, str]:
    """M5 landing gate.  Returns ``(allow_stop, reason)``.

    The model's LAND is a *phase* termination request, not a mission termination signal: it is
    only forwarded to the baseline success logic when the residual formula is accepted.
    """
    if not model_stop:
        return False, "no_stop"
    if not gate_enabled:
        return True, "gate_disabled"
    if accept(residual):
        return True, "accepted"
    return False, "suppressed_residual_pending"


# --------------------------------------------------------------------------------------
# fallback takeover (M5 L2): a near-zero action that would trip the stuck detector
# --------------------------------------------------------------------------------------
def body_frame_bearing(
    current_position: Sequence[float], current_quaternion: Sequence[float], target: Sequence[float]
) -> Tuple[float, float, float]:
    """Return ``(bearing_rad, elev_rad, distance_m)`` from the drone body frame to ``target``.

    ``bearing`` is the yaw change needed so that the body x axis points at the target, and
    ``elev`` is the vertical offset (positive = up in the sim frame).  Uses only ``math`` so it
    stays dependency free.
    """
    dx = float(target[0]) - float(current_position[0])
    dy = float(target[1]) - float(current_position[1])
    dz = float(target[2]) - float(current_position[2])
    x, y, z, w = (float(v) for v in current_quaternion[:4])
    # yaw of the body frame (AirSim uses NED-ish x forward, y right)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    bearing_world = math.atan2(dy, dx)
    bearing = _wrap(bearing_world - yaw)
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    return bearing, dz, dist


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def takeover_action(
    current_position: Sequence[float],
    current_quaternion: Sequence[float],
    target: Sequence[float],
    config: LTLfConfig,
    *,
    yaw_threshold_rad: float = 0.25,
) -> dict:
    """Replacement action that makes progress toward the current sub-goal.

    Mirrors the simulator's own action decoding (``move_path_by_actions``):
    ``fwd`` is a displacement in metres along ``yaw + pred_yaw``, a command with
    ``|pred_yaw| >= 0.25`` makes the simulator rotate in place first (and then only move
    vertically), so we keep the yaw command under that threshold whenever we want translation.
    """
    bearing, dz, dist = body_frame_bearing(current_position, current_quaternion, target)
    if abs(bearing) >= yaw_threshold_rad:
        # pure rotation this step, no translation (and no vertical drift)
        return {"fwd": 0.0, "down": 0.0, "yaw": max(-1.1, min(1.1, bearing))}
    fwd = max(config.takeover_min_fwd_m, min(config.takeover_fwd_m, dist))
    return {"fwd": float(fwd), "down": 0.0, "yaw": bearing}


def is_near_zero_action(action: dict, eps: float = 0.01) -> bool:
    """Same predicate the baseline uses for its implicit LAND (``aerovla_wrapper_ui.py:195``)."""
    return (
        float(action.get("fwd", 0.0)) < eps
        and abs(float(action.get("down", 0.0))) < eps
        and abs(float(action.get("yaw", 0.0))) < eps
    )


# --------------------------------------------------------------------------------------
# per-step logging
# --------------------------------------------------------------------------------------
STEP_LOG_FIELDS = (
    "seq",
    "task_id",
    "map",
    "mode",
    "step",
    "stage_idx",
    "stage_ap",
    "position",
    "guide_point",
    "stage_description",
    "events",
    "new_events",
    "residual",
    "accepted",
    "violated",
    "model_stop",
    "model_stop_cause",
    "gated_stop",
    "gate_reason",
    "takeover",
    "takeover_reason",
    "action_model",
    "action_executed",
    "distance_to_final",
    "distance_to_guide",
    "stage_switch",
    "ltlf_switches",
)


class EpisodeLogger:
    """Writes one JSONL file per episode plus a summary dict."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._files: Dict[str, "os.IO"] = {}
        self.summaries: List[dict] = []

    def write_step(self, seq: str, record: dict) -> None:
        path = os.path.join(self.log_dir, "%s.jsonl" % _safe(seq))
        with open(path, "a") as f:
            f.write(json.dumps(record, sort_keys=False) + "\n")

    def write_summary(self, summary: dict) -> None:
        self.summaries.append(summary)
        path = os.path.join(self.log_dir, "%s.summary.json" % _safe(summary.get("seq", "unknown")))
        with open(path, "w") as f:
            json.dump(summary, f, indent=1)

    def flush_index(self) -> str:
        path = os.path.join(self.log_dir, "_index.json")
        with open(path, "w") as f:
            json.dump({"n_episodes": len(self.summaries), "episodes": self.summaries}, f, indent=1)
        return path


def _safe(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(name))
