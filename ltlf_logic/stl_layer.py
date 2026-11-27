"""Idea-2 stage 2: the online STL layer that sits next to the LTLf stage machine.

The LTLf layer (idea 1) decides *what* to do next: it tracks the ordering of events with a residual
automaton and rewrites the prompt per stage.  The STL layer adds the three things a boolean
formula cannot express, all of them cheap and model-agnostic:

1. **Robustness monitoring.**  At every step we compute the quantitative robustness of the scheduled
   specification (per-stage margin in metres, deadline status, global safety margin).  This is logged
   and summarised per episode, and it is what makes "close to failing" visible.

2. **Deadline-pressure prompting.**  Each stage has a window ``[lo, hi]`` in macro steps derived from
   the expert demonstration.  When the remaining budget for the *current* stage falls below a
   fraction of the window, a single short sentence is appended to the stage description.  The
   baseline prompt anchors (``<image>\\nFly ... and find the target. ...\\nAction: ``) are preserved,
   so the distribution shift is one sentence, not a template change.

3. **Safety-envelope action shield.**  The safety predicate is ``G(alt_min <= altitude <= alt_max)``
   with the band calibrated on the expert demonstrations.  The executed action is a *continuous*
   decoded triple ``(fwd, down, yaw)``; given the current pose the next altitude is
   ``-(z + down)``, so keeping the band is an interval constraint on ``down`` (and ``fwd`` is clipped
   to a per-step maximum).  The shield is therefore an exact, minimal projection - no gradient, no
   retraining, no extra model call.  It is reported per episode (how often it fired and what it
   changed) so its effect can be ablated.

Everything here is deterministic and unit-tested without the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .stl import (
    NEG_INF,
    POS_INF,
    Sample,
    Stl,
    altitude_range,
    at,
    describe,
    mk_and,
    trace_from_positions,
)
from .stl_tasks import DEFAULT_EVENT_RADIUS_M, StlSpec, build_spec

__all__ = ["StlLayerConfig", "StlLayer", "DEFAULT_DEADLINE_TEMPLATES"]


# Two hand written templates; the first is used for Chinese-free English instructions (the baseline
# was trained on English descriptions), the second is a fallback when the window is nearly over.
DEFAULT_DEADLINE_TEMPLATES = (
    "You only have about {n} more steps to reach it.",
    "Hurry: it must be reached within about {n} steps.",
)


@dataclass
class StlLayerConfig:
    enabled: bool = True
    shield_enabled: bool = True
    deadline_enabled: bool = True
    # warn when the remaining budget for the current stage is below this fraction of its window
    deadline_pressure_frac: float = 0.6
    # never warn later than this many steps before the deadline
    deadline_min_steps: int = 2
    deadline_templates: Tuple[str, ...] = DEFAULT_DEADLINE_TEMPLATES
    # envelope (metres, altitude above ground); filled from the expert calibration
    alt_min_m: float = 1.0
    alt_max_m: float = 30.0
    fwd_max_m: float = 8.0
    # "arrived" hysteresis: a stage counts as robustly satisfied only this far inside the radius
    gate_margin_m: float = 0.5
    event_radius_m: float = DEFAULT_EVENT_RADIUS_M

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "shield_enabled": self.shield_enabled,
            "deadline_enabled": self.deadline_enabled,
            "deadline_pressure_frac": self.deadline_pressure_frac,
            "deadline_min_steps": self.deadline_min_steps,
            "alt_min_m": self.alt_min_m,
            "alt_max_m": self.alt_max_m,
            "fwd_max_m": self.fwd_max_m,
            "gate_margin_m": self.gate_margin_m,
            "event_radius_m": self.event_radius_m,
        }


@dataclass
class _EnvStlState:
    spec: StlSpec
    positions: List[Tuple[float, float, float]] = field(default_factory=list)
    last_step: int = -1
    best_margin: Dict[str, float] = field(default_factory=dict)
    best_step: Dict[str, int] = field(default_factory=dict)
    rho_safety: float = POS_INF
    worst_alt: float = 0.0
    shield_fired: int = 0
    shield_changed_m: float = 0.0
    shield_events: List[dict] = field(default_factory=list)
    deadline_warnings: int = 0
    last_warning: Dict[str, int] = field(default_factory=dict)
    warned_deadline: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    stage_margins: Dict[str, float] = field(default_factory=dict)


class StlLayer:
    """Per-environment online STL bookkeeping.  Indexed exactly like the batch state."""

    def __init__(
        self,
        tasks: Sequence[Mapping],
        config: Optional[StlLayerConfig] = None,
        *,
        envelope: Optional[Mapping[str, float]] = None,
        step_len: float = 4.3,
    ):
        self.config = config or StlLayerConfig()
        env = dict(envelope or {})
        cfg_alt_min = float(env.get("alt_min_m", self.config.alt_min_m))
        cfg_alt_max = float(env.get("alt_max_m", self.config.alt_max_m))
        self.config.alt_min_m = cfg_alt_min
        self.config.alt_max_m = cfg_alt_max
        self.states: List[_EnvStlState] = []
        for t in tasks:
            spec = build_spec(
                t,
                event_radius=self.config.event_radius_m,
                step_len=step_len,
                envelope={"alt_min_m": cfg_alt_min, "alt_max_m": cfg_alt_max},
            )
            self.states.append(_EnvStlState(spec=spec))

    # -- basics ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.states)

    def spec(self, i: int) -> StlSpec:
        return self.states[i].spec

    # -- 1. monitoring -------------------------------------------------------
    def observe(self, i: int, position: Sequence[float], step: Optional[int] = None) -> dict:
        """Feed one pose.  ``step`` is the *control step*, which can advance while the pose does not.

        The simulator publishes the same pose for all sub-samples of a macro step, but the control
        clock still advances; deadlines must follow the clock, so ``step`` is tracked separately from
        the number of distinct poses in the trace.
        """
        st = self.states[i]
        if step is None:
            step = st.last_step + 1 if st.last_step >= 0 else 0
        st.last_step = max(st.last_step, int(step))
        p = (float(position[0]), float(position[1]), float(position[2]))
        if st.positions and all(abs(p[k] - st.positions[-1][k]) < 1e-9 for k in range(3)):
            return self._snapshot(i)
        st.positions.append(p)
        for ap in st.spec.aps:
            c = st.spec.centers[ap]
            d = math.sqrt(sum((p[k] - c[k]) ** 2 for k in range(3)))
            m = self.config.event_radius_m - d
            if ap not in st.best_margin or m > st.best_margin[ap]:
                st.best_margin[ap] = m
                st.best_step[ap] = int(step)
            st.stage_margins[ap] = m
        alt = -p[2]
        st.worst_alt = max(abs(alt - 0.5 * (self.config.alt_min_m + self.config.alt_max_m)), st.worst_alt)
        st.rho_safety = min(
            alt - self.config.alt_min_m, self.config.alt_max_m - alt, st.rho_safety
        )
        return self._snapshot(i)

    def _snapshot(self, i: int) -> dict:
        st = self.states[i]
        step = max(0, st.last_step)
        due = {}
        for ap, (lo, hi) in st.spec.windows.items():
            due[ap] = {"lo": lo, "hi": hi, "remaining": hi - step}
        return {
            "step": step,
            "rho_safety": None if st.rho_safety in (POS_INF, NEG_INF) else round(st.rho_safety, 4),
            "stage_margins": {k: round(v, 4) for k, v in st.stage_margins.items()},
            "deadlines": due,
        }

    def stage_margin(self, i: int, ap: str) -> float:
        return self.states[i].best_margin.get(ap, NEG_INF)

    def robustly_arrived(self, i: int, ap: str) -> bool:
        """Hysteretic "arrived": strictly inside the radius by ``gate_margin_m``."""
        return self.stage_margin(i, ap) > self.config.gate_margin_m

    # -- 2. deadline pressure ------------------------------------------------
    def deadline_hint(self, i: int, ap: str, step: int) -> Optional[str]:
        st = self.states[i]
        if not self.config.deadline_enabled:
            return None
        win = st.spec.windows.get(ap)
        if win is None:
            return None
        lo, hi = win
        marker = (int(lo), int(hi))
        if st.warned_deadline.get(ap) == marker:
            return None  # already warned about this deadline
        remaining = hi - step
        if remaining <= 0:
            st.warned_deadline[ap] = marker
            st.deadline_warnings += 1
            return "You are already behind schedule for it, reach it as soon as possible."
        width = max(1, hi - lo)
        if remaining <= max(self.config.deadline_min_steps, int(round(self.config.deadline_pressure_frac * width))):
            st.warned_deadline[ap] = marker
            st.deadline_warnings += 1
            tmpl = self.config.deadline_templates[0] if remaining > self.config.deadline_min_steps else \
                self.config.deadline_templates[-1]
            return tmpl.format(n=max(1, remaining))
        return None

    # -- 3. safety shield ----------------------------------------------------
    def shield(self, i: int, action: Mapping[str, float], position: Sequence[float]) -> Tuple[dict, dict]:
        """Project ``(fwd, down, yaw)`` onto the safety envelope; returns (action, info)."""
        st = self.states[i]
        a = {k: float(v) for k, v in action.items()}
        info = {"fired": False, "reason": "", "delta": 0.0, "action_before": dict(a)}
        if not (self.config.enabled and self.config.shield_enabled):
            return a, info
        z = float(position[2])
        alt_min, alt_max = self.config.alt_min_m, self.config.alt_max_m
        down = a.get("down", 0.0)
        # next altitude = -(z + down) must stay within [alt_min, alt_max]
        down_lo = -alt_max - z
        down_hi = -alt_min - z
        reasons: List[str] = []
        if down_hi < down_lo:  # current altitude already outside the band: do not make it worse
            down_lo = down_hi = -z - 0.5 * (alt_min + alt_max)
        new_down = min(max(down, down_lo), down_hi)
        if abs(new_down - down) > 1e-6:
            reasons.append("altitude")
            a["down"] = new_down
        fwd = a.get("fwd", 0.0)
        new_fwd = min(max(fwd, 0.0), self.config.fwd_max_m)
        if abs(new_fwd - fwd) > 1e-6:
            reasons.append("fwd")
            a["fwd"] = new_fwd
        delta = math.sqrt(
            (a.get("fwd", 0.0) - info["action_before"].get("fwd", 0.0)) ** 2
            + (a.get("down", 0.0) - info["action_before"].get("down", 0.0)) ** 2
        )
        # the altitude displacement itself is what matters most: metres of vertical change removed
        info["delta"] = round(abs(a.get("down", 0.0) - info["action_before"].get("down", 0.0)), 4)
        if reasons:
            info["fired"] = True
            info["reason"] = "+".join(reasons)
            st.shield_fired += 1
            st.shield_changed_m += info["delta"]
            if len(st.shield_events) < 40:
                st.shield_events.append(
                    {
                        "step": max(0, st.last_step),
                        "reason": info["reason"],
                        "before": {k: round(v, 3) for k, v in info["action_before"].items()},
                        "after": {k: round(v, 3) for k, v in a.items()},
                    }
                )
        return a, info

    # -- summary -------------------------------------------------------------
    def summary(self, i: int) -> dict:
        st = self.states[i]
        spec = st.spec
        steps = max(1, st.last_step)
        per_stage = {}
        for ap in spec.aps:
            win = spec.windows[ap]
            m = st.best_margin.get(ap, NEG_INF)
            s = st.best_step.get(ap, -1)
            per_stage[ap] = {
                "best_margin_m": None if m == NEG_INF else round(m, 4),
                "best_step": s,
                "window": list(win),
                "inside_window": bool(m > 0 and win[0] <= s <= win[1]),
                "late": bool(m > 0 and s > win[1]),
                "never_entered": bool(m <= 0),
            }
        reached = [ap for ap in spec.aps if per_stage[ap]["best_margin_m"] is not None and per_stage[ap]["best_margin_m"] > 0]
        return {
            "enabled": bool(self.config.enabled),
            "steps": max(0, st.last_step),
            "config": self.config.to_dict(),
            "rho_safety": None if st.rho_safety in (POS_INF, NEG_INF) else round(st.rho_safety, 4),
            "safety_violated": bool(st.rho_safety < 0),
            "per_stage": per_stage,
            "stages_reached": len(reached),
            "stages_total": len(spec.aps),
            "schedule_satisfied": len(reached) == len(spec.aps)
            and all(per_stage[ap]["inside_window"] for ap in spec.aps),
            "stage_late": sum(1 for ap in spec.aps if per_stage[ap]["late"]),
            "shield_fired": st.shield_fired,
            "shield_fired_rate": round(st.shield_fired / steps, 4),
            "shield_changed_m_total": round(st.shield_changed_m, 3),
            "shield_events": st.shield_events,
            "deadline_warnings": st.deadline_warnings,
            "windows": {k: list(v) for k, v in spec.windows.items()},
        }
