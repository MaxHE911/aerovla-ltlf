"""L0: offline replay of a task spec over a pose sequence + independent ESR judge.

Two algorithms live here on purpose:

* :func:`online_equivalent_esr` mimics what the *online* monitor does (event diffing +
  ``prog`` + ``accept``);
* :func:`independent_esr` rebuilds the event sequence from positions and evaluates the spec
  with the direct finite-trace semantics (:mod:`ltlf_logic.semantics`), i.e. without the
  residual machinery.  ``idea1`` §5.5 requires the offline judge to be independent of the
  online logic layer; disagreements are reported instead of hidden.

Nothing here touches the simulator: it only needs poses plus the task record built by M0.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .formula import accept, normalize, parse, prog, to_str
from .semantics import sat
from .tasks import task_landmarks

__all__ = [
    "replay_positions",
    "online_equivalent_esr",
    "independent_esr",
    "events_from_positions",
    "negative_cases",
    "check_task_replay",
]


def events_from_positions(
    task: dict,
    positions: Sequence[Sequence[float]],
    *,
    radius: Optional[float] = None,
    dedupe_consecutive: bool = True,
) -> List[FrozenSet[str]]:
    """Rebuild the event trace (one event set per sampling point) from a pose sequence.

    ``dedupe_consecutive`` drops repeated identical poses: the simulator returns the same
    post-step state 5 times per control step (see ``work/stage0_audit.md`` risk 1), so without
    de-duplication the trace would contain 5x redundant time points.
    """
    radius = float(radius if radius is not None else task.get("event_radius_m", 12.0))
    landmarks = task_landmarks(task)
    trace: List[FrozenSet[str]] = []
    prev: Optional[Tuple[float, float, float]] = None
    for p in positions:
        pos = (float(p[0]), float(p[1]), float(p[2]))
        if dedupe_consecutive and prev is not None and all(abs(pos[i] - prev[i]) < 1e-9 for i in range(3)):
            continue
        prev = pos
        events = frozenset(L.ap for L in landmarks if L.distance_to(pos) < radius)
        trace.append(events)
    return trace


def online_equivalent_esr(
    task: dict,
    positions: Sequence[Sequence[float]],
    *,
    radius: Optional[float] = None,
    dedupe_consecutive: bool = True,
) -> dict:
    """ESR computed exactly like the online monitor: diffing inside-sets, then progression."""
    phi = parse(task["formula"])
    landmarks = task_landmarks(task)
    radius = float(radius if radius is not None else task.get("event_radius_m", 12.0))
    residual = normalize(phi)
    inside_prev: FrozenSet[str] = frozenset()
    order: List[str] = []
    prev = None
    points = 0
    for p in positions:
        pos = (float(p[0]), float(p[1]), float(p[2]))
        if dedupe_consecutive and prev is not None and all(abs(pos[i] - prev[i]) < 1e-9 for i in range(3)):
            continue
        prev = pos
        points += 1
        events = frozenset(L.ap for L in landmarks if L.distance_to(pos) < radius)
        new = events - inside_prev
        if new:
            order.extend(sorted(new))
        residual = prog(residual, new)
        inside_prev = events
    return {
        "accepted": accept(residual),
        "residual": to_str(residual),
        "events_order": order,
        "sampling_points": points,
        "formula": to_str(phi),
    }


def independent_esr(
    task: dict,
    positions: Sequence[Sequence[float]],
    *,
    radius: Optional[float] = None,
    dedupe_consecutive: bool = True,
) -> dict:
    """ESR recomputed with the direct finite-trace semantics (no progression, no residual)."""
    phi = parse(task["formula"])
    trace = events_from_positions(task, positions, radius=radius, dedupe_consecutive=dedupe_consecutive)
    order: List[str] = []
    for ev in trace:
        order.extend(sorted(ev))
    return {
        "accepted": sat(phi, trace),
        "events_order": order,
        "sampling_points": len(trace),
        "multi_event_points": sum(1 for ev in trace if len(ev) > 1),
        "formula": to_str(phi),
    }


def replay_positions(
    task: dict, positions: Sequence[Sequence[float]], *, radius: Optional[float] = None
) -> dict:
    """Both judges side by side; ``agree`` flags any divergence."""
    a = online_equivalent_esr(task, positions, radius=radius)
    b = independent_esr(task, positions, radius=radius)
    return {
        "accepted_online": a["accepted"],
        "accepted_independent": b["accepted"],
        "agree": a["accepted"] == b["accepted"],
        "residual": a["residual"],
        "events_order": a["events_order"],
        "sampling_points": a["sampling_points"],
        "multi_event_points": b["multi_event_points"],
        "formula": a["formula"],
    }


# --------------------------------------------------------------------------------------
# negative samples: a correct judge must reject all of these
# --------------------------------------------------------------------------------------
def _waypoint_trace(task: dict, order: Sequence[str], *, end_at_goal: bool = True) -> List[Tuple[float, float, float]]:
    """Synthetic pose trace that visits the given atoms in order (one pose per atom)."""
    centers: Dict[str, Tuple[float, float, float]] = {L.ap: tuple(L.center) for L in task_landmarks(task)}
    trace = [(-1000.0, -1000.0, 100.0)]  # a far away start point, no event
    for ap in order:
        trace.append(centers[ap])
    return trace


def negative_cases(task: dict) -> List[dict]:
    """Deterministic mutation suite for one task.

    Each case is ``(name, positions, expected_accept)``; a correct ESR judge must return False
    for every case here (the positive control is the unmodified expert order).
    """
    order = list(task["event_order"])
    cases: List[dict] = []

    # 0. positive control: exactly the required order
    cases.append({"name": "positive_control", "positions": _waypoint_trace(task, order), "expected": True})

    # 1. missing event (drop one intermediate landmark)
    if len(order) > 2:
        dropped = order[0]
        cases.append(
            {"name": "missing_%s" % dropped, "positions": _waypoint_trace(task, [a for a in order if a != dropped]),
             "expected": False}
        )
    # 2. missing final goal -> cannot be accepted
    cases.append(
        {"name": "missing_final_goal", "positions": _waypoint_trace(task, order[:-1]), "expected": False}
    )

    # 3. swapped order (first two phases) for ordered specs
    if len(order) >= 3:
        swapped = [order[1], order[0]] + order[2:]
        cases.append({"name": "swapped_first_two", "positions": _waypoint_trace(task, swapped), "expected": False})

    # 4. reverse order (also meaningful for a two-stage spec: goal first, landmark after)
    if len(order) >= 2:
        cases.append({"name": "reversed", "positions": _waypoint_trace(task, list(reversed(order))), "expected": False})

    # 5. violation of an explicit G (avoid) clause, if the spec has one
    if "G(" in task["formula"]:
        avoid = task.get("avoid_atom")
        if avoid:
            bad_order = order + [avoid]
            cases.append({"name": "violates_G", "positions": _waypoint_trace(task, bad_order), "expected": False})

    return cases


def check_task_replay(task: dict, expert_positions: Sequence[Sequence[float]], *, radius: Optional[float] = None) -> dict:
    """Run the positive expert replay plus the negative suite for one task."""
    positive = replay_positions(task, expert_positions, radius=radius)
    negatives = []
    for case in negative_cases(task):
        res = replay_positions(task, case["positions"], radius=radius)
        negatives.append(
            {
                "name": case["name"],
                "expected": case["expected"],
                "online": res["accepted_online"],
                "independent": res["accepted_independent"],
                "agree": res["agree"],
                "ok": res["accepted_online"] == case["expected"] and res["accepted_independent"] == case["expected"],
            }
        )
    return {
        "task_id": task["task_id"],
        "formula": task["formula"],
        "n_stages": task["n_stages"],
        "st_rt": task["st_rt"],
        "expert_accepted_online": positive["accepted_online"],
        "expert_accepted_independent": positive["accepted_independent"],
        "experts_agree": positive["agree"],
        "events_order": positive["events_order"],
        "sampling_points": positive["sampling_points"],
        "negatives": negatives,
        "negatives_all_ok": all(n["ok"] for n in negatives),
        "judges_agree_everywhere": positive["agree"] and all(n["agree"] for n in negatives),
    }
