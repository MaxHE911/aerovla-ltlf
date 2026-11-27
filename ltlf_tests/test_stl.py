"""Unit tests for idea 2: sampled-time STL, spec construction, and the online STL layer.

Everything here runs without the simulator, without torch and without the baseline package, so these
tests are the cheap safety net for the STL path (``ltlf_tests/test_formula.py`` etc. cover idea 1).
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ltlf_logic.formula import parse
from ltlf_logic.semantics import sat
from ltlf_logic.stl import (
    NEG_INF,
    POS_INF,
    F,
    G,
    Not,
    Until,
    altitude_range,
    at,
    describe,
    displacement_max,
    mk_and,
    mk_or,
    trace_from_positions,
)
from ltlf_logic.stl_layer import StlLayer, StlLayerConfig
from ltlf_logic.stl_tasks import build_spec, derive_step_len_from_logs, evaluate_trace

RADIUS = 12.0


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def line_trace(n: int, step: float = 1.0):
    return trace_from_positions([(i * step, 0.0, 0.0) for i in range(n)])


def make_task(*, frames=100, route=100.0, lm=((-60.0, 0.0, -4.0), (-30.0, 0.0, -4.0)), final=(0.0, 0.0, -4.0)):
    """A synthetic M0-style task on a straight line from -90 to 0, flown at 4 m altitude."""
    landmarks = []
    route_lms = []
    for k, c in enumerate(lm):
        landmarks.append(
            {"ap": "L%d" % k, "area_index": 100 + k, "center": list(c), "asset": "a%d" % k, "description": "d%d" % k}
        )
        route_lms.append({"area_index": 100 + k, "frame": int(round(frames * (k + 1) / (len(lm) + 1)))})
    return {
        "task_id": "M/seq1",
        "seq": "seq1",
        "map": "M",
        "st_rt": "RT",
        "start_position": [-90.0, 0.0, -4.0],
        "expert_frames": frames,
        "expert_route_length_m": route,
        "landmarks": landmarks,
        "final_target": {"ap": "g", "area_index": 999, "center": list(final), "asset": "f", "description": "df"},
        "all_landmarks_along_route": route_lms,
    }


# --------------------------------------------------------------------------------------
# 1. semantics
# --------------------------------------------------------------------------------------
def test_predicate_margin_is_metres():
    tr = line_trace(5)
    phi = at("L", (2.0, 0.0, 0.0), 1.0)
    # at t=2 the point is exactly the centre -> margin = radius
    assert phi.robustness(tr, 2) == pytest.approx(1.0)
    # at t=0 the point is 2 m away -> margin = 1 - 2
    assert phi.robustness(tr, 0) == pytest.approx(-1.0)


def test_and_or_not_aggregate_like_min_max():
    tr = trace_from_positions([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    a = at("A", (0.0, 0.0, 0.0), 10.0)  # margins 10, 0
    b = at("B", (5.0, 0.0, 0.0), 7.0)   # margins 2, -3
    assert mk_and(a, b).robustness(tr) == pytest.approx(2.0)   # pointwise at t=0: min(10, 2)
    assert mk_or(a, b).robustness(tr) == pytest.approx(10.0)
    assert Not(a).robustness(tr) == pytest.approx(-10.0)


def test_eventually_always_bounds():
    tr = trace_from_positions([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)])
    far = at("far", (20.0, 0.0, 0.0), 12.0)
    assert F(far, 0, 1).robustness(tr) == pytest.approx(2.0)      # reached at t=1: 12-10
    assert G(far, 0, 3).robustness(tr) == pytest.approx(-8.0)     # worst at t=0: 12-20


def test_bounded_window_respects_lower_bound():
    tr = trace_from_positions([(0.0, 0.0, 0.0), (30.0, 0.0, 0.0)])
    p = at("p", (30.0, 0.0, 0.0), 1.0)
    # the event only happens at t=1, so a window that excludes t=1 must fail
    assert F(p, 0, 0).robustness(tr) < 0
    assert F(p, 1, 1).robustness(tr) > 0


def test_until_semantics():
    tr = trace_from_positions([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    left = at("safe", (0.0, 0.0, 0.0), 100.0)
    right = at("goal", (2.0, 0.0, 0.0), 1.0)
    # goal is reached at t=2 and ``safe`` holds everywhere before -> robustness = min(1, 99)
    assert Until(left, right, 0, 3).robustness(tr) == pytest.approx(1.0)


def test_partial_bounds_are_optimistic_vs_pessimistic():
    tr = trace_from_positions([(0.0, 0.0, 0.0)])  # only one sample
    p = at("p", (100.0, 0.0, 0.0), 10.0)
    phi = F(p, 0, 5)
    lo = phi.robustness(tr)
    hi = phi.robustness_upper(tr)
    assert lo < 0 < hi  # not satisfied yet, but can still be satisfied


def test_displacement_and_altitude_predicates():
    tr = trace_from_positions([(0.0, 0.0, -5.0), (3.0, 0.0, -5.0)])
    assert displacement_max(4.0).robustness(tr, 1) == pytest.approx(1.0)
    assert displacement_max(2.0).robustness(tr, 1) == pytest.approx(-1.0)
    band = altitude_range(1.0, 8.0)
    assert band.robustness(tr, 0) == pytest.approx(3.0)  # altitude 5 -> min(5-1, 8-5)=3


# --------------------------------------------------------------------------------------
# 2. spec construction
# --------------------------------------------------------------------------------------
def test_windows_are_ordered_and_well_formed():
    spec = build_spec(make_task(), step_len=4.0)
    los = [spec.windows[ap][0] for ap in spec.aps]
    assert los == sorted(los)                       # windows follow the expert order
    assert all(lo < hi for lo, hi in spec.windows.values())
    assert all(lo >= 0 for lo, _ in spec.windows.values())


def test_chain_equals_nested_ltlf_formula():
    """The STL ``chain`` boolean must agree with the LTLf formula it mirrors."""
    task = make_task()
    spec = build_spec(task, step_len=4.0)
    # a trace that visits L0, then L1, then the goal, in order
    positions = [(-90.0, 0.0, 0.0), (-60.0, 0.0, 0.0), (-30.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    trace = trace_from_positions(positions)
    # events implied by the same radius rule
    events = []
    for p in positions:
        ev = set()
        for ap in spec.aps:
            c = spec.centers[ap]
            if math.dist(p, c) < RADIUS:
                ev.add(ap)
        events.append(frozenset(ev))
    phi = parse(task_ltlf_string(task))
    assert sat(phi, events) == (spec.chain.robustness(trace) > 0)


def task_ltlf_string(task):
    """Mirror of the M0 task formula for the synthetic task: F(L0 & F(L1 & F(g)))."""
    aps = ["L0", "L1", "g"]
    s = "F(%s)" % aps[-1]
    for ap in reversed(aps[:-1]):
        s = "F(%s & %s)" % (ap, s)
    return s


def test_chain_not_satisfied_when_order_wrong():
    task = make_task()
    spec = build_spec(task, step_len=4.0)
    positions = [(-90.0, 0.0, 0.0), (0.0, 0.0, 0.0)]  # jumps straight to the goal
    trace = trace_from_positions(positions)
    assert spec.chain.robustness(trace) < 0


def test_schedule_has_deadlines_and_is_stricter_than_chain():
    task = make_task()
    spec = build_spec(task, step_len=4.0)
    # hover at the start until every window is over, then pass the landmarks in order: the
    # unbounded ``chain`` is satisfied, the scheduled specification is not
    n_hi = max(hi for _, hi in spec.windows.values())
    positions = [(-90.0, 0.0, -4.0)] * (n_hi + 3)
    positions += [(-60.0, 0.0, -4.0), (-30.0, 0.0, -4.0), (0.0, 0.0, -4.0)]
    ev = evaluate_trace(spec, positions)
    assert ev["chain_sat"] is True
    assert ev["schedule_sat"] is False


def test_safety_envelope_from_calibration():
    task = make_task()
    spec = build_spec(task, step_len=4.0, envelope={"alt_min_m": 1.0, "alt_max_m": 6.0})
    ok = evaluate_trace(spec, [(-90.0, 0.0, -3.0), (-60.0, 0.0, -3.0)])
    high = evaluate_trace(spec, [(-90.0, 0.0, -12.0), (-60.0, 0.0, -12.0)])
    assert ok["safety_sat"] is True
    assert high["safety_sat"] is False


# --------------------------------------------------------------------------------------
# 3. online layer: shield + deadline prompting
# --------------------------------------------------------------------------------------
def test_shield_projects_altitude_into_envelope():
    layer = StlLayer([make_task()], StlLayerConfig(alt_min_m=2.0, alt_max_m=8.0))
    # altitude 5 m (z=-5); a down=-10 would put us at 15 m -> must be clipped to 8 m
    act, info = layer.shield(0, {"fwd": 3.0, "down": -10.0, "yaw": 0.0}, (-90.0, 0.0, -5.0))
    assert info["fired"] and "altitude" in info["reason"]
    alt_after = -( -5.0 + act["down"] )
    assert 2.0 - 1e-6 <= alt_after <= 8.0 + 1e-6


def test_shield_clips_forward_and_is_idempotent():
    layer = StlLayer([make_task()], StlLayerConfig(alt_min_m=1.0, alt_max_m=20.0, fwd_max_m=6.0))
    act, info = layer.shield(0, {"fwd": 50.0, "down": 0.0, "yaw": 0.0}, (-90.0, 0.0, -5.0))
    assert act["fwd"] == pytest.approx(6.0) and info["fired"]
    act2, info2 = layer.shield(0, act, (-90.0, 0.0, -5.0))
    assert not info2["fired"] and act2 == act


def test_shield_leaves_safe_action_untouched():
    layer = StlLayer([make_task()], StlLayerConfig(alt_min_m=1.0, alt_max_m=20.0, fwd_max_m=6.0))
    act, info = layer.shield(0, {"fwd": 3.0, "down": 0.0, "yaw": 0.1}, (-90.0, 0.0, -5.0))
    assert not info["fired"] and act["fwd"] == pytest.approx(3.0)


def test_deadline_hint_fires_once_near_the_window():
    layer = StlLayer([make_task()], StlLayerConfig(deadline_enabled=True, deadline_pressure_frac=0.6))
    spec = layer.spec(0)
    lo, hi = spec.windows["L0"]
    # far before the window -> no hint
    assert layer.deadline_hint(0, "L0", 0) is None
    # just before the deadline -> a hint, and only once for that step
    hint = layer.deadline_hint(0, "L0", hi - 1)
    assert hint and "steps" in hint
    assert layer.deadline_hint(0, "L0", hi - 1) is None


def test_monitor_tracks_best_margin_and_safety():
    layer = StlLayer([make_task()], StlLayerConfig(alt_min_m=1.0, alt_max_m=8.0))
    for p in [(-90.0, 0.0, -4.0), (-62.0, 0.0, -4.0), (-59.0, 0.0, -4.0)]:
        layer.observe(0, p)
    # L0 centre is at -60: closest approach 1 m -> margin = 12 - 1 = 11
    assert layer.stage_margin(0, "L0") == pytest.approx(11.0)
    assert layer.robustly_arrived(0, "L0") is True
    summ = layer.summary(0)
    # the last pose is at altitude 4 m, inside the [1, 8] band -> margin min(3, 4) = 3
    assert summ["rho_safety"] == pytest.approx(3.0)
    assert summ["stages_reached"] >= 1


def test_summary_reports_late_stage():
    layer = StlLayer([make_task()], StlLayerConfig())
    spec = layer.spec(0)
    hi = spec.windows["L0"][1]
    # hover far away for the whole window, then reach L0 well after its deadline.  The pose is
    # unchanged while hovering, so the control step must be passed explicitly.
    for i in range(hi + 6):
        p = (-60.0, 0.0, 0.0) if i == hi + 5 else (-90.0, 0.0, 0.0)
        layer.observe(0, p, step=i)
    st = layer.summary(0)["per_stage"]["L0"]
    assert st["late"] is True and st["inside_window"] is False
