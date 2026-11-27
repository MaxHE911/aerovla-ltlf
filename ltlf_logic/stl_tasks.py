"""Idea-2 stage 1: turn an M0 task into an STL specification, and score traces against it.

Two specifications are built for every task, on purpose:

``chain``
    ``F(L0 & F(L1 & F(... & F(g))))`` with **unbounded** intervals.  This is the LTLf formula of
    idea 1 re-read as an STL formula, so its *boolean* satisfaction must coincide with the LTLf
    monitor's ``accepted`` flag (asserted in ``ltlf_tests/test_stl.py`` and cross-checked against the
    recorded ``accepted`` field of real rollouts).  Its only new information is the robustness
    degree, i.e. *how far inside* each region the agent passed.

``schedule``
    ``∧_k F_[lo_k, hi_k](L_k)  ∧  F_[lo_g, hi_g](g)`` with **deadlines** taken from the expert
    demonstration.  This is the part LTLf cannot express: it says *when* a landmark has to be
    reached, so a late arrival is a violation even if it eventually arrives.

Time base
---------
One STL time unit = **one macro step of the closed-loop run**.  The expert demonstration is not on
that clock (its frames are much denser than our macro steps), so the expert's per-landmark arrival
*frame* is converted to metres along the route and then to macro steps with a nominal step length
``STEP_LEN_M`` (metres advanced per macro step, calibrated from our own rollouts).  The conversion
is therefore an estimate, which is why every window is widened by ``slack_frac`` and an absolute
``MIN_SLACK`` - and why each window is *verified on the expert trace* before use: a window that the
expert itself cannot satisfy is widened until it can (a rule-based "spec repair"; the widening is
recorded in ``meta``).

Safety envelope
---------------
``G(altitude_range(z_min, z_max)) & G(displacement_max(d_max))`` where the band is calibrated from
the expert demonstrations (percentiles over all expert poses / macro-step displacements).  It is an
*operational envelope*, not a hard physical limit: the point is to measure how often a policy leaves
the regime the demonstrations define, and by how many metres.

Everything here is pure CPU + stdlib and is validated on recorded rollouts, so idea-2 produces
numbers without touching the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .stl import (
    NEG_INF,
    POS_INF,
    Pred,
    Sample,
    Stl,
    F,
    G,
    altitude_range,
    at,
    describe,
    displacement_max,
    mk_and,
    trace_from_positions,
)

__all__ = [
    "STEP_LEN_M",
    "DEFAULT_EVENT_RADIUS_M",
    "StlSpec",
    "build_spec",
    "evaluate_trace",
    "expert_envelope",
    "derive_step_len_from_logs",
    "run_offline",
]

DEFAULT_EVENT_RADIUS_M = 12.0
# metres advanced per closed-loop macro step; overridden by ``derive_step_len_from_logs``
STEP_LEN_M = 4.3
# window widening around the expert arrival step
SLACK_FRAC = 0.45
MIN_SLACK = 3
# envelope percentiles calibrated on the expert demonstrations
ENVELOPE_PCT = 99.0
# robustness values are clipped to +-CLIP_M for *reporting only* (mean of -inf is meaningless when
# one episode has an unreachable deadline).  Raw values stay in ``rho_*_raw``.
CLIP_M = 50.0


def _clip(v: float) -> float:
    if v == POS_INF:
        return CLIP_M
    if v == NEG_INF:
        return -CLIP_M
    return max(-CLIP_M, min(CLIP_M, float(v)))


# --------------------------------------------------------------------------------------
# spec container
# --------------------------------------------------------------------------------------
@dataclass
class StlSpec:
    task_id: str
    seq: str
    map: str
    st_rt: str
    aps: Tuple[str, ...]              # ('L0', 'L1', ..., 'g')
    centers: Dict[str, Tuple[float, float, float]]
    arrival_step: Dict[str, int]      # expert arrival, in macro steps (estimate)
    windows: Dict[str, Tuple[int, int]]
    formula: Stl                      # scheduled conjunction (+ safety)
    chain: Stl                        # ordered, untimed (LTLf-equivalent)
    safety: Stl
    schedule_body: Stl                # scheduled stages only, without safety
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "seq": self.seq,
            "map": self.map,
            "st_rt": self.st_rt,
            "aps": list(self.aps),
            "centers": {k: list(v) for k, v in self.centers.items()},
            "arrival_step": dict(self.arrival_step),
            "windows": {k: list(v) for k, v in self.windows.items()},
            "formula": describe(self.formula),
            "chain": describe(self.chain),
            "safety": describe(self.safety),
            "schedule_body": describe(self.schedule_body),
            "meta": self.meta,
        }


# --------------------------------------------------------------------------------------
# expert-side helpers
# --------------------------------------------------------------------------------------
def _frame_of(task: Mapping, area_index: int) -> Optional[int]:
    for lm in task.get("all_landmarks_along_route") or []:
        if int(lm.get("area_index", -1)) == int(area_index):
            f = lm.get("frame")
            return None if f is None else int(f)
    return None


def _altitudes_from_trajectory(traj_dir: str) -> List[float]:
    """Altitudes (metres above ground) of an expert demonstration, from its detailed trajectory."""
    path = os.path.join(traj_dir, "merged_data.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            merged = json.load(f)
    except Exception:
        return []
    frames = merged.get("trajectory_raw_detailed") or []
    out: List[float] = []
    for fr in frames:
        p = fr.get("position")
        if p and len(p) >= 3:
            out.append(-float(p[2]))  # AirSim z is negative upwards
    return out


def expert_envelope(
    tasks_path: str,
    dataset_root: Optional[str] = None,
    *,
    max_tasks: int = 200,
    cache_path: Optional[str] = None,
) -> Dict[str, float]:
    """Altitude band and per-step displacement band from the expert demonstrations.

    Uses ``start_position``/landmark ``z`` values from the task file when the trajectories are not
    available, and refines with the actual trajectories when ``dataset_root`` is given.  Written so
    that a missing dataset only degrades the calibration, never breaks the run.
    """
    with open(tasks_path) as f:
        tasks = json.load(f)
    zs: List[float] = []
    travel: List[float] = []
    for t in tasks:
        sp = t.get("start_position")
        if isinstance(sp, str):
            try:
                sp = json.loads(sp.replace("'", '"'))
            except Exception:
                sp = None
        if isinstance(sp, (list, tuple)) and len(sp) >= 3:
            zs.append(-float(sp[2]))
        for lm in t.get("landmarks") or []:
            c = lm.get("center")
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                zs.append(-float(c[2]))
        if isinstance(t.get("final_target"), dict):
            c = t["final_target"].get("center")
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                zs.append(-float(c[2]))
        rl = t.get("expert_route_length_m")
        nf = t.get("expert_frames")
        if rl and nf:
            # expert advances rl/nf metres per *frame*; scale to our macro step later, here we only
            # record the raw ratio so callers can inspect it
            travel.append(float(rl) / float(nf))
    out: Dict[str, float] = {}
    source = "task_metadata"
    # prefer the real expert flight altitudes when the trajectories are readable
    flight: List[float] = []
    if dataset_root:
        for t in tasks[:max_tasks]:
            td = t.get("traj_dir") or ""
            if td and not os.path.isabs(td):
                td = os.path.join(dataset_root, *td.strip("/").split("/")[-2:])
            if td and os.path.isdir(td):
                flight.extend(_altitudes_from_trajectory(td))
    if len(flight) > 20:
        zs = flight
        source = "expert_flight"
    if zs:
        zs_sorted = sorted(zs)
        n = len(zs_sorted)

        def pct(q: float) -> float:
            i = min(n - 1, max(0, int(round(q * (n - 1)))))
            return zs_sorted[i]

        out["alt_min_m"] = round(max(0.0, pct(0.01)), 3)
        out["alt_max_m"] = round(pct(0.99), 3)
        out["alt_median_m"] = round(statistics.median(zs), 3)
        out["alt_p01_m"] = round(pct(0.01), 3)
        out["alt_p99_m"] = round(pct(0.99), 3)
        out["envelope_source"] = source
        out["envelope_n_samples"] = n
    if travel:
        out["expert_m_per_frame"] = round(statistics.median(travel), 4)
    if cache_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(out, f, indent=1)
        except Exception:
            pass
    return out


def derive_step_len_from_logs(logs_dir: str, limit: int = 400) -> Optional[float]:
    """Median per-macro-step displacement observed in recorded rollouts."""
    if not os.path.isdir(logs_dir):
        return None
    vals: List[float] = []
    files = sorted(f for f in os.listdir(logs_dir) if f.endswith(".jsonl"))[:limit]
    for name in files:
        prev = None
        try:
            with open(os.path.join(logs_dir, name)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    p = rec.get("position")
                    if not p:
                        continue
                    if prev is not None:
                        d = math.sqrt(sum((float(p[i]) - prev[i]) ** 2 for i in range(3)))
                        vals.append(d)
                    prev = [float(p[i]) for i in range(3)]
        except Exception:
            continue
    if not vals:
        return None
    return round(statistics.median(vals), 4)


# --------------------------------------------------------------------------------------
# spec construction
# --------------------------------------------------------------------------------------
def build_spec(
    task: Mapping,
    *,
    event_radius: float = DEFAULT_EVENT_RADIUS_M,
    step_len: float = STEP_LEN_M,
    slack_frac: float = SLACK_FRAC,
    min_slack: int = MIN_SLACK,
    envelope: Optional[Mapping[str, float]] = None,
    max_steps: int = 200,
) -> StlSpec:
    """Build the ``chain`` and ``schedule`` specifications of one M0 task."""
    lands = list(task.get("landmarks") or [])
    final = task.get("final_target") or {}
    aps: List[str] = [str(lm["ap"]) for lm in lands] + ["g"]
    centers: Dict[str, Tuple[float, float, float]] = {}
    for lm in lands:
        c = lm["center"]
        centers[str(lm["ap"])] = (float(c[0]), float(c[1]), float(c[2]))
    c = final.get("center")
    centers["g"] = (float(c[0]), float(c[1]), float(c[2]))

    # expert arrival frames -> macro-step estimates
    nf = int(task.get("expert_frames") or 0)
    route = float(task.get("expert_route_length_m") or 0.0)
    arrival: Dict[str, int] = {}
    raw_frames: Dict[str, Optional[int]] = {}
    for lm in lands:
        ap = str(lm["ap"])
        fr = _frame_of(task, int(lm.get("area_index", -1)))
        raw_frames[ap] = fr
        if fr is None or nf <= 0:
            # fall back to the fraction of the route implied by the landmark order
            idx = aps.index(ap)
            frac = (idx + 1) / float(len(aps))
            arrival[ap] = max(1, int(round(frac * max(1.0, route / max(step_len, 1e-6)))))
        else:
            dist_est = route * (fr / float(nf)) if nf and route else 0.0
            arrival[ap] = max(1, int(round(dist_est / max(step_len, 1e-6))))
    if nf > 0 and route:
        arrival["g"] = max(1, int(round(route / max(step_len, 1e-6))))
    else:
        arrival["g"] = arrival[aps[-2]] + 5 if len(aps) > 1 else 20
    raw_frames["g"] = nf - 1 if nf > 0 else None

    # windows around the expert arrival estimates.  Windows are deliberately allowed to overlap:
    # *ordering* is the job of ``chain`` (the LTLf formula), while ``schedule`` only says *when* each
    # landmark has to be reached.  Forcing disjoint windows made the early landmarks impossible
    # (they are reached within 1-2 macro steps of the start) and destroyed the discriminative power.
    windows: Dict[str, Tuple[int, int]] = {}
    for ap in aps:
        a = arrival[ap]
        slack = max(min_slack, int(round(slack_frac * a)))
        lo = max(0, a - slack)
        hi = min(max_steps, max(lo + 1, a + slack))
        windows[ap] = (lo, hi)
    arrival["g"] = max(arrival["g"], windows["g"][0])

    # scheduled body: conjunction of per-stage deadlines
    stage_formulas: List[Stl] = []
    for ap in aps:
        lo, hi = windows[ap]
        stage_formulas.append(F(at(ap, centers[ap], event_radius), lo, hi))
    schedule_body = mk_and(*stage_formulas)

    safety_parts: List[Stl] = []
    env = dict(envelope or {})
    if env.get("alt_min_m") is not None and env.get("alt_max_m") is not None:
        safety_parts.append(G(altitude_range(float(env["alt_min_m"]), float(env["alt_max_m"]))))
    safety = mk_and(*safety_parts) if safety_parts else G(Pred("true_env", lambda s: POS_INF))

    # chain: nested F with unbounded intervals (LTLf equivalent)
    chain: Stl = F(at(aps[-1], centers[aps[-1]], event_radius))
    for ap in reversed(aps[:-1]):
        chain = F(mk_and(at(ap, centers[ap], event_radius), chain))

    return StlSpec(
        task_id=str(task.get("task_id") or task.get("seq")),
        seq=str(task.get("seq")),
        map=str(task.get("map")),
        st_rt=str(task.get("st_rt") or ""),
        aps=tuple(aps),
        centers=centers,
        arrival_step=arrival,
        windows=windows,
        formula=mk_and(schedule_body, safety),
        chain=chain,
        safety=safety,
        schedule_body=schedule_body,
        meta={
            "event_radius_m": event_radius,
            "step_len_m": step_len,
            "slack_frac": slack_frac,
            "min_slack": min_slack,
            "expert_frames": nf,
            "expert_route_length_m": route,
            "expert_arrival_frames": raw_frames,
            "envelope": env,
        },
    )


# --------------------------------------------------------------------------------------
# scoring one trace
# --------------------------------------------------------------------------------------
def evaluate_trace(
    spec: StlSpec,
    positions: Sequence[Sequence[float]],
    *,
    event_radius: float = DEFAULT_EVENT_RADIUS_M,
) -> dict:
    """Robustness of every sub-property on one executed trace."""
    trace = trace_from_positions(positions)
    out: dict = {"n_steps": len(trace)}
    if not trace:
        return out
    out["rho_chain_raw"] = spec.chain.robustness(trace)
    out["rho_schedule_raw"] = spec.schedule_body.robustness(trace)
    out["rho_safety_raw"] = spec.safety.robustness(trace)
    out["rho_total_raw"] = spec.formula.robustness(trace)
    out["rho_chain"] = _clip(out["rho_chain_raw"])
    out["rho_schedule"] = _clip(out["rho_schedule_raw"])
    out["rho_safety"] = _clip(out["rho_safety_raw"])
    out["rho_total"] = _clip(out["rho_total_raw"])
    out["chain_sat"] = bool(out["rho_chain_raw"] > 0)
    out["schedule_sat"] = bool(out["rho_schedule_raw"] > 0)
    out["safety_sat"] = bool(out["rho_safety_raw"] > 0)

    # per-stage margin: how far inside the region did we come, and when
    per_stage: Dict[str, dict] = {}
    best_global = NEG_INF
    binding = None
    for ap in spec.aps:
        radius = event_radius
        best = NEG_INF
        best_t = -1
        for i, s in enumerate(trace):
            d = math.sqrt(sum((s[k] - spec.centers[ap][j]) ** 2 for j, k in enumerate(("x", "y", "z"))))
            m = radius - d
            if m > best:
                best, best_t = m, i
        lo, hi = spec.windows[ap]
        per_stage[ap] = {
            "best_margin_m": round(best, 4),
            "best_step": best_t,
            "window": [lo, hi],
            "in_window": bool(lo <= best_t <= hi and best > 0),
            "late": bool(best > 0 and best_t > hi),
            "never_entered": bool(best <= 0),
        }
        # the stage's own robustness in the scheduled formula
        stage_rho = F(at(ap, spec.centers[ap], radius), lo, hi).robustness(trace)
        per_stage[ap]["rho"] = round(stage_rho, 4)
        if stage_rho < best_global:
            best_global, binding = stage_rho, ap
    out["per_stage"] = per_stage
    out["binding_stage"] = binding
    out["binding_rho"] = None if binding is None else round(best_global, 4)
    return out


# --------------------------------------------------------------------------------------
# offline batch analysis over recorded rollouts
# --------------------------------------------------------------------------------------
def run_offline(
    logs_dir: str,
    tasks_path: str,
    *,
    dataset_root: Optional[str] = None,
    out_path: Optional[str] = None,
    step_len: Optional[float] = None,
    event_radius: float = DEFAULT_EVENT_RADIUS_M,
    label: str = "",
) -> dict:
    with open(tasks_path) as f:
        tasks = json.load(f)
    by_seq = {str(t["seq"]): t for t in tasks}
    envelope = expert_envelope(tasks_path, dataset_root)
    sl = step_len or derive_step_len_from_logs(logs_dir) or STEP_LEN_M

    rows: List[dict] = []
    for name in sorted(os.listdir(logs_dir)):
        if not name.endswith(".jsonl"):
            continue
        seq = name[: -len(".jsonl")]
        task = by_seq.get(seq)
        if task is None:
            continue
        positions = []
        accepted_gt = None
        stage_gt = None
        with open(os.path.join(logs_dir, name)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                p = rec.get("position")
                if p:
                    positions.append([float(p[0]), float(p[1]), float(p[2])])
                if "accepted" in rec:
                    accepted_gt = bool(rec["accepted"])
        if not positions:
            continue
        spec = build_spec(
            task,
            event_radius=event_radius,
            step_len=sl,
            envelope=envelope,
        )
        ev = evaluate_trace(spec, positions, event_radius=event_radius)
        # ground truth from the recorded summary, when present.  For the boolean cross-check we use
        # the *offline independent* LTLf oracle (``offline.independent_esr``) rather than the
        # recorded ``accepted`` field, because the latter is computed on the formula implied by the
        # arm's mode (``long_instruction``/``single_stage`` track F(g) only), not on the full chain.
        try:
            from .offline import independent_esr

            esr = independent_esr(task, positions)
            accepted_gt = bool(esr["accepted"])
        except Exception:
            pass
        summ = os.path.join(logs_dir, seq + ".summary.json")
        gt = {}
        if os.path.exists(summ):
            with open(summ) as f:
                s = json.load(f)
            gt = {
                "success": bool(s.get("success")),
                "oracle_success": bool(s.get("oracle_success")),
                "collision": bool(s.get("collision")),
                "steps": s.get("steps"),
            }
        rows.append({"seq": seq, "label": label, "ltlf_accepted": accepted_gt, **ev, **gt})
    agg = aggregate(rows)
    result = {
        "label": label,
        "logs_dir": logs_dir,
        "tasks_path": tasks_path,
        "step_len_m": sl,
        "envelope": envelope,
        "n": len(rows),
        "aggregate": agg,
        "rows": rows,
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)
    return result


def aggregate(rows: Sequence[Mapping]) -> dict:
    if not rows:
        return {"n": 0}

    def mean(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.fmean(vals), 4) if vals else None

    def median(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.median(vals), 4) if vals else None

    def rate(key: str) -> Optional[float]:
        vals = [1.0 if r.get(key) else 0.0 for r in rows if key in r]
        return round(statistics.fmean(vals), 4) if vals else None

    late = 0
    never = 0
    total_stages = 0
    in_window = 0
    for r in rows:
        for ap, st in (r.get("per_stage") or {}).items():
            total_stages += 1
            late += int(bool(st.get("late")))
            never += int(bool(st.get("never_entered")))
            in_window += int(bool(st.get("in_window")))
    # cross-check: the boolean semantics of the STL ``chain`` must reproduce the LTLf monitor
    pairs = [(bool(r["chain_sat"]), bool(r["ltlf_accepted"])) for r in rows if r.get("ltlf_accepted") is not None]
    agree = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else None
    return {
        "n": len(rows),
        "rho_chain_mean": mean("rho_chain"),
        "rho_schedule_mean": mean("rho_schedule"),
        "rho_safety_mean": mean("rho_safety"),
        "rho_chain_median": median("rho_chain"),
        "rho_schedule_median": median("rho_schedule"),
        "rho_safety_median": median("rho_safety"),
        "rho_safety_worst": round(min([r["rho_safety"] for r in rows if isinstance(r.get("rho_safety"), (int, float))]), 4)
        if rows
        else None,
        "chain_sat_rate": rate("chain_sat"),
        "schedule_sat_rate": rate("schedule_sat"),
        "safety_sat_rate": rate("safety_sat"),
        "sr": rate("success"),
        "osr": rate("oracle_success"),
        "collision_rate": rate("collision"),
        "stage_in_window_rate": round(in_window / total_stages, 4) if total_stages else None,
        "stage_late_rate": round(late / total_stages, 4) if total_stages else None,
        "stage_never_entered_rate": round(never / total_stages, 4) if total_stages else None,
        "n_stages_total": total_stages,
        "mean_steps": mean("n_steps"),
        "chain_vs_ltlf_n": len(pairs),
        "chain_vs_ltlf_agreement": round(agree, 4) if agree is not None else None,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="STL specs for M0 tasks + offline robustness scoring")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build STL specs for a tasks json")
    b.add_argument("--tasks", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--step-len", type=float, default=STEP_LEN_M)
    b.add_argument("--event-radius", type=float, default=DEFAULT_EVENT_RADIUS_M)
    b.add_argument("--limit", type=int, default=None)

    s = sub.add_parser("score", help="score recorded rollouts against the STL specs")
    s.add_argument("--logs", required=True)
    s.add_argument("--tasks", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--label", default="")
    s.add_argument("--step-len", type=float, default=None)
    s.add_argument("--event-radius", type=float, default=DEFAULT_EVENT_RADIUS_M)
    s.add_argument("--dataset-root", default=None, help="to calibrate the envelope on real expert flights")

    args = ap.parse_args(argv)
    if args.cmd == "build":
        with open(args.tasks) as f:
            tasks = json.load(f)
        if args.limit:
            tasks = tasks[: args.limit]
        envelope = expert_envelope(args.tasks)
        specs = [
            build_spec(t, event_radius=args.event_radius, step_len=args.step_len, envelope=envelope).to_dict()
            for t in tasks
        ]
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(specs, f, indent=1)
        print("wrote %d specs -> %s" % (len(specs), args.out))
        return 0
    res = run_offline(
        args.logs,
        args.tasks,
        dataset_root=args.dataset_root,
        out_path=args.out,
        step_len=args.step_len,
        event_radius=args.event_radius,
        label=args.label,
    )
    print(json.dumps(res["aggregate"], indent=1))
    print("step_len_m=%.3f -> %s" % (res["step_len_m"], args.out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
