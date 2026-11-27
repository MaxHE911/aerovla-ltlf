"""STL (signal temporal logic) core: sampled-time semantics with quantitative robustness.

Why sampled time
----------------
The AeroVLA baseline has a *continuous* 3-D action ``(fwd, down, yaw)`` but a *discrete* execution
clock: one macro action produces exactly one new pose (``AirVLNSimulatorClientTool_AeroVLA.py``
repeats the same state sample 5 times) and its wall-clock duration varies with ``fwd``
(``duration = fwd / 1.0`` s plus a 0.5 s stop).  A real-time STL semantics would therefore be a
fiction: there is one observation per macro step and the step length in seconds is not constant.
We consequently interpret every time interval in **macro steps** (integer), which is the honest
sampled-time STL and keeps deadlines directly comparable to the expert demonstration's step count.

Why robustness and not just satisfaction
---------------------------------------
LTLf (``ltlf_logic.formula``/``monitor``) answers a boolean question: did the event happen.  STL
answers *how comfortably* it happened.  All predicates here are **metric margins in metres**
(``mu = threshold - distance``), so the robustness degree ``rho`` is itself in metres and can be
reported and threshold-ed: ``rho > 0`` means satisfied, ``rho = 0.4`` means "4 dm inside the
region", ``rho = -3`` means "3 m outside".  That is the property the LTLf layer cannot express.

Semantics (standard, min/max aggregation)
-----------------------------------------
    rho(ap,  t) = mu_ap(x_t)
    rho(!p,  t) = -rho(p, t)
    rho(p&q, t) = min(rho(p,t), rho(q,t))
    rho(p|q, t) = max(rho(p,t), rho(q,t))
    rho(F_[a,b] p, t) = max_{t' in [t+a, t+b]} rho(p, t')
    rho(G_[a,b] p, t) = min_{t' in [t+a, t+b]} rho(p, t')
    rho(p U_[a,b] q, t) = max_{t' in [t+a,t+b]} min( rho(q,t'),
                                                         min_{t'' in [t, t')} rho(p, t'') )

Missing samples
---------------
Intervals are clamped to the available trace.  If a *required* index lies beyond the end of the
trace, the formula is evaluated under the worst-case completion for the final (``rho``) value, and
reported as "incomplete" by :meth:`Stl.bounds`.  For ``F`` the worst case is ``-inf``, for ``G`` it
is ``+inf``-free (``min`` over what exists), i.e.

* ``F``: value = max over available samples, but if the window is not exhausted the true value may
  be *larger* -> ``bounds()`` returns ``(value, +inf)``.
* ``G``: value = min over available samples; if not exhausted the true value may be *smaller* ->
  ``bounds()`` returns ``(-inf, value)``.

``robustness()`` returns the *lower* bound (pessimistic, "what is already guaranteed"), which is
what a run-time monitor needs; ``robustness_upper()`` returns the optimistic bound used by the
planner/shield to know whether a violation is still avoidable.

Everything in this module is pure stdlib so that it can be unit-tested without the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "NEG_INF",
    "POS_INF",
    "Sample",
    "Stl",
    "Top",
    "Bot",
    "Pred",
    "Not",
    "And",
    "Or",
    "Eventually",
    "Always",
    "Until",
    "mk_and",
    "mk_or",
    "mk_not",
    "F",
    "G",
    "at",
    "altitude_range",
    "displacement_max",
    "trace_from_positions",
    "robustness",
    "bounds",
    "describe",
]

NEG_INF = float("-inf")
POS_INF = float("inf")

# A sample is one observation of the signal set at one macro step.  Only ``x``/``y``/``z`` are
# guaranteed; ``disp`` (metres travelled since the previous step) and ``yaw`` are filled by
# :func:`trace_from_positions` when available.
Sample = Mapping[str, float]


# --------------------------------------------------------------------------------------
# formula AST
# --------------------------------------------------------------------------------------
class Stl:
    """Base class.  All nodes are immutable and hashable (used for dedup in specs)."""

    def robustness(self, trace: Sequence[Sample], t: int = 0) -> float:
        """Robustness under the *complete* trace semantics (what is already guaranteed)."""
        return self._rho(trace, t, optimistic=False)

    def robustness_upper(self, trace: Sequence[Sample], t: int = 0) -> float:
        """Best case still reachable if the trace continues (``+inf`` = not yet decided)."""
        return _online_bounds(self, trace, t)[1]

    def bounds(self, trace: Sequence[Sample], t: int = 0) -> Tuple[float, float]:
        """``(guaranteed, best possible)`` for a possibly incomplete trace."""
        return _online_bounds(self, trace, t)

    def _rho(self, trace: Sequence[Sample], t: int, optimistic: bool) -> float:  # pragma: no cover
        raise NotImplementedError

    def horizon(self) -> Optional[int]:
        """Look-ahead needed in steps; ``None`` means unbounded."""
        return 0

    def atoms(self) -> Tuple[str, ...]:
        return ()

    def satisfied(self, trace: Sequence[Sample], t: int = 0) -> bool:
        return self.robustness(trace, t) > 0.0

    # -- rendering ---------------------------------------------------------------------
    def __str__(self) -> str:  # pragma: no cover - convenience
        return describe(self)

    def __repr__(self) -> str:  # pragma: no cover
        return "Stl(%s)" % describe(self)


@dataclass(frozen=True)
class Top(Stl):
    def _rho(self, trace, t, optimistic):
        return POS_INF

    def horizon(self):
        return 0

    def __str__(self):
        return "true"


@dataclass(frozen=True)
class Bot(Stl):
    def _rho(self, trace, t, optimistic):
        return NEG_INF

    def horizon(self):
        return 0

    def __str__(self):
        return "false"


@dataclass(frozen=True)
class Pred(Stl):
    """Atomic metric predicate: ``margin(sample) > 0`` means satisfied, unit = metres."""

    name: str
    margin: Callable[[Sample], float]
    unit: str = "m"

    def _rho(self, trace, t, optimistic):
        if t < 0 or t >= len(trace):
            return NEG_INF
        return float(self.margin(trace[t]))

    def atoms(self):
        return (self.name,)

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Not(Stl):
    child: Stl

    def _rho(self, trace, t, optimistic):
        inner = self.child._rho(trace, t, optimistic)
        return -inner if not optimistic else -self.child._rho(trace, t, False)

    def horizon(self):
        return self.child.horizon()

    def atoms(self):
        return self.child.atoms()

    def __str__(self):
        return "!(%s)" % describe(self.child)


@dataclass(frozen=True)
class And(Stl):
    parts: Tuple[Stl, ...]

    def _rho(self, trace, t, optimistic):
        vals = [p._rho(trace, t, optimistic) for p in self.parts]
        return min(vals) if vals else POS_INF

    def horizon(self):
        hs = [p.horizon() for p in self.parts]
        if any(h is None for h in hs):
            return None
        return max(hs) if hs else 0

    def atoms(self):
        out: List[str] = []
        for p in self.parts:
            out.extend(p.atoms())
        return tuple(out)

    def __str__(self):
        return "(%s)" % " & ".join(describe(p) for p in self.parts)


@dataclass(frozen=True)
class Or(Stl):
    parts: Tuple[Stl, ...]

    def _rho(self, trace, t, optimistic):
        vals = [p._rho(trace, t, optimistic) for p in self.parts]
        return max(vals) if vals else NEG_INF

    def horizon(self):
        hs = [p.horizon() for p in self.parts]
        if any(h is None for h in hs):
            return None
        return max(hs) if hs else 0

    def atoms(self):
        out: List[str] = []
        for p in self.parts:
            out.extend(p.atoms())
        return tuple(out)

    def __str__(self):
        return "(%s)" % " | ".join(describe(p) for p in self.parts)


@dataclass(frozen=True)
class Eventually(Stl):
    """``F_[lo,hi] child`` - bounded by default (``hi=None`` = unbounded)."""

    child: Stl
    lo: int = 0
    hi: Optional[int] = None

    def _rho(self, trace, t, optimistic):
        n = len(trace)
        start = t + self.lo
        end = n - 1 if self.hi is None else min(n - 1, t + self.hi)
        if start >= n:
            # window entirely in the future: unknown.  pessimistic = not yet satisfied.
            return POS_INF if optimistic and self.hi is None else NEG_INF
        if start < 0:
            start = 0
        best = NEG_INF
        for i in range(start, end + 1):
            v = self.child._rho(trace, i, optimistic)
            if v > best:
                best = v
                if best == POS_INF:
                    break
        return best

    def horizon(self):
        if self.hi is None:
            return None
        h = self.child.horizon()
        return None if h is None else self.hi + h

    def atoms(self):
        return self.child.atoms()

    def __str__(self):
        return "F_[%d,%s](%s)" % (self.lo, "inf" if self.hi is None else self.hi, describe(self.child))


@dataclass(frozen=True)
class Always(Stl):
    """``G_[lo,hi] child``."""

    child: Stl
    lo: int = 0
    hi: Optional[int] = None

    def _rho(self, trace, t, optimistic):
        n = len(trace)
        start = t + self.lo
        end = n - 1 if self.hi is None else min(n - 1, t + self.hi)
        if start >= n:
            return NEG_INF if optimistic and self.hi is None else NEG_INF
        if start < 0:
            start = 0
        worst = POS_INF
        for i in range(start, end + 1):
            v = self.child._rho(trace, i, optimistic)
            if v < worst:
                worst = v
                if worst == NEG_INF:
                    break
        return worst

    def horizon(self):
        if self.hi is None:
            return None
        h = self.child.horizon()
        return None if h is None else self.hi + h

    def atoms(self):
        return self.child.atoms()

    def __str__(self):
        return "G_[%d,%s](%s)" % (self.lo, "inf" if self.hi is None else self.hi, describe(self.child))


@dataclass(frozen=True)
class Until(Stl):
    left: Stl
    right: Stl
    lo: int = 0
    hi: Optional[int] = None

    def _rho(self, trace, t, optimistic):
        n = len(trace)
        start = t + self.lo
        end = n - 1 if self.hi is None else min(n - 1, t + self.hi)
        if start >= n:
            return POS_INF if optimistic and self.hi is None else NEG_INF
        if start < 0:
            start = 0
        best = NEG_INF
        prefix = POS_INF  # min of left over [t, k)
        for k in range(t, end + 1):
            v_right = self.right._rho(trace, k, optimistic)
            cand = min(v_right, prefix)
            if cand > best:
                best = cand
            v_left = self.left._rho(trace, k, optimistic)
            if v_left < prefix:
                prefix = v_left
            if best == POS_INF:
                break
        if optimistic and best == NEG_INF and (self.hi is None or end < t + self.hi):
            return POS_INF
        return best

    def horizon(self):
        if self.hi is None:
            return None
        hs = [x for x in (self.left.horizon(), self.right.horizon()) if x is not None]
        if len(hs) < 2 and (self.left.horizon() is None or self.right.horizon() is None):
            return None
        return self.hi + max(hs)

    def atoms(self):
        return self.left.atoms() + self.right.atoms()

    def __str__(self):
        return "(%s U_[%d,%s] %s)" % (
            describe(self.left),
            self.lo,
            "inf" if self.hi is None else self.hi,
            describe(self.right),
        )


# --------------------------------------------------------------------------------------
# constructors
# --------------------------------------------------------------------------------------
def mk_and(*parts: Stl) -> Stl:
    flat: List[Stl] = []
    for p in parts:
        if isinstance(p, Top):
            continue
        if isinstance(p, And):
            flat.extend(p.parts)
        else:
            flat.append(p)
    if not flat:
        return Top()
    if len(flat) == 1:
        return flat[0]
    # dedup, keep order (formulas are hashable dataclasses; lambdas make them unhashable, so
    # compare by rendered form instead)
    seen: Dict[str, Stl] = {}
    for p in flat:
        seen.setdefault(describe(p), p)
    return And(tuple(seen.values()))


def mk_or(*parts: Stl) -> Stl:
    flat: List[Stl] = []
    for p in parts:
        if isinstance(p, Bot):
            continue
        if isinstance(p, Or):
            flat.extend(p.parts)
        else:
            flat.append(p)
    if not flat:
        return Bot()
    if len(flat) == 1:
        return flat[0]
    return Or(tuple(flat))


def mk_not(p: Stl) -> Stl:
    if isinstance(p, Top):
        return Bot()
    if isinstance(p, Bot):
        return Top()
    return Not(p)


def F(p: Stl, lo: int = 0, hi: Optional[int] = None) -> Eventually:
    return Eventually(p, int(lo), None if hi is None else int(hi))


def G(p: Stl, lo: int = 0, hi: Optional[int] = None) -> Always:
    return Always(p, int(lo), None if hi is None else int(hi))


# --------------------------------------------------------------------------------------
# predicates over the signals we actually have (position + displacement)
# --------------------------------------------------------------------------------------
def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def at(name: str, center: Sequence[float], radius: float) -> Pred:
    """``mu = radius - ||pos - center||``: positive inside the ball, unit metres."""
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))

    def margin(s: Sample) -> float:
        return float(radius) - _dist((s["x"], s["y"], s["z"]), (cx, cy, cz))

    return Pred(name=name, margin=margin)


def altitude_range(z_min: float, z_max: float, name: Optional[str] = None) -> Pred:
    """AirSim ``z`` is negative upwards, so altitude = ``-z``.  ``mu`` = distance to the band."""
    def margin(s: Sample) -> float:
        alt = -float(s["z"])
        return min(alt - float(z_min), float(z_max) - alt)

    return Pred(name=name or ("alt[%g,%g]" % (z_min, z_max)), margin=margin)


def displacement_max(d_max: float, name: Optional[str] = None) -> Pred:
    """Bound on the per-macro-step travelled distance (metres)."""
    def margin(s: Sample) -> float:
        return float(d_max) - float(s.get("disp", 0.0))

    return Pred(name=name or ("disp<=%g" % d_max), margin=margin)


def min_separation_from(points: Iterable[Sequence[float]], d_min: float, prefix: str = "sep") -> Pred:
    """Keep at least ``d_min`` from every given point (a crude obstacle proxy: area centres)."""
    pts = [(float(p[0]), float(p[1]), float(p[2])) for p in points]

    def margin(s: Sample) -> float:
        p = (s["x"], s["y"], s["z"])
        return min(_dist(p, q) for q in pts) - float(d_min) if pts else POS_INF

    return Pred(name="%s>=%g" % (prefix, d_min), margin=margin)


# --------------------------------------------------------------------------------------
# trace helpers
# --------------------------------------------------------------------------------------
def trace_from_positions(positions: Sequence[Sequence[float]], yaws: Optional[Sequence[float]] = None) -> List[Sample]:
    """Build the signal dicts consumed by predicates from a list of 3-D positions."""
    out: List[Sample] = []
    prev: Optional[Sequence[float]] = None
    for i, p in enumerate(positions):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        disp = 0.0 if prev is None else math.sqrt((x - prev[0]) ** 2 + (y - prev[1]) ** 2 + (z - prev[2]) ** 2)
        s: Dict[str, float] = {"x": x, "y": y, "z": z, "disp": disp, "t": float(i)}
        if yaws is not None and i < len(yaws):
            s["yaw"] = float(yaws[i])
        out.append(s)
        prev = (x, y, z)
    return out


# --------------------------------------------------------------------------------------
# online (prefix) bounds
# --------------------------------------------------------------------------------------
def _online_bounds(phi: Stl, trace: Sequence[Sample], t: int = 0) -> Tuple[float, float]:
    """``(guaranteed, best possible)`` when the trace may continue after its last sample.

    * ``F``: the guarantee is what the prefix already achieved; the best case is unbounded until the
      window is exhausted.
    * ``G``: the best case is what the prefix already guarantees to stay above; the guarantee is
      unbounded below until the window is exhausted.
    * ``Until``: falls back to ``(-inf, +inf)`` (not used online; kept total).
    """
    n = len(trace)
    if isinstance(phi, Top):
        return (POS_INF, POS_INF)
    if isinstance(phi, Bot):
        return (NEG_INF, NEG_INF)
    if isinstance(phi, Pred):
        if 0 <= t < n:
            v = float(phi.margin(trace[t]))
            return (v, v)
        return (NEG_INF, POS_INF)
    if isinstance(phi, Not):
        lo, hi = _online_bounds(phi.child, trace, t)
        return (-hi, -lo)
    if isinstance(phi, And):
        bs = [_online_bounds(c, trace, t) for c in phi.parts]
        return (min(b[0] for b in bs), min(b[1] for b in bs)) if bs else (POS_INF, POS_INF)
    if isinstance(phi, Or):
        bs = [_online_bounds(c, trace, t) for c in phi.parts]
        return (max(b[0] for b in bs), max(b[1] for b in bs)) if bs else (NEG_INF, NEG_INF)
    if isinstance(phi, Until):
        return (NEG_INF, POS_INF)
    if isinstance(phi, (Eventually, Always)):
        start = t + phi.lo
        end = n - 1 if phi.hi is None else min(n - 1, t + phi.hi)
        exhausted = phi.hi is not None and end >= t + phi.hi
        if start > n - 1:
            return (NEG_INF, POS_INF) if isinstance(phi, Eventually) else (NEG_INF, POS_INF)
        idx = range(max(0, start), end + 1)
        bs = [_online_bounds(phi.child, trace, i) for i in idx]
        if not bs:
            return (NEG_INF, POS_INF)
        if isinstance(phi, Eventually):
            lo = max(b[0] for b in bs)
            hi = max(b[1] for b in bs) if exhausted else POS_INF
            return (lo, hi)
        hi = min(b[1] for b in bs)
        lo = min(b[0] for b in bs) if exhausted else NEG_INF
        return (lo, hi)
    raise TypeError("not an STL node: %r" % (phi,))


# --------------------------------------------------------------------------------------
# free functions (mirroring the class API, for convenience)
# --------------------------------------------------------------------------------------
def robustness(phi: Stl, trace: Sequence[Sample], t: int = 0) -> float:
    return phi.robustness(trace, t)


def bounds(phi: Stl, trace: Sequence[Sample], t: int = 0) -> Tuple[float, float]:
    return phi.bounds(trace, t)


def describe(phi: Stl) -> str:
    """Rendered form; used for logging and as the dedup key."""
    if isinstance(phi, Top):
        return "true"
    if isinstance(phi, Bot):
        return "false"
    if isinstance(phi, Pred):
        return phi.name
    if isinstance(phi, Not):
        return "!(%s)" % describe(phi.child)
    if isinstance(phi, And):
        return "(%s)" % " & ".join(describe(p) for p in phi.parts)
    if isinstance(phi, Or):
        return "(%s)" % " | ".join(describe(p) for p in phi.parts)
    if isinstance(phi, Eventually):
        return "F_[%d,%s](%s)" % (phi.lo, "inf" if phi.hi is None else phi.hi, describe(phi.child))
    if isinstance(phi, Always):
        return "G_[%d,%s](%s)" % (phi.lo, "inf" if phi.hi is None else phi.hi, describe(phi.child))
    if isinstance(phi, Until):
        return "(%s U_[%d,%s] %s)" % (
            describe(phi.left),
            phi.lo,
            "inf" if phi.hi is None else phi.hi,
            describe(phi.right),
        )
    raise TypeError("not an STL node: %r" % (phi,))
