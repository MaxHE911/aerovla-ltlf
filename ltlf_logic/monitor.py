"""M4: online monitor — observation stream -> events -> residual update.

Rules (``work/idea1.md`` §2/M4, frozen decisions #3 and #4):

* an atomic proposition ``p_r`` is True at a sampling point iff ``||x_t - c_r||_2 < 12 m``
  (3-D Euclidean distance to the landmark centre, simulator pose ground truth);
* every sampling point is progressed exactly once, **including** sampling points with no new
  event (empty event set), otherwise temporal constraints cannot advance;
* a landmark that stays inside its radius triggers only once; leaving and re-entering
  triggers again (``inside_prev`` diffing);
* several landmarks entered at the same sampling point are one sampling point with a
  multi-event set (standard LTLf set-event semantics);
* task completion is ``accept(residual)``; ``residual is BOT`` means permanent violation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .formula import BOT, Formula, accept, normalize, prog, to_str

__all__ = ["Landmark", "Monitor", "DEFAULT_EVENT_RADIUS_M"]

DEFAULT_EVENT_RADIUS_M = 12.0


@dataclass(frozen=True)
class Landmark:
    """A grounded event source: one landmark instance on the map."""

    ap: str
    center: Tuple[float, float, float]
    asset: str = ""
    radius: float = DEFAULT_EVENT_RADIUS_M
    description: str = ""
    area_index: Optional[int] = None

    def distance_to(self, position: Sequence[float]) -> float:
        return math.sqrt(sum((float(position[i]) - float(self.center[i])) ** 2 for i in range(3)))

    def is_inside(self, position: Sequence[float]) -> bool:
        return self.distance_to(position) < self.radius

    def to_dict(self) -> dict:
        return {
            "ap": self.ap,
            "center": [float(x) for x in self.center],
            "asset": self.asset,
            "radius": self.radius,
            "description": self.description,
            "area_index": self.area_index,
        }


def _same_position(a: Sequence[float], b: Sequence[float], tol: float = 1e-6) -> bool:
    return all(abs(float(a[i]) - float(b[i])) < tol for i in range(3))


class Monitor:
    """Tracks the residual formula of one episode while the drone flies."""

    def __init__(
        self,
        phi: Formula,
        landmarks: Iterable[Landmark],
        *,
        dedupe_repeated_positions: bool = False,
        log_positions: bool = True,
    ):
        self.phi = normalize(phi)
        self.landmarks: List[Landmark] = list(landmarks)
        self.residual: Formula = self.phi
        self.inside_prev: FrozenSet[str] = frozenset()
        self.step_index = 0
        self.last_position: Optional[Tuple[float, float, float]] = None
        self.dedupe_repeated_positions = dedupe_repeated_positions
        self.log_positions = log_positions
        self.event_log: List[dict] = []
        self._by_ap: Dict[str, Landmark] = {L.ap: L for L in self.landmarks}

    # -- queries ---------------------------------------------------------------
    @property
    def accepted(self) -> bool:
        return accept(self.residual)

    @property
    def failed(self) -> bool:
        return self.residual is BOT

    @property
    def violated(self) -> bool:
        """True once the residual became unsatisfiable (a G clause was broken)."""
        return self.residual is BOT

    def events_at(self, position: Sequence[float]) -> FrozenSet[str]:
        return frozenset(L.ap for L in self.landmarks if L.is_inside(position))

    # -- progression -----------------------------------------------------------
    def step(self, position: Sequence[float]) -> dict:
        """Consume one sampling point (one control step) and return the log record."""
        pos = tuple(float(x) for x in position[:3])
        if self.dedupe_repeated_positions and self.last_position is not None and _same_position(pos, self.last_position):
            return {
                "step": self.step_index,
                "position": list(pos),
                "duplicate": True,
                "events": sorted(self.events_at(pos)),
                "new_events": [],
                "residual": to_str(self.residual),
                "accept": self.accepted,
                "failed": self.failed,
            }

        events_now = self.events_at(pos)
        new_events = events_now - self.inside_prev
        self.residual = prog(self.residual, new_events)
        self.inside_prev = events_now
        self.last_position = pos
        record = {
            "step": self.step_index,
            "position": list(pos),
            "events": sorted(events_now),
            "new_events": sorted(new_events),
            "residual": to_str(self.residual),
            "accept": self.accepted,
            "failed": self.failed,
        }
        self.step_index += 1
        if self.log_positions:
            self.event_log.append(record)
        return record

    def step_many(self, positions: Iterable[Sequence[float]]) -> List[dict]:
        """Consume several sampling points (e.g. all sub-poses returned by one control step)."""
        return [self.step(p) for p in positions]

    # -- reporting -------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "formula": to_str(self.phi),
            "residual_final": to_str(self.residual),
            "accepted": self.accepted,
            "violated": self.violated,
            "sampling_points": self.step_index,
            "event_count": sum(len(r["new_events"]) for r in self.event_log),
            "events_order": [e for r in self.event_log for e in r["new_events"]],
            "landmarks": [L.to_dict() for L in self.landmarks],
        }
