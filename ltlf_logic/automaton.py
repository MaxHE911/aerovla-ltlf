"""M1: residual automaton of an LTLf formula (Brzozowski-style expansion, no external deps).

Nodes are residual formulas, edges are event *sets* (one sampling point may contain
several events; the empty set is a legal sampling point).  The construction is bounded by
the subformula closure, so state counts stay small for the small task formulas we use.

The automaton is used for
  * diagnostics: how many residual states a spec has (guards against syntactic blow-up);
  * planning (M2): enumerate accepting event sequences from the current residual;
  * analysis: the swap predicate ``Swap(psi, a, b)`` of ``idea1.md`` §2/M2.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .formula import Formula, accept, atoms, normalize, prog, to_str

__all__ = [
    "ResidualAutomaton",
    "build_automaton",
    "enumerate_accepting_paths",
    "swap_equivalent",
    "reaches_acceptance",
]


@dataclass
class ResidualAutomaton:
    """Residual (derivative) automaton of ``phi``."""

    phi: Formula
    atoms: Tuple[str, ...]
    states: List[Formula]
    edges: Dict[Tuple[int, FrozenSet[str]], int]
    accepting: List[bool]
    truncated: bool = False
    max_states: int = 500
    _index: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {s._key(): i for i, s in enumerate(self.states)}

    # -- queries ---------------------------------------------------------------
    def index_of(self, f: Formula) -> Optional[int]:
        return self._index.get(f._key())

    def is_accepting(self, idx: int) -> bool:
        return self.accepting[idx]

    def step(self, idx: int, events: Iterable[str]) -> Optional[int]:
        return self.edges.get((idx, frozenset(events)))

    @property
    def num_states(self) -> int:
        return len(self.states)

    def describe(self) -> str:
        lines = ["residual automaton: %d states, %d atoms, truncated=%s"
                 % (self.num_states, len(self.atoms), self.truncated)]
        for i, s in enumerate(self.states):
            flag = " [accept]" if self.accepting[i] else ""
            lines.append("  s%-2d %s%s" % (i, to_str(s), flag))
        for (i, ev), j in sorted(self.edges.items(), key=lambda kv: (kv[0][0], sorted(kv[0][1]))):
            lines.append("  s%-2d --%-18s--> s%-2d" % (i, ",".join(sorted(ev)) or "{}", j))
        return "\n".join(lines)


def _event_sets(atom_list: Sequence[str], subsets: bool) -> List[FrozenSet[str]]:
    """Candidate event sets for one sampling point (all non-empty subsets by default)."""
    out: List[FrozenSet[str]] = []
    if subsets and len(atom_list) <= 8:
        for r in range(1, len(atom_list) + 1):
            for combo in itertools.combinations(atom_list, r):
                out.append(frozenset(combo))
    else:
        for a in atom_list:
            out.append(frozenset([a]))
    return out


def build_automaton(
    phi: Formula,
    atom_list: Optional[Sequence[str]] = None,
    *,
    max_states: int = 500,
    subsets: bool = True,
) -> ResidualAutomaton:
    """Expand the residual system of ``phi``.

    ``subsets=True`` explores every non-empty subset of atoms as one sampling point, which is
    what the monitor can actually observe (several landmarks can be entered inside one step).
    The empty set is intentionally excluded: it can only be a self-loop for the ``F``-only
    task formulas used here and would not add reachable states.
    """
    phi = normalize(phi)
    atom_list = tuple(sorted(atom_list if atom_list is not None else atoms(phi)))
    ev_sets = _event_sets(atom_list, subsets)

    states: List[Formula] = [phi]
    index: Dict[str, int] = {phi._key(): 0}
    edges: Dict[Tuple[int, FrozenSet[str]], int] = {}
    accepting: List[bool] = [accept(phi)]
    truncated = False

    i = 0
    while i < len(states):
        s = states[i]
        for ev in ev_sets:
            nxt = prog(s, ev)
            k = nxt._key()
            if k in index:
                j = index[k]
            else:
                if len(states) >= max_states:
                    truncated = True
                    continue
                j = len(states)
                index[k] = j
                states.append(nxt)
                accepting.append(accept(nxt))
            edges[(i, ev)] = j
        i += 1

    return ResidualAutomaton(
        phi=phi,
        atoms=atom_list,
        states=states,
        edges=edges,
        accepting=accepting,
        truncated=truncated,
        max_states=max_states,
    )


def enumerate_accepting_paths(
    aut: ResidualAutomaton,
    start: Optional[Formula] = None,
    *,
    max_len: int = 6,
    max_paths: int = 64,
    singleton_only: bool = True,
) -> List[Tuple[FrozenSet[str], ...]]:
    """Shortest event sequences leading from ``start`` (default: the spec itself) to ``accept``.

    Paths are minimal: once a state is accepting the path stops there (a trailing event is
    never needed).  Sequences are returned ordered by length, then lexicographically, which
    makes the chosen plan deterministic.

    ``singleton_only`` (default) restricts every step to a single event: one phase visits one
    landmark, so a phase sequence never needs a simultaneous multi-landmark step.  The
    residual automaton itself is still built over all event subsets (used for state counting
    and for the monitor).
    """
    start_idx = 0 if start is None else aut.index_of(start)
    if start_idx is None:
        raise KeyError("start residual is not part of this automaton: %s" % to_str(start))

    found: List[Tuple[FrozenSet[str], ...]] = []
    # BFS over (state, seq) with a global visited set per state is *not* used on purpose:
    # the same state can be reached by different orders, and the order is what we want.
    frontier: List[Tuple[int, Tuple[FrozenSet[str], ...]]] = [(start_idx, ())]
    seen_sequences = set()
    while frontier:
        next_frontier: List[Tuple[int, Tuple[FrozenSet[str], ...]]] = []
        for idx, seq in frontier:
            for (i, ev), j in aut.edges.items():
                if i != idx or not ev:
                    continue
                if singleton_only and len(ev) != 1:
                    continue
                new_seq = seq + (ev,)
                if len(new_seq) > max_len:
                    continue
                if aut.accepting[j]:
                    if new_seq not in seen_sequences:
                        seen_sequences.add(new_seq)
                        found.append(new_seq)
                    continue
                if new_seq not in seen_sequences:
                    seen_sequences.add(new_seq)
                    next_frontier.append((j, new_seq))
        if len(found) >= max_paths:
            break
        frontier = next_frontier
        if not frontier:
            break

    def sort_key(seq: Tuple[FrozenSet[str], ...]):
        return (len(seq), tuple(tuple(sorted(ev)) for ev in seq))

    found.sort(key=sort_key)
    return found[:max_paths]


def swap_equivalent(phi: Formula, a: str, b: str) -> bool:
    """``Swap(psi, a, b)``: does the order of the two single-event steps matter?

    True means both orders lead to the same residual formula, i.e. the two obligations are
    independent at this point (single-step special case of the STAP independence criterion).
    """
    rab = prog(prog(normalize(phi), frozenset([a])), frozenset([b]))
    rba = prog(prog(normalize(phi), frozenset([b])), frozenset([a]))
    return rab == rba


def reaches_acceptance(phi: Formula, trace: Iterable[Iterable[str]]) -> bool:
    residual = normalize(phi)
    for events in trace:
        residual = prog(residual, frozenset(events))
    return accept(residual)
