"""Independent LTLf semantics on finite event traces (used to cross-check progression).

This module is deliberately written *without* progression: it evaluates the formula
directly on a finite sequence of event sets, following the standard LTLf definition.
Tests compare ``accept(prog_trace(phi, trace))`` against ``sat(phi, trace)`` on exhaustive
small traces, so a bug in the residual machinery cannot hide behind itself.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .formula import AP, BOT, TOP, And, F, Formula, G, Not, Or, U

__all__ = ["sat", "sat_prefix"]


def sat(phi: Formula, trace: Sequence[Iterable[str]]) -> bool:
    """True iff ``phi`` holds on the whole (finite) event trace."""
    return sat_prefix(phi, trace, 0)


def sat_prefix(phi: Formula, trace: Sequence[Iterable[str]], i: int) -> bool:
    """True iff ``phi`` holds starting at position ``i`` of ``trace``."""
    n = len(trace)
    sets: List[frozenset] = [frozenset(x) for x in trace]

    def at(f: Formula, j: int) -> bool:
        if f is TOP:
            return True
        if f is BOT:
            return False
        if isinstance(f, AP):
            return j < n and f.name in sets[j]
        if isinstance(f, Not):
            if isinstance(f.operand, AP):
                return not (j < n and f.operand.name in sets[j])
            return not at(f.operand, j)
        if isinstance(f, And):
            return all(at(op, j) for op in f.operands)
        if isinstance(f, Or):
            return any(at(op, j) for op in f.operands)
        if isinstance(f, F):
            return any(at(f.operand, k) for k in range(j, n))
        if isinstance(f, G):
            return all(at(f.operand, k) for k in range(j, n))
        if isinstance(f, U):
            for k in range(j, n):
                if at(f.right, k):
                    if all(at(f.left, m) for m in range(j, k)):
                        return True
            return False
        raise TypeError("unknown formula node: %r" % (f,))

    return at(phi, i)
