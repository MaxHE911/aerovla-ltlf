"""LTLf task-logic layer (v1) — external monitor/planner/gate over the AeroVLA baseline.

Modules
-------
formula      : formula representation, canonical normalization, progression, acceptance
semantics    : independent finite-trace semantics (cross-check oracle)
automaton    : M1 residual automaton, accepting-path enumeration, swap predicate
monitor      : M4 event detection + residual update from observed positions
planner      : M2 landmark grounding -> phase sequence (v1: fixed order)
tasks        : M0 offline spec construction from episode data
gate         : M5 LAND gating and fallback takeover
offline      : L0 offline replay / independent ESR judge
"""

from .formula import (  # noqa: F401
    AP,
    BOT,
    TOP,
    And,
    F,
    Formula,
    G,
    Not,
    Or,
    U,
    accept,
    atoms,
    mk_and,
    mk_f,
    mk_g,
    mk_not,
    mk_or,
    normalize,
    parse,
    prog,
    prog_trace,
    to_str,
)

__version__ = "1.0.0"
