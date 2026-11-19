"""LTLf core: formula representation, canonical normalization, progression, empty-trace acceptance.

Design constraints (frozen in ``work/idea1.md`` §2 / M1):

1. ``And`` / ``Or`` are **n-ary, flattened, de-duplicated and sorted**.  Nesting binary
   conjunctions instead leads to unbounded syntactic growth of residuals.
2. ``G`` is a **first-class** operator with its own progression rule.  It must never be
   encoded as a plain negation: with ``F(a & F g) & !z`` the constraint on ``z``
   silently disappears after ``z`` is entered, with ``G(!z)`` it stays violated forever.
3. Task completion is decided by :func:`accept` (satisfaction on the *empty* trace,
   i.e. "is the remaining requirement void now?"), **not** by ``residual == TOP``:
   an ``G`` formula that is still pending is not ``TOP`` but is already satisfied.

Semantics used here is standard LTLf over *event traces*: one time point = one sampling
point (a control step), and the atomic proposition ``p`` holds at that point iff the
corresponding event happened there.  Progression ``prog(phi, events)`` consumes one time
point and returns the residual formula that must hold on the remaining suffix.

All formulas produced by the ``mk_*`` constructors are in normal form:
negation is pushed down to atomic propositions (NNF), ``And``/``Or`` are canonical.
:func:`normalize` converts an arbitrary tree into normal form; the other public
functions assume normal-form input.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Iterator, List, Sequence, Tuple

__all__ = [
    "Formula",
    "Top",
    "Bot",
    "AP",
    "Not",
    "And",
    "Or",
    "F",
    "G",
    "U",
    "TOP",
    "BOT",
    "mk_and",
    "mk_or",
    "mk_not",
    "mk_f",
    "mk_g",
    "mk_u",
    "normalize",
    "prog",
    "prog_trace",
    "accept",
    "atoms",
    "subformulas",
    "to_str",
    "parse",
]


# --------------------------------------------------------------------------------------
# formula nodes
# --------------------------------------------------------------------------------------
class Formula:
    """Base class.  Equality/hash are structural (via :meth:`_key`)."""

    __slots__ = ()

    def _key(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        return isinstance(other, Formula) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __str__(self) -> str:
        return to_str(self)

    __repr__ = __str__


class _True(Formula):
    __slots__ = ()

    def _key(self) -> str:
        return "TOP"

    def __str__(self) -> str:
        return "TOP"


class _False(Formula):
    __slots__ = ()

    def _key(self) -> str:
        return "BOT"

    def __str__(self) -> str:
        return "BOT"


class AP(Formula):
    """Atomic proposition: an event label (not a physical object)."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError("AP name must be a non-empty string")
        self.name = name

    def _key(self) -> str:
        return "ap:" + self.name


class Not(Formula):
    """Negation.  In normal form the operand is always an :class:`AP`."""

    __slots__ = ("operand",)

    def __init__(self, operand: Formula):
        self.operand = operand

    def _key(self) -> str:
        return "not(" + self.operand._key() + ")"


class And(Formula):
    """n-ary conjunction, canonical: flattened, de-duplicated, sorted, len >= 2."""

    __slots__ = ("operands",)

    def __init__(self, operands: Sequence[Formula]):
        ops = tuple(operands)
        if len(ops) < 2:
            raise ValueError("And needs >= 2 operands in normal form")
        self.operands = ops

    def _key(self) -> str:
        return "and(" + ",".join(op._key() for op in self.operands) + ")"


class Or(Formula):
    """n-ary disjunction, canonical: flattened, de-duplicated, sorted, len >= 2."""

    __slots__ = ("operands",)

    def __init__(self, operands: Sequence[Formula]):
        ops = tuple(operands)
        if len(ops) < 2:
            raise ValueError("Or needs >= 2 operands in normal form")
        self.operands = ops

    def _key(self) -> str:
        return "or(" + ",".join(op._key() for op in self.operands) + ")"


class F(Formula):
    """Eventually."""

    __slots__ = ("operand",)

    def __init__(self, operand: Formula):
        self.operand = operand

    def _key(self) -> str:
        return "F(" + self.operand._key() + ")"


class G(Formula):
    """Globally (always)."""

    __slots__ = ("operand",)

    def __init__(self, operand: Formula):
        self.operand = operand

    def _key(self) -> str:
        return "G(" + self.operand._key() + ")"


class U(Formula):
    """Until (only needed if a spec asks for it; negation of U is unsupported in v1)."""

    __slots__ = ("left", "right")

    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right

    def _key(self) -> str:
        return "U(" + self.left._key() + "," + self.right._key() + ")"


TOP: Formula = _True()
BOT: Formula = _False()


# --------------------------------------------------------------------------------------
# normal form constructors
# --------------------------------------------------------------------------------------
def _is_literal(f: Formula) -> bool:
    return isinstance(f, (AP, Not))


def _mk_literal_order_key(f: Formula) -> str:
    return f._key()


def mk_and(operands: Iterable[Formula]) -> Formula:
    """Build a canonical n-ary conjunction.  Inputs are already in normal form."""
    flat: List[Formula] = []
    for f in operands:
        if f is BOT:
            return BOT
        if f is TOP:
            continue
        if isinstance(f, And):
            flat.extend(f.operands)
        else:
            flat.append(f)

    # de-duplicate while keeping structure
    seen = set()
    uniq: List[Formula] = []
    for f in flat:
        k = f._key()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(f)

    # literal contradiction
    atom_pos = {f.name for f in uniq if isinstance(f, AP)}
    atom_neg = {f.operand.name for f in uniq if isinstance(f, Not) and isinstance(f.operand, AP)}
    if atom_pos & atom_neg:
        return BOT

    # absorption: drop ``A | ...`` if a conjunct ``A`` is already required
    keys = {f._key() for f in uniq}
    reduced: List[Formula] = []
    for f in uniq:
        if isinstance(f, Or) and any(op._key() in keys for op in f.operands):
            continue
        reduced.append(f)
    uniq = reduced

    if not uniq:
        return TOP
    if len(uniq) == 1:
        return uniq[0]
    uniq.sort(key=_mk_literal_order_key)
    return And(tuple(uniq))


def mk_or(operands: Iterable[Formula]) -> Formula:
    """Build a canonical n-ary disjunction.  Inputs are already in normal form."""
    flat: List[Formula] = []
    for f in operands:
        if f is TOP:
            return TOP
        if f is BOT:
            continue
        if isinstance(f, Or):
            flat.extend(f.operands)
        else:
            flat.append(f)

    seen = set()
    uniq: List[Formula] = []
    for f in flat:
        k = f._key()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(f)

    atom_pos = {f.name for f in uniq if isinstance(f, AP)}
    atom_neg = {f.operand.name for f in uniq if isinstance(f, Not) and isinstance(f.operand, AP)}
    if atom_pos & atom_neg:
        return TOP

    keys = {f._key() for f in uniq}
    reduced: List[Formula] = []
    for f in uniq:
        if isinstance(f, And) and any(op._key() in keys for op in f.operands):
            continue
        reduced.append(f)
    uniq = reduced

    if not uniq:
        return BOT
    if len(uniq) == 1:
        return uniq[0]
    uniq.sort(key=_mk_literal_order_key)
    return Or(tuple(uniq))


def mk_not(f: Formula) -> Formula:
    """Negation, immediately pushed to normal form."""
    if f is TOP:
        return BOT
    if f is BOT:
        return TOP
    if isinstance(f, Not):
        return f.operand
    if isinstance(f, AP):
        return Not(f)
    return _nnf(f, True)


def mk_f(f: Formula) -> Formula:
    if f is TOP:
        return TOP
    if f is BOT:
        return BOT
    if isinstance(f, F):
        return f
    return F(f)


def mk_g(f: Formula) -> Formula:
    if f is TOP:
        return TOP
    # NOTE: G(BOT) is *not* BOT: it is satisfied by the empty trace (nothing left to visit),
    # so folding it to BOT would break ``accept``.
    if isinstance(f, G):
        return f
    return G(f)


def mk_u(left: Formula, right: Formula) -> Formula:
    if right is TOP:
        return TOP
    if right is BOT:
        return BOT
    if left is BOT:
        return right
    if left is TOP:
        return mk_f(right)
    return U(left, right)


# --------------------------------------------------------------------------------------
# normalization of arbitrary trees (NNF + canonical n-ary)
# --------------------------------------------------------------------------------------
def _nnf(f: Formula, neg: bool) -> Formula:
    if f is TOP:
        return BOT if neg else TOP
    if f is BOT:
        return TOP if neg else BOT
    if isinstance(f, AP):
        return Not(f) if neg else f
    if isinstance(f, Not):
        return _nnf(f.operand, not neg)
    if isinstance(f, And):
        children = [_nnf(op, neg) for op in f.operands]
        return mk_or(children) if neg else mk_and(children)
    if isinstance(f, Or):
        children = [_nnf(op, neg) for op in f.operands]
        return mk_and(children) if neg else mk_or(children)
    if isinstance(f, F):
        return mk_g(_nnf(f.operand, True)) if neg else mk_f(_nnf(f.operand, False))
    if isinstance(f, G):
        return mk_f(_nnf(f.operand, True)) if neg else mk_g(_nnf(f.operand, False))
    if isinstance(f, U):
        if neg:
            raise NotImplementedError(
                "negation of U is not supported in v1 (it would require the Release operator)"
            )
        return mk_u(_nnf(f.left, False), _nnf(f.right, False))
    raise TypeError("unknown formula node: %r" % (f,))


def normalize(f: Formula) -> Formula:
    """Return ``f`` in normal form (negation normal form + canonical n-ary connectives)."""
    return _nnf(f, False)


# --------------------------------------------------------------------------------------
# progression, acceptance, utilities
# --------------------------------------------------------------------------------------
def prog(f: Formula, events: FrozenSet[str] | Iterable[str]) -> Formula:
    """Progress normal-form formula ``f`` with the events observed at one sampling point.

    ``events`` is the set of atomic propositions that hold at that point (possibly empty).
    """
    if not isinstance(events, (set, frozenset)):
        events = frozenset(events)

    if f is TOP or f is BOT:
        return f
    if isinstance(f, AP):
        return TOP if f.name in events else BOT
    if isinstance(f, Not):
        if not isinstance(f.operand, AP):  # pragma: no cover - normal form guarantees AP
            raise ValueError("Not must wrap an AP in normal form; call normalize() first")
        return BOT if f.operand.name in events else TOP
    if isinstance(f, And):
        return mk_and([prog(op, events) for op in f.operands])
    if isinstance(f, Or):
        return mk_or([prog(op, events) for op in f.operands])
    if isinstance(f, F):
        return mk_or([prog(f.operand, events), f])
    if isinstance(f, G):
        # G x == x & X G x  ->  prog = prog(x) & G x ; if prog(x) is BOT the whole thing is BOT.
        return mk_and([prog(f.operand, events), f])
    if isinstance(f, U):
        return mk_or([prog(f.right, events), mk_and([prog(f.left, events), f])])
    raise TypeError("unknown formula node: %r" % (f,))


def prog_trace(f: Formula, trace: Iterable[Iterable[str]]) -> Formula:
    """Progress through a whole event trace (one iterable of event names per time point)."""
    residual = normalize(f)
    for events in trace:
        residual = prog(residual, frozenset(events))
    return residual


def accept(f: Formula) -> bool:
    """Satisfaction on the *empty* trace (the "weak truth" function t(.) of LTLf).

    ``TOP``/``G x``  -> True   (vacuous satisfaction)
    ``BOT``/``F x``/``U``/``AP`` -> False
    ``Not(AP)``      -> True   ("the event never happened on the empty suffix")
    ``And``/``Or``   -> recurse
    """
    if f is TOP:
        return True
    if f is BOT:
        return False
    if isinstance(f, AP):
        return False
    if isinstance(f, Not):
        if isinstance(f.operand, AP):
            return True
        return not accept(f.operand)  # defensive; normal form should not reach this
    if isinstance(f, And):
        return all(accept(op) for op in f.operands)
    if isinstance(f, Or):
        return any(accept(op) for op in f.operands)
    if isinstance(f, F):
        return False
    if isinstance(f, G):
        return True
    if isinstance(f, U):
        return False
    raise TypeError("unknown formula node: %r" % (f,))


def atoms(f: Formula) -> FrozenSet[str]:
    """All atomic proposition names occurring in ``f``."""
    out: set[str] = set()

    def walk(x: Formula) -> None:
        if isinstance(x, AP):
            out.add(x.name)
        elif isinstance(x, Not):
            walk(x.operand)
        elif isinstance(x, (And, Or)):
            for op in x.operands:
                walk(op)
        elif isinstance(x, (F, G)):
            walk(x.operand)
        elif isinstance(x, U):
            walk(x.left)
            walk(x.right)

    walk(f)
    return frozenset(out)


def subformulas(f: Formula) -> List[Formula]:
    """All distinct subformulas of ``f`` (used for closure-size diagnostics)."""
    seen: dict[str, Formula] = {}

    def walk(x: Formula) -> None:
        k = x._key()
        if k in seen:
            return
        seen[k] = x
        if isinstance(x, Not):
            walk(x.operand)
        elif isinstance(x, (And, Or)):
            for op in x.operands:
                walk(op)
        elif isinstance(x, (F, G)):
            walk(x.operand)
        elif isinstance(x, U):
            walk(x.left)
            walk(x.right)

    walk(f)
    return list(seen.values())


def to_str(f: Formula) -> str:
    """Human-readable infix rendering (also used in logs)."""
    if f is TOP:
        return "top"
    if f is BOT:
        return "bot"
    if isinstance(f, AP):
        return f.name
    if isinstance(f, Not):
        return "!" + to_str(f.operand)
    if isinstance(f, And):
        return "(" + " & ".join(to_str(op) for op in f.operands) + ")"
    if isinstance(f, Or):
        return "(" + " | ".join(to_str(op) for op in f.operands) + ")"
    if isinstance(f, F):
        return "F(%s)" % to_str(f.operand)
    if isinstance(f, G):
        return "G(%s)" % to_str(f.operand)
    if isinstance(f, U):
        return "(%s U %s)" % (to_str(f.left), to_str(f.right))
    raise TypeError("unknown formula node: %r" % (f,))


# --------------------------------------------------------------------------------------
# tiny spec DSL (this is *not* an NL->LTLf translator: it only reads the LTLf syntax itself)
# --------------------------------------------------------------------------------------
_TOKEN_HINT = "supported syntax: atoms (letters/digits/_:), !, &, |, F(..), G(..), U(..), top, bot, parentheses"


def parse(text: str) -> Formula:
    """Parse an LTLf formula written in the small spec DSL.

    Examples
    --------
    >>> to_str(parse("F(a & F(g))"))
    'F((a & F(g)))'
    >>> to_str(parse("G(!z) & F(a & F(g))"))
    '((F((a & F(g)))) & G(!z))'
    """
    tokens = _tokenize(text)
    node, pos = _parse_binary(tokens, 0)
    if pos != len(tokens):
        raise ValueError("unexpected token %r; %s" % (tokens[pos], _TOKEN_HINT))
    return normalize(node)


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
        elif ch in "()!&|,":
            tokens.append(ch)
            i += 1
        else:
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] in "_:"):
                j += 1
            if j == i:
                raise ValueError("bad character %r in spec; %s" % (ch, _TOKEN_HINT))
            tokens.append(text[i:j])
            i = j
    return tokens


def _parse_binary(tokens: List[str], pos: int) -> Tuple[Formula, int]:
    node, pos = _parse_unary(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ("&", "|"):
        op = tokens[pos]
        rhs, pos = _parse_unary(tokens, pos + 1)
        node = mk_and([node, rhs]) if op == "&" else mk_or([node, rhs])
    return node, pos


def _parse_unary(tokens: List[str], pos: int) -> Tuple[Formula, int]:
    if pos >= len(tokens):
        raise ValueError("unexpected end of spec")
    tok = tokens[pos]
    if tok == "!":
        node, pos = _parse_unary(tokens, pos + 1)
        return mk_not(node), pos
    if tok in ("F", "G"):
        if pos + 1 >= len(tokens) or tokens[pos + 1] != "(":
            raise ValueError("%s must be applied as %s(...); %s" % (tok, tok, _TOKEN_HINT))
        node, pos = _parse_binary(tokens, pos + 2)
        if pos >= len(tokens) or tokens[pos] != ")":
            raise ValueError("missing ')' after %s(...)" % tok)
        return (mk_f(node) if tok == "F" else mk_g(node)), pos + 1
    if tok == "U":
        raise ValueError("U is written infix as (a U b) via U(...) only in tests; %s" % _TOKEN_HINT)
    if tok == "(":
        node, pos = _parse_binary(tokens, pos + 1)
        if pos >= len(tokens) or tokens[pos] != ")":
            raise ValueError("missing ')'")
        return node, pos + 1
    if tok in ("&", "|", ")", ","):
        raise ValueError("unexpected operator %r; %s" % (tok, _TOKEN_HINT))
    if tok == "top":
        return TOP, pos + 1
    if tok == "bot":
        return BOT, pos + 1
    return AP(tok), pos + 1
