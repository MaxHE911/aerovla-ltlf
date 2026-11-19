"""Stage-1 tests: LTLf kernel (representation, normalization, progression, acceptance).

Run from the repository root:
    python -m unittest discover -s ltlf_tests -t . -v
"""

from __future__ import annotations

import itertools
import unittest

from ltlf_logic.automaton import (
    build_automaton,
    enumerate_accepting_paths,
    swap_equivalent,
)
from ltlf_logic.formula import (
    AP,
    BOT,
    TOP,
    And,
    F,
    G,
    Not,
    Or,
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
    subformulas,
    to_str,
)
from ltlf_logic.semantics import sat

A, B, G_AP, Z = AP("a"), AP("b"), AP("g"), AP("z")


def run(phi, trace):
    """Progress through a trace and return the final residual."""
    return prog_trace(phi, trace)


class TestNormalization(unittest.TestCase):
    def test_nary_flatten_sort_dedup(self):
        f = mk_and([A, mk_and([B, G_AP]), A, TOP])
        self.assertIsInstance(f, And)
        self.assertEqual(len(f.operands), 3)
        self.assertEqual(to_str(f), "(a & b & g)")
        # order must not depend on insertion order
        self.assertEqual(to_str(mk_and([G_AP, B, A])), to_str(mk_and([A, B, G_AP])))

    def test_constant_folding(self):
        self.assertIs(mk_and([A, BOT]), BOT)
        self.assertIs(mk_or([A, TOP]), TOP)
        self.assertEqual(mk_and([A, TOP, B]), mk_and([A, B]))
        self.assertEqual(to_str(mk_or([A, BOT])), "a")

    def test_complement(self):
        self.assertIs(mk_and([A, mk_not(A)]), BOT)
        self.assertIs(mk_or([A, mk_not(A)]), TOP)

    def test_absorption(self):
        # a & (a | b) == a
        self.assertEqual(mk_and([A, mk_or([A, B])]), A)
        # a | (a & b) == a
        self.assertEqual(mk_or([A, mk_and([A, B])]), A)

    def test_nnf(self):
        # !(a & F(g)) -> G(!g) | !a   (F/G are duals; operands are sorted by canonical key)
        f = normalize(mk_not(parse("a & F(g)")))
        self.assertEqual(to_str(f), "(G(!g) | !a)")
        # double negation
        self.assertEqual(normalize(mk_not(mk_not(parse("F(a)")))), parse("F(a)"))

    def test_idempotent(self):
        for text in ["F(a & F(g))", "G(!z) & F(a & F(b & F(g)))", "F(a) & F(b)", "!(F(a) | G(b))"]:
            f = parse(text)
            self.assertEqual(normalize(f), f)
            self.assertEqual(normalize(normalize(f)), f)

    def test_G_is_first_class(self):
        # G must survive normalization; it must not be rewritten to a negation of F
        f = parse("G(!z)")
        self.assertIsInstance(f, G)
        self.assertIsInstance(f.operand, Not)
        self.assertIn("G", to_str(f))

    def test_closed_under_progression(self):
        for text in ["F(a & F(g))", "G(!z) & F(a & F(b & F(g)))", "F(a) & F(b) & F(g)"]:
            phi = parse(text)
            for events in [set(), {"a"}, {"a", "b"}, {"z"}, {"g"}]:
                r = prog(phi, events)
                self.assertEqual(normalize(r), r, "progression must stay in normal form: %s" % to_str(r))


class TestProgressionSemantics(unittest.TestCase):
    """The literal acceptance tests required by work1.md stage 1."""

    def test_single_target_F_g(self):
        phi = parse("F(g)")
        self.assertFalse(accept(run(phi, [])))
        self.assertFalse(accept(run(phi, [{"a"}])))
        self.assertFalse(accept(run(phi, [{"a"}, {"b"}])))
        self.assertTrue(accept(run(phi, [{"a"}, {"g"}])))
        # once g happened the obligation is discharged and no event can un-discharge it
        self.assertTrue(accept(run(phi, [{"g"}, {"a"}])))

    def test_two_stage_order(self):
        phi = parse("F(a & F(g))")
        # a -> g accepted
        self.assertTrue(accept(run(phi, [{"a"}, {"g"}])))
        # g -> a NOT accepted (order violated); residual still pending g
        self.assertFalse(accept(run(phi, [{"g"}, {"a"}])))
        # g -> a -> g accepted
        self.assertTrue(accept(run(phi, [{"g"}, {"a"}, {"g"}])))
        # a alone is not enough
        self.assertFalse(accept(run(phi, [{"a"}])))

    def test_three_stage_order(self):
        phi = parse("F(a & F(b & F(g)))")
        self.assertTrue(accept(run(phi, [{"a"}, {"b"}, {"g"}])))
        self.assertFalse(accept(run(phi, [{"a"}, {"g"}, {"b"}])))
        self.assertFalse(accept(run(phi, [{"b"}, {"a"}, {"g"}])))
        self.assertFalse(accept(run(phi, [{"g"}, {"b"}, {"a"}])))
        self.assertFalse(accept(run(phi, [{"a"}, {"b"}])))
        self.assertTrue(accept(run(phi, [{"a"}, {"b"}, {"b"}, {"g"}])))

    def test_unordered_permutations(self):
        phi = parse("F(a) & F(b) & F(g)")
        for perm in itertools.permutations([("a",), ("b",), ("g",)]):
            trace = [{e} for (e,) in perm]
            self.assertTrue(accept(run(phi, trace)), "permutation %s should be accepted" % (perm,))
        self.assertFalse(accept(run(phi, [{"a"}, {"b"}])))

    def test_persistent_constraint_G(self):
        phi = parse("G(!z) & F(a & F(g))")
        # entering z kills the residual permanently
        r = run(phi, [{"z"}])
        self.assertIs(r, BOT)
        self.assertFalse(accept(r))
        # ... and later events cannot repair it
        self.assertIs(run(phi, [{"z"}, {"a"}, {"g"}]), BOT)
        # completing without z is accepted, even though the residual is not syntactically TOP
        r_ok = run(phi, [{"a"}, {"g"}])
        self.assertIsNot(r_ok, TOP)
        self.assertTrue(accept(r_ok))
        self.assertIn("G", to_str(r_ok))
        # z after completion is still a violation
        self.assertIs(run(phi, [{"a"}, {"g"}, {"z"}]), BOT)

    def test_negation_is_not_G(self):
        """Regression for idea1 decision #2: F(a&F g) & !z loses the constraint, G(!z) does not."""
        # `!z` is a *state* formula: it is only evaluated at the current sampling point and is
        # discharged at the next step, so entering z later is silently forgiven.
        bad = parse("F(a & F(g)) & !z")
        self.assertTrue(accept(run(bad, [{"a"}, {"z"}, {"g"}])))
        good = parse("F(a & F(g)) & G(!z)")
        self.assertFalse(accept(run(good, [{"a"}, {"z"}, {"g"}])))
        self.assertIs(run(good, [{"a"}, {"z"}, {"g"}]), BOT)

    def test_empty_event_set_is_a_time_point(self):
        # F(!a) is discharged by one empty sampling point
        phi = parse("F(!a)")
        self.assertFalse(accept(normalize(phi)))
        self.assertTrue(accept(run(phi, [set()])))
        # an empty step must not change an F-obligation progress
        phi2 = parse("F(g)")
        self.assertEqual(run(phi2, [{"a"}, set()]), run(phi2, [{"a"}]))

    def test_multi_event_sampling_point(self):
        # Standard LTLf with set-events: when a and g happen at the SAME sampling point,
        # F(a & F g) is satisfied (F g can hold at that same point).  This is the documented
        # semantics; task construction keeps landmarks far enough apart that one 12 m step
        # cannot touch two landmarks, and any such case is reported by the offline judge.
        phi = parse("F(a & F(g))")
        self.assertTrue(accept(run(phi, [{"a", "g"}])))
        # unordered spec accepts the simultaneous step too
        self.assertTrue(accept(run(parse("F(a) & F(g)"), [{"a", "g"}])))
        # a strict three-stage order CAN be collapsed into one simultaneous step under standard
        # set-event semantics (each nested F can be satisfied at that same point).  Task
        # construction therefore keeps landmark pairs > 2*radius apart, and the offline judge
        # flags any task where two landmarks were entered at the same sampling point.
        phi3 = parse("F(a & F(b & F(g)))")
        self.assertTrue(accept(run(phi3, [{"a", "b", "g"}])))
        self.assertFalse(accept(run(phi3, [{"b", "g"}, {"a"}])))
        # G violation inside a multi-event step is still a violation
        self.assertIs(run(parse("G(!z) & F(a)"), [{"a", "z"}]), BOT)

    def test_progression_matches_independent_semantics(self):
        """Cross-check the residual machinery against the direct finite-trace semantics."""
        alphabet = ["a", "b", "g", "z"]
        formulas = [
            "F(g)",
            "F(a & F(g))",
            "F(a & F(b & F(g)))",
            "F(a) & F(b) & F(g)",
            "G(!z) & F(a & F(g))",
            "F(a | b)",
            "F(a & F(g)) & G(!z)",
            "F(a) & G(!z)",
            "F(a & (b | g))",
            "!(F(a) & G(z))",
        ]
        checked = 0
        for text in formulas:
            phi = parse(text)
            for length in range(0, 4):
                for combo in itertools.product(alphabet, repeat=length):
                    # every sampling point gets exactly the one event for simplicity, plus a
                    # variant where all events are simultaneous
                    trace = [{e} for e in combo]
                    self.assertEqual(
                        accept(run(phi, trace)),
                        sat(phi, trace),
                        "disagreement for %s on %s" % (text, trace),
                    )
                    trace2 = [set(combo)] if combo else []
                    self.assertEqual(
                        accept(run(phi, trace2)),
                        sat(phi, trace2),
                        "disagreement for %s on simultaneous %s" % (text, trace2),
                    )
                    checked += 2
        self.assertGreater(checked, 1000)


class TestResidualAutomaton(unittest.TestCase):
    def state_counts(self):
        return {
            "F(g)": build_automaton(parse("F(g)")).num_states,
            "F(a & F(g))": build_automaton(parse("F(a & F(g))")).num_states,
            "F(a & F(b & F(g)))": build_automaton(parse("F(a & F(b & F(g)))")).num_states,
            "F(a) & F(b) & F(g)": build_automaton(parse("F(a) & F(b) & F(g)")).num_states,
            "G(!z) & F(a & F(g))": build_automaton(parse("G(!z) & F(a & F(g))")).num_states,
        }

    def test_state_counts_are_small_and_reported(self):
        counts = self.state_counts()
        for text, n in counts.items():
            self.assertLessEqual(n, 16, "%s exploded to %d residual states" % (text, n))
        self.assertEqual(counts["F(a) & F(b) & F(g)"], 8, "unordered 3-goal spec must give 2**3 states")
        # print for the implementation log
        print("\n[M1] residual state counts:", counts)

    def test_no_syntactic_blowup_under_closure(self):
        phi = parse("F(a & F(b & F(g)))")
        closure = len(subformulas(phi))
        aut = build_automaton(phi)
        self.assertFalse(aut.truncated)
        self.assertLessEqual(aut.num_states, 4 * closure)

    def test_accepting_paths(self):
        phi = parse("F(a & F(b & F(g)))")
        aut = build_automaton(phi)
        paths = enumerate_accepting_paths(aut, singleton_only=True)
        self.assertTrue(paths)
        shortest = [tuple(next(iter(ev)) for ev in p) for p in paths if len(p) == len(paths[0])]
        # the fixed-order spec admits exactly ONE minimal single-event phase order
        self.assertEqual(shortest, [("a", "b", "g")])

    def test_unordered_paths_count(self):
        phi = parse("F(a) & F(b) & F(g)")
        aut = build_automaton(phi)
        paths = enumerate_accepting_paths(aut, singleton_only=True)
        shortest = {tuple(next(iter(ev)) for ev in p) for p in paths if len(p) == len(paths[0])}
        self.assertEqual(len(shortest), 6, "three unordered goals must admit 6 orders")

    def test_swap_predicate(self):
        # ordered spec at the start: swapping a and g matters
        self.assertFalse(swap_equivalent(parse("F(a & F(g))"), "a", "g"))
        # unordered spec: swapping does not matter
        self.assertTrue(swap_equivalent(parse("F(a) & F(g)"), "a", "g"))
        # three-stage spec *after* a has been visited: b and g are still ordered
        phi = parse("F(a & F(b & F(g)))")
        after_a = prog(normalize(phi), frozenset(["a"]))
        self.assertFalse(swap_equivalent(after_a, "b", "g"))
        # before a is visited, entering b or g early is ignored either way -> equivalent
        self.assertTrue(swap_equivalent(phi, "b", "g"))

    def test_atoms(self):
        self.assertEqual(atoms(parse("G(!z) & F(a & F(g))")), frozenset({"a", "g", "z"}))


class TestParser(unittest.TestCase):
    def test_roundtrip(self):
        for text in ["F(a & F(g))", "G(!z) & F(a & F(b & F(g)))", "(a & b) | g", "F(top)"]:
            self.assertEqual(normalize(parse(to_str(parse(text)))), parse(text))

    def test_bad_syntax(self):
        for bad in ["F(", "&", "a &", "U(a,b)"]:
            with self.assertRaises(ValueError):
                parse(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
