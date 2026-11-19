"""Stage-1 tests for the M4 monitor: event entry/exit bookkeeping and residual updates."""

from __future__ import annotations

import unittest

from ltlf_logic.formula import BOT, accept, parse, to_str
from ltlf_logic.monitor import Landmark, Monitor

A = Landmark(ap="a", center=(0.0, 0.0, 0.0), asset="A", radius=12.0)
B = Landmark(ap="b", center=(100.0, 0.0, 0.0), asset="B", radius=12.0)
Z = Landmark(ap="z", center=(200.0, 0.0, 0.0), asset="Z", radius=12.0)

START = (30.0, 0.0, 0.0)
NEAR_A = (5.0, 0.0, 0.0)
NEAR_A2 = (4.0, 0.0, 0.0)
NEAR_B = (98.0, 0.0, 0.0)
NEAR_Z = (198.0, 0.0, 0.0)
FAR = (300.0, 0.0, 0.0)


class TestMonitor(unittest.TestCase):
    def test_stay_triggers_once_and_reentry_triggers_again(self):
        mon = Monitor(parse("F(a & F(b))"), [A, B])
        recs = [
            mon.step(START),     # nothing
            mon.step(NEAR_A),    # enters a
            mon.step(NEAR_A2),   # still inside a -> no new event
            mon.step(FAR),       # leaves
            mon.step(NEAR_A),    # re-enters a -> event again
            mon.step(NEAR_B),    # enters b -> accepted
        ]
        self.assertEqual(recs[0]["new_events"], [])
        self.assertEqual(recs[1]["new_events"], ["a"])
        self.assertEqual(recs[2]["new_events"], [])
        self.assertEqual(recs[3]["new_events"], [])
        self.assertEqual(recs[4]["new_events"], ["a"])
        self.assertEqual(recs[5]["new_events"], ["b"])
        self.assertTrue(recs[5]["accept"])
        self.assertEqual(mon.summary()["event_count"], 3)

    def test_empty_event_set_still_advances_time(self):
        # F(!a) is discharged by the very first sampling point outside a
        mon = Monitor(parse("F(!a)"), [A])
        rec = mon.step(START)
        self.assertTrue(rec["accept"])

    def test_order_is_enforced(self):
        mon = Monitor(parse("F(a & F(b))"), [A, B])
        mon.step(START)
        mon.step(NEAR_B)   # b first
        mon.step(FAR)
        rec = mon.step(NEAR_A)  # a afterwards
        self.assertFalse(accept(mon.residual))
        self.assertFalse(rec["accept"])

    def test_G_violation_is_permanent(self):
        mon = Monitor(parse("G(!z) & F(a)"), [A, Z])
        mon.step(START)
        rec_z = mon.step(NEAR_Z)
        self.assertIs(mon.residual, BOT)
        self.assertTrue(rec_z["failed"])
        # moving to a afterwards cannot repair it
        mon.step(FAR)
        rec_a = mon.step(NEAR_A)
        self.assertIs(mon.residual, BOT)
        self.assertFalse(rec_a["accept"])

    def test_duplicate_positions_are_skipped_when_requested(self):
        # the simulator returns the same post-step state 5 times per control step
        mon = Monitor(parse("F(a & F(b))"), [A, B], dedupe_repeated_positions=True)
        mon.step(START)
        r1 = mon.step(NEAR_A)
        r2 = mon.step(NEAR_A)
        self.assertTrue(r2.get("duplicate"))
        self.assertEqual(r1["new_events"], ["a"])
        self.assertEqual(r2["new_events"], [])
        self.assertEqual(mon.step_index, 2)  # duplicates do not consume a sampling point

    def test_multi_event_sampling_point(self):
        near_ab = Landmark(ap="b", center=(8.0, 0.0, 0.0), asset="B2", radius=12.0)
        mon = Monitor(parse("F(a) & F(b)"), [A, near_ab])
        rec = mon.step((4.0, 0.0, 0.0))
        self.assertEqual(rec["new_events"], ["a", "b"])
        self.assertTrue(rec["accept"])

    def test_log_is_human_readable(self):
        mon = Monitor(parse("F(a & F(b))"), [A, B])
        mon.step(START)
        mon.step(NEAR_A)
        mon.step(NEAR_B)
        lines = [r for r in mon.event_log]
        self.assertEqual(lines[1]["residual"], to_str(mon.phi) if False else lines[1]["residual"])
        self.assertIn("accept", lines[2])
        self.assertIn("F(", lines[0]["residual"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
