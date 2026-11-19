"""Fake-executor test of the phase state machine + LAND handling (no simulator, no torch).

work1.md stage 4 asks for the phase state machine and the LAND gate to be validated with a mock
executor *before* touching the simulator.  The harness below replays a scripted flight through
the same pure components the real loop uses:

    Monitor (event diffing + progression)  ->  next_target_atom (phase from residual)
    ->  should_gate_stop / takeover_action (M5)  ->  assert stage bookkeeping

The remaining gap (the real ``eval_loop`` glue) is covered by the closed-loop smoke runs; this
test pins down the logic that decides "which phase am I in, may I stop, do I need a fallback".
"""

from __future__ import annotations

import math
import unittest

from ltlf_logic.formula import accept, parse, prog, to_str
from ltlf_logic.monitor import Landmark, Monitor
from ltlf_logic.runner import (
    LTLfConfig,
    TaskPlan,
    is_near_zero_action,
    next_target_atom,
    should_gate_stop,
    takeover_action,
)

# three phases: L0 at x=50, L1 at x=120, g at x=200 (radius 12 m)
LM = [
    Landmark(ap="L0", center=(50.0, 0.0, 0.0), asset="lamp", description="d0"),
    Landmark(ap="L1", center=(120.0, 0.0, 0.0), asset="car", description="d1"),
    Landmark(ap="g", center=(200.0, 0.0, 0.0), asset="dog", description="dg"),
]
TASK = {
    "task_id": "fake/seq",
    "formula": "F((L0 & F((L1 & F(g)))))",
    "formula_single_goal": "F(g)",
    "event_order": ["L0", "L1", "g"],
    "landmarks": [
        {"ap": "L0", "area_index": 0, "center": [50.0, 0.0, 0.0], "asset": "lamp", "description": "d0"},
        {"ap": "L1", "area_index": 1, "center": [120.0, 0.0, 0.0], "asset": "car", "description": "d1"},
    ],
    "n_stages": 3,
    "st_rt": "RT",
    "final_target": {"ap": "g", "area_index": 2, "center": [200.0, 0.0, 0.0], "asset": "dog",
                     "description": "dg"},
    "long_instruction_en": "First pass the lamp, then the car, then finally reach the dog.",
}


class FakeLoop:
    """Minimal replica of the phase machine parts of ``LTLfBatchState``."""

    def __init__(self, config: LTLfConfig, task=TASK):
        self.config = config
        self.plan = TaskPlan.from_task_dict(task, mode=config.mode, single_goal=config.mode != "fixed_order")
        self.monitor = Monitor(self.plan.formula, self.plan.landmarks, dedupe_repeated_positions=True)
        self.atoms = sorted({L.ap for L in self.plan.landmarks})
        self.current_atom = None
        self.switches = 0
        self.suppressed = 0
        self.accepted_stops = 0
        self.takeovers = 0
        self.stage_starts = []
        self._refresh(step=0, initial=True)

    # -- injection -------------------------------------------------------------
    def _refresh(self, step, initial=False):
        if self.config.mode != "fixed_order":
            return
        atom = next_target_atom(self.monitor.residual, self.plan, preset_atoms=self.atoms)
        if atom is None:
            return
        if atom == self.current_atom:
            return
        if self.current_atom is not None:
            self.switches += 1
        self.current_atom = atom
        self.stage_starts.append((atom, step))

    @property
    def guide(self):
        if self.config.mode != "fixed_order" or self.current_atom is None:
            return self.plan.landmarks[-1].center
        return self.plan.landmark(self.current_atom).center

    # -- one control step ------------------------------------------------------
    def step(self, position, model_stop, action):
        executed = dict(action)
        zero = is_near_zero_action(action)
        allow, reason = should_gate_stop(model_stop, self.monitor.residual, self.config.gate_enabled)
        if model_stop and allow:
            self.accepted_stops += 1
        if model_stop and not allow:
            self.suppressed += 1
        if self.config.takeover_enabled and not accept(self.monitor.residual) and zero:
            executed = takeover_action(position, (0.0, 0.0, 0.0, 1.0), self.guide, self.config)
            self.takeovers += 1
        self.monitor.step(position)
        self._refresh(step=1)
        return executed, allow, reason


class TestPhaseMachine(unittest.TestCase):
    def test_land_at_intermediate_landmark_is_suppressed_and_phase_advances(self):
        loop = FakeLoop(LTLfConfig.for_mode("fixed_order"))
        self.assertEqual(loop.current_atom, "L0")

        # fly to L0; the model says LAND right after entering it (phase boundary stop)
        loop.step((45.0, 0.0, 0.0), model_stop=False, action={"fwd": 4.0, "down": 0.0, "yaw": 0.0})
        executed, allow, reason = loop.step((50.0, 0.0, 0.0), model_stop=True,
                                            action={"fwd": 0.0, "down": 0.0, "yaw": 0.0})
        # LAND was NOT forwarded: the mission is not finished
        self.assertFalse(allow)
        self.assertEqual(reason, "suppressed_residual_pending")
        self.assertEqual(loop.suppressed, 1)
        # the phase machine advanced to the next landmark
        self.assertEqual(loop.current_atom, "L1")
        self.assertEqual([a for a, _ in loop.stage_starts], ["L0", "L1"])
        # and the near-zero action was replaced by a fallback push toward the new guide point
        self.assertTrue(executed["fwd"] > 0.0)
        self.assertEqual(loop.takeovers, 1)

    def test_full_sequence_accepts_and_land_is_forwarded_at_the_end(self):
        loop = FakeLoop(LTLfConfig.for_mode("fixed_order"))
        for pos in [(40.0, 0, 0), (50.0, 0, 0), (110.0, 0, 0), (120.0, 0, 0), (190.0, 0, 0), (200.0, 0, 0)]:
            loop.step(pos, model_stop=False, action={"fwd": 4.0, "down": 0.0, "yaw": 0.0})
        self.assertTrue(accept(loop.monitor.residual))
        executed, allow, reason = loop.step((200.0, 0, 0), model_stop=True,
                                            action={"fwd": 0.0, "down": 0.0, "yaw": 0.0})
        self.assertTrue(allow)
        self.assertEqual(reason, "accepted")
        self.assertEqual(loop.accepted_stops, 1)
        self.assertEqual(loop.accepted_stops + loop.suppressed, 1)

    def test_disabled_gate_passes_land_through(self):
        cfg = LTLfConfig.for_mode("single_stage")
        loop = FakeLoop(cfg)
        self.assertIsNone(loop.current_atom)
        _exec, allow, reason = loop.step((10.0, 0, 0), model_stop=True,
                                         action={"fwd": 0.0, "down": 0.0, "yaw": 0.0})
        self.assertTrue(allow)
        self.assertEqual(reason, "gate_disabled")
        self.assertEqual(loop.suppressed, 0)
        self.assertEqual(loop.takeovers, 0)  # no takeover in the single-stage arm

    def test_out_of_order_event_does_not_skip_a_phase(self):
        loop = FakeLoop(LTLfConfig.for_mode("fixed_order"))
        # enter L1's radius before L0 was visited
        loop.step((120.0, 0.0, 0.0), model_stop=False, action={"fwd": 4.0, "down": 0.0, "yaw": 0.0})
        self.assertEqual(loop.current_atom, "L0", "entering L1 first must not skip the L0 phase")

    def test_violation_with_G_clause(self):
        task = dict(TASK)
        task["formula"] = "G(!z) & " + task["formula"]
        task["landmarks"] = task["landmarks"] + [
            {"ap": "z", "area_index": 3, "center": [0.0, 60.0, 0.0], "asset": "cone", "description": "dz"}
        ]
        cfg = LTLfConfig.for_mode("fixed_order")
        loop = FakeLoop(cfg, task=task)
        # enter the forbidden area (60 m to the side)
        loop.step((0.0, 60.0, 0.0), model_stop=False, action={"fwd": 4.0, "down": 0.0, "yaw": 0.0})
        self.assertTrue(loop.monitor.violated)
        self.assertIs(loop.monitor.residual, __import__("ltlf_logic.formula", fromlist=["BOT"]).BOT)
        # the gate never forwards a LAND that would count as success
        self.assertFalse(should_gate_stop(True, loop.monitor.residual, True)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
