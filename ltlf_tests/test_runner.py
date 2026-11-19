"""Stage-3/4 tests for the pure logic-layer helpers (no simulator, no baseline imports)."""

from __future__ import annotations

import math
import unittest

from ltlf_logic.formula import accept, parse, prog, to_str
from ltlf_logic.runner import (
    LTLfConfig,
    TaskPlan,
    body_frame_bearing,
    build_stage_instruction,
    is_near_zero_action,
    next_target_atom,
    should_gate_stop,
    takeover_action,
)


def _task(n_stages: int) -> dict:
    landmarks = [
        {
            "ap": "L%d" % i,
            "area_index": 100 + i,
            "center": [float(i * 30), 0.0, 0.0],
            "asset": "asset%d" % i,
            "description": "A real forty word description of landmark number %d that looks like the dataset text." % i,
        }
        for i in range(n_stages - 1)
    ]
    names = ["L%d" % i for i in range(n_stages - 1)] + ["g"]
    formula = "F(g)" if n_stages == 1 else "F(%s)" % " & F(".join(names[:-1] + ["g"]) + ")" * (n_stages - 1)
    inner = "g"
    for name in reversed(names[:-1]):
        inner = "%s & F(%s)" % (name, inner)
    return {
        "task_id": "M/seq",
        "formula": "F(%s)" % inner,
        "formula_single_goal": "F(g)",
        "event_order": names,
        "landmarks": landmarks,
        "n_stages": n_stages,
        "st_rt": "RT",
        "final_target": {
            "ap": "g",
            "area_index": 200,
            "center": [float((n_stages - 1) * 30), 0.0, 0.0],
            "asset": "goal",
            "description": "The real final target description from the episode instruction.",
        },
        "long_instruction_en": "First pass the asset0, then finally reach the goal.",
    }


class TestInstructionInjection(unittest.TestCase):
    """The baseline only reads the text between two anchors - that must be our description."""

    def test_anchors_and_roundtrip(self):
        desc = "A black dog is located on a neatly mowed lawn beside a building with windows and a red car."
        instr = build_stage_instruction(desc)
        # exactly the baseline's extraction (aerovla_wrapper_ui.py:93)
        got = instr.split("degrees from you.")[1].split(" Please control")[0].strip()
        self.assertEqual(got, desc)

    def test_rejects_description_with_anchors(self):
        with self.assertRaises(ValueError):
            build_stage_instruction("bad degrees from you. text")


class TestPlans(unittest.TestCase):
    def test_single_goal_plan_matches_baseline_values(self):
        task = _task(3)
        plan = TaskPlan.from_task_dict(task, mode="single_stage", single_goal=True)
        self.assertEqual(len(plan.landmarks), 1)
        self.assertEqual(plan.landmarks[0].ap, "g")
        self.assertEqual(tuple(plan.landmarks[0].center), tuple(task["final_target"]["center"]))
        self.assertEqual(plan.landmarks[0].description, task["final_target"]["description"])

    def test_long_plan_order(self):
        task = _task(3)
        plan = TaskPlan.from_task_dict(task, mode="fixed_order", single_goal=False)
        self.assertEqual([L.ap for L in plan.landmarks], ["L0", "L1", "g"])
        self.assertEqual(plan.n_stages, 3)


class TestNextTarget(unittest.TestCase):
    def test_fixed_order_progression(self):
        plan = TaskPlan.from_task_dict(_task(3), mode="fixed_order", single_goal=False)
        residual = plan.formula
        self.assertEqual(next_target_atom(residual, plan), "L0")
        residual = prog(residual, frozenset(["L0"]))
        self.assertEqual(next_target_atom(residual, plan), "L1")
        residual = prog(residual, frozenset(["L1"]))
        self.assertEqual(next_target_atom(residual, plan), "g")
        residual = prog(residual, frozenset(["g"]))
        self.assertTrue(accept(residual))
        self.assertIsNone(next_target_atom(residual, plan))

    def test_early_event_is_not_lost(self):
        """Entering L1 before L0 must not skip L0: the residual still asks for L0 first."""
        plan = TaskPlan.from_task_dict(_task(3), mode="fixed_order", single_goal=False)
        residual = prog(plan.formula, frozenset(["L1"]))
        self.assertEqual(next_target_atom(residual, plan), "L0")

    def test_unsatisfiable_gives_none(self):
        plan = TaskPlan.from_task_dict(_task(3), mode="fixed_order", single_goal=False)
        phi = parse("G(!z) & " + plan.formula.__str__())
        residual = prog(phi, frozenset(["z"]))
        self.assertFalse(accept(residual))
        # no accepting path exists from BOT
        self.assertIsNone(next_target_atom(residual, plan))


class TestGate(unittest.TestCase):
    def test_gate_rules(self):
        pending = parse("F(a & F(g))")
        done = prog(prog(pending, frozenset(["a"])), frozenset(["g"]))
        self.assertTrue(accept(done))
        # no stop -> nothing to gate
        self.assertEqual(should_gate_stop(False, pending, True), (False, "no_stop"))
        # gate disabled passes everything through (single-stage / baseline arms)
        self.assertEqual(should_gate_stop(True, pending, False), (True, "gate_disabled"))
        # gate enabled + residual pending -> suppressed
        self.assertEqual(should_gate_stop(True, pending, True), (False, "suppressed_residual_pending"))
        # gate enabled + residual accepted -> forwarded to the baseline success logic
        self.assertEqual(should_gate_stop(True, done, True), (True, "accepted"))


class TestTakeover(unittest.TestCase):
    def test_body_frame_bearing_identity_quaternion(self):
        # identity quaternion: body x == world x, so a target straight ahead has bearing 0
        bearing, dz, dist = body_frame_bearing((0, 0, 0), (0, 0, 0, 1), (10, 0, 0))
        self.assertAlmostEqual(bearing, 0.0, places=6)
        self.assertAlmostEqual(dist, 10.0, places=6)
        self.assertAlmostEqual(dz, 0.0, places=6)
        # target to the right (world +y) -> bearing +90 deg
        bearing, _, _ = body_frame_bearing((0, 0, 0), (0, 0, 0, 1), (0, 10, 0))
        self.assertAlmostEqual(math.degrees(bearing), 90.0, places=4)

    def test_takeover_moves_forward_when_aligned(self):
        cfg = LTLfConfig.for_mode("fixed_order")
        action = takeover_action((0, 0, 0), (0, 0, 0, 1), (10, 0, 0), cfg)
        self.assertGreater(action["fwd"], 0)
        self.assertAlmostEqual(action["yaw"], 0.0, places=6)
        self.assertEqual(action["down"], 0.0)

    def test_takeover_rotates_when_target_behind(self):
        cfg = LTLfConfig.for_mode("fixed_order")
        action = takeover_action((0, 0, 0), (0, 0, 0, 1), (-10, 0, 0), cfg)
        self.assertEqual(action["fwd"], 0.0)
        self.assertEqual(action["down"], 0.0)
        self.assertGreater(abs(action["yaw"]), 1.0)

    def test_takeover_forward_is_capped(self):
        cfg = LTLfConfig.for_mode("fixed_order")
        action = takeover_action((0, 0, 0), (0, 0, 0, 1), (500, 0, 0), cfg)
        self.assertLessEqual(action["fwd"], cfg.takeover_fwd_m)

    def test_near_zero_predicate_matches_baseline_land_rule(self):
        self.assertTrue(is_near_zero_action({"fwd": 0.0, "down": 0.0, "yaw": 0.0}))
        self.assertTrue(is_near_zero_action({"fwd": 0.005, "down": -0.004, "yaw": 0.0}))
        self.assertFalse(is_near_zero_action({"fwd": 2.0, "down": 0.0, "yaw": 0.0}))


class TestModes(unittest.TestCase):
    def test_mode_defaults(self):
        self.assertFalse(LTLfConfig.for_mode("single_stage").gate_enabled)
        self.assertFalse(LTLfConfig.for_mode("single_stage").disable_distance_increasing)
        self.assertTrue(LTLfConfig.for_mode("fixed_order").gate_enabled)
        self.assertTrue(LTLfConfig.for_mode("fixed_order").takeover_enabled)
        self.assertTrue(LTLfConfig.for_mode("fixed_order").disable_distance_increasing)
        self.assertFalse(LTLfConfig.for_mode("long_instruction").gate_enabled)
        self.assertTrue(LTLfConfig.for_mode("long_instruction").disable_distance_increasing)
        with self.assertRaises(ValueError):
            LTLfConfig.for_mode("nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
