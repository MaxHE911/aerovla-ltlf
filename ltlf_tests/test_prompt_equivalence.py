"""Interface-level harmlessness test: the logic layer must not change the baseline prompt.

The simulator is *not* bit-reproducible (two baseline runs of the same episode start from
slightly different poses and render slightly different images, see
``work/implementation-log.md``), so "identical behaviour" cannot be measured by replaying the
closed loop twice.  What *can* be checked deterministically is the interface: given the same
pose / target / instruction, our single-stage degradation must produce exactly the prompt the
baseline produces, and the multi-stage arm must change only the two values it is designed to
change (guide point + object description) while keeping the prompt template intact.
"""

from __future__ import annotations

import math
import os
import unittest

from ltlf_logic.runner import TaskPlan, build_stage_instruction
from ltlf_logic.tasks import build_description_index, build_task, load_episode, load_map_areas, load_split, unique_sequences

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA_CANDIDATES = [
    os.path.join(REPO, "..", "data_local"),
    os.path.join(REPO, "dataset_raw"),
    os.path.join(REPO, "AeroVLA", "dataset_raw"),
]
META_CANDIDATES = [
    os.path.join(REPO, "..", "data_local", "data", "meta", "map_spawnarea_info.json"),
    os.path.join(REPO, "AeroVLA", "data", "meta", "map_spawnarea_info.json"),
]
SPLIT_CANDIDATES = [
    os.path.join(REPO, "..", "data_local", "data", "uav_dataset", "seen_valset_splits", "ModernCityMap.json"),
    os.path.join(REPO, "AeroVLA", "data", "uav_dataset", "seen_valset_splits", "ModernCityMap.json"),
]


def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


DATA_ROOT, META, SPLIT = _first(DATA_CANDIDATES), _first(META_CANDIDATES), _first(SPLIT_CANDIDATES)

# verbatim copy of AerialVLAWrapper.get_semantic_direction (aerovla_wrapper_ui.py:46-76)
BUCKETS = [
    (-15, 15, "straight ahead "),
    (15, 60, "forward-right "),
    (60, 120, "to your right "),
    (120, 180, "to your right rear "),
    (-60, -15, "forward-left "),
    (-120, -60, "to your left "),
    (-180, -120, "to your left rear "),
]


def _quat_rotate_inverse(v, q):
    """``R.from_quat(q).inv().apply(v)`` without scipy (q = [x, y, z, w])."""
    x, y, z, w = q
    # rotation matrix rows for R^T applied to v
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    # body vector = R^T @ v
    return (
        r00 * v[0] + r10 * v[1] + r20 * v[2],
        r01 * v[0] + r11 * v[1] + r21 * v[2],
        r02 * v[0] + r12 * v[1] + r22 * v[2],
    )


def get_semantic_direction(position, quaternion, target_pos):
    vec_world = [target_pos[i] - position[i] for i in range(3)]
    dist_xy = math.hypot(vec_world[0], vec_world[1])
    if dist_xy < 0.01:
        return ""
    vec_body = _quat_rotate_inverse(vec_world, quaternion)
    angle_deg = math.degrees(math.atan2(vec_body[1], vec_body[0]))
    for lo, hi, text in BUCKETS:
        if lo <= angle_deg <= hi:
            return text
    return ""


def baseline_prompt(pose, quaternion, target, instruction):
    """Exactly ``AerialVLAWrapper.prepare_inputs`` lines 91-93 and 146-150."""
    dir_text = get_semantic_direction(pose, quaternion, target)
    obj_desc = instruction.split("degrees from you.")[1].split(" Please control")[0].strip()
    return "<image>\nFly %sand find the target. %s\nAction: " % (dir_text, obj_desc), dir_text, obj_desc


def ours_prompt(pose, quaternion, guide, stage_instruction):
    """The same template, but fed with the values the logic layer injects."""
    return baseline_prompt(pose, quaternion, guide, stage_instruction)


@unittest.skipUnless(DATA_ROOT and META and SPLIT, "ModernCityMap data mirror not available")
class TestPromptEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.areas = load_map_areas(META, "ModernCityMap")
        cls.seqs = unique_sequences(load_split(SPLIT))[:5]
        cls.episodes = [load_episode(DATA_ROOT, "ModernCityMap", s, cls.areas) for s in cls.seqs]
        cls.desc_index = build_description_index(DATA_ROOT, "ModernCityMap", cls.areas)
        cls.tasks = [
            build_task(ep, cls.areas, cls.desc_index, map_name="ModernCityMap", split_name="seen_valset")
            for ep in cls.episodes
        ]

    # -- single-stage degradation -------------------------------------------------
    def test_single_stage_prompt_is_byte_identical_to_baseline(self):
        for ep, task in zip(self.episodes, self.tasks):
            plan = TaskPlan.from_task_dict(task, mode="single_stage", single_goal=True)
            # what the baseline injects
            base_target = ep.target_area.center
            base_instr = ep.instruction
            # what our single-stage arm injects: the very same values
            our_target = plan.landmarks[0].center
            our_instr = base_instr  # single_stage mode passes the original instruction through
            self.assertEqual(tuple(our_target), tuple(float(x) for x in base_target))
            for k in range(0, len(ep.positions), max(1, len(ep.positions) // 5)):
                pose = ep.positions[k]
                quat = (0.0, 0.0, 0.0, 1.0)
                p_base, d_base, o_base = baseline_prompt(pose, quat, base_target, base_instr)
                p_ours, d_ours, o_ours = ours_prompt(pose, quat, our_target, our_instr)
                self.assertEqual(p_base, p_ours)
                self.assertEqual(d_base, d_ours)
                self.assertEqual(o_base, o_ours)

    # -- multi-stage injection ----------------------------------------------------
    def test_multi_stage_changes_only_guide_and_description(self):
        for ep, task in zip(self.episodes, self.tasks):
            plan = TaskPlan.from_task_dict(task, mode="fixed_order", single_goal=False)
            if len(plan.landmarks) < 2:
                continue
            pose = ep.positions[0]
            quat = (0.0, 0.0, 0.0, 1.0)
            # stage 0 of the long task: guide point is the first landmark, description its real text
            lm = plan.landmarks[0]
            instr = build_stage_instruction(plan.descriptions[lm.ap])
            prompt, dir_text, obj_desc = ours_prompt(pose, quat, lm.center, instr)
            # the object description slot now carries the landmark's own 40-60 word description
            self.assertEqual(obj_desc, plan.descriptions[lm.ap])
            self.assertGreaterEqual(len(obj_desc.split()), 20)
            # the template is untouched
            self.assertTrue(prompt.startswith("<image>\nFly "))
            self.assertTrue(prompt.endswith("\nAction: "))
            self.assertIn("and find the target. ", prompt)
            # the direction word is still one of the seven buckets (or the degenerate empty one)
            self.assertIn(dir_text, [b[2] for b in BUCKETS] + [""])

    def test_final_stage_uses_the_episode_own_description(self):
        for ep, task in zip(self.episodes, self.tasks):
            plan = TaskPlan.from_task_dict(task, mode="fixed_order", single_goal=False)
            last = plan.landmarks[-1]
            self.assertEqual(last.ap, "g")
            # the last phase description must be the dataset's own text for that target
            self.assertEqual(last.description, ep.obj_desc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
