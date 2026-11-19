"""Stage-2 tests: M0 task construction + L0 offline replay on real ModernCityMap data.

The data-dependent tests are skipped when the local data mirror is absent (the same tests are
run on the server, where the full dataset lives).
"""

from __future__ import annotations

import math
import os
import unittest

from ltlf_logic.formula import accept, parse, prog, to_str
from ltlf_logic.offline import check_task_replay, events_from_positions, independent_esr
from ltlf_logic.tasks import (
    build_description_index,
    build_task,
    load_episode,
    load_map_areas,
    load_split,
    unique_sequences,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA_CANDIDATES = [
    os.path.join(REPO, "..", "data_local"),          # local mirror (Windows workstation)
    os.path.join(REPO, "dataset_raw"),               # server: symlink at repo root
    os.path.join(REPO, "AeroVLA", "dataset_raw"),    # server: symlink inside the baseline copy
]
META_CANDIDATES = [
    os.path.join(REPO, "..", "data_local", "data", "meta", "map_spawnarea_info.json"),
    os.path.join(REPO, "AeroVLA", "data", "meta", "map_spawnarea_info.json"),
]
SPLIT_CANDIDATES = [
    os.path.join(REPO, "..", "data_local", "data", "uav_dataset", "seen_valset_splits", "ModernCityMap.json"),
    os.path.join(REPO, "AeroVLA", "data", "uav_dataset", "seen_valset_splits", "ModernCityMap.json"),
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


DATA_ROOT = _first_existing(DATA_CANDIDATES)
META = _first_existing(META_CANDIDATES)
SPLIT = _first_existing(SPLIT_CANDIDATES)

# verified against the baseline eval log of the stage-0 smoke run (target <29.40, 9.08, -0.09>)
SMOKE_SEQ = "84eabab1-99ff-4cb8-bde9-ad9459196148"


@unittest.skipUnless(DATA_ROOT and META and SPLIT, "local ModernCityMap data mirror not available")
class TestM0OnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.areas = load_map_areas(META, "ModernCityMap")
        cls.episode = load_episode(DATA_ROOT, "ModernCityMap", SMOKE_SEQ, cls.areas)
        cls.desc_index = build_description_index(DATA_ROOT, "ModernCityMap", cls.areas)
        cls.task = build_task(
            cls.episode, cls.areas, cls.desc_index,
            map_name="ModernCityMap", split_name="seen_valset",
        )

    def test_target_coordinate_matches_baseline_convention(self):
        """M0 must use area[9:12] - the same point the baseline flies to (env_uav.py:139)."""
        center = self.episode.target_area.center
        self.assertAlmostEqual(center[0], 29.40, places=1)
        self.assertAlmostEqual(center[1], 9.08, places=1)
        self.assertAlmostEqual(center[2], -0.09, places=1)

    def test_split_is_deduplicated(self):
        entries = load_split(SPLIT)
        seqs = unique_sequences(entries)
        self.assertEqual(len(seqs), len(set(seqs)))
        self.assertGreater(len(entries), len(seqs), "split lists several start frames per trajectory")

    def test_task_schema_and_formula(self):
        t = self.task
        for key in ["task_id", "formula", "event_order", "landmarks", "final_target", "long_instruction_zh",
                    "long_instruction_en", "generator_version", "st_rt", "n_stages"]:
            self.assertIn(key, t)
        self.assertEqual(t["n_stages"], t["n_landmarks"] + 1)
        self.assertEqual(t["event_order"][-1], "g")
        # the long spec must be F(L0 & F(... F(g))) with one nested F per stage
        self.assertTrue(t["formula"].startswith("F("))
        self.assertEqual(t["formula"].count("F("), t["n_stages"])

    def test_landmarks_have_real_long_descriptions(self):
        for lm in self.task["landmarks"]:
            self.assertIsNotNone(lm["description"])
            words = len(lm["description"].split())
            self.assertGreaterEqual(words, 20, "stage descriptions must be real 40-60 word texts")
            self.assertIsNotNone(lm["description_source_seq"])
        # the final target's description is the episode's own instruction text, verbatim
        self.assertEqual(self.task["final_target"]["description"], self.episode.obj_desc)

    def test_landmarks_are_pairwise_separated(self):
        centers = [tuple(l["center"]) for l in self.task["landmarks"]] + [tuple(self.task["final_target"]["center"])]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                self.assertGreater(math.dist(centers[i], centers[j]), 24.0)

    def test_expert_trajectory_satisfies_its_own_task(self):
        rep = check_task_replay(self.task, self.episode.positions)
        self.assertTrue(rep["expert_accepted_online"])
        self.assertTrue(rep["expert_accepted_independent"])
        self.assertTrue(rep["experts_agree"])
        self.assertTrue(rep["negatives_all_ok"], [n for n in rep["negatives"] if not n["ok"]])

    def test_negative_mutations_are_rejected(self):
        rep = check_task_replay(self.task, self.episode.positions)
        names = {n["name"] for n in rep["negatives"]}
        self.assertIn("missing_final_goal", names)
        self.assertIn("reversed", names)
        for n in rep["negatives"]:
            self.assertFalse(n["expected"] is True and n["online"] is False, n)

    def test_repeated_poses_are_deduplicated(self):
        # the simulator returns the same post-step pose 5x; dedupe must remove those duplicates
        episode = self.episode
        padded = []
        for p in episode.positions[:10]:
            padded.extend([p] * 5)
        a = events_from_positions(self.task, padded)
        b = events_from_positions(self.task, episode.positions[:10])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
