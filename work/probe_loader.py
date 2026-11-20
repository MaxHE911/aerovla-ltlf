"""Probe: how many episodes does the baseline loader actually build from a split file?

The shipped splits contain one record per *start frame* (`{"json": ..., "frame": N}`).  The loader
in `env_uav.py:114-161` only uses `item["json"]`, so records that differ only in `frame` collapse.
This script measures that without touching the simulator (the vector-env workers are stubbed out).

Run from the AeroVLA directory with PYTHONPATH pointing at it:

    cd AeroVLA && PYTHONPATH=. python /path/to/probe_loader.py <split.json>
"""
import json
import os
import sys
import tempfile

split = sys.argv[1]
save_dir = tempfile.mkdtemp(prefix="probe_save_")
sys.argv = [
    "probe", "--run_type", "eval", "--batchSize", "1",
    "--dataset_path", "dataset_raw/",
    "--eval_save_path", save_dir,
    "--eval_json_path", split,
    "--map_spawn_area_json_path", "data/meta/map_spawnarea_info.json",
    "--object_name_json_path", "data/meta/object_description.json",
]

from src.vlnce_src.env_uav import AirVLNENV  # noqa: E402


class Probe(AirVLNENV):
    def init_VectorEnvUtil(self):
        pass

    def delete_VectorEnvUtil(self):
        pass


entries = json.load(open(split))
env = Probe(batch_size=1, dataset_path="dataset_raw/", save_path=save_dir, eval_json_path=split)

unique_paths = {e["json"] for e in entries}
frames = sorted({e.get("frame") for e in entries})
start_poses = {tuple(d["trajectory"][0]["position"]) for d in env.data}

print("split 记录数            :", len(entries))
print("唯一 json 路径数        :", len(unique_paths))
print("env.data（真正评测 episode 数）:", len(env.data))
print("frame 字段取值（前几个）:", frames[:6], "..." if len(frames) > 6 else "")
print("每条 episode 的起始位姿都不同?", len(start_poses) == len(env.data))
print("同一条轨迹的多次记录 -> 起始位姿相同?",
      len({tuple(d["trajectory"][0]["position"]) for d in env.data}) < len(entries))
