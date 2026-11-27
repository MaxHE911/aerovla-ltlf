"""Probe: what is actually inside a split file, and what does the loader turn it into?

Answers three questions:
  1. what does one split record contain (and is the prompt part of it)?
  2. how many (trajectory, frame) records vs unique trajectories, per trajectory distribution
  3. how many trajectories exist on disk vs how many the eval split references
"""
import collections
import json
import os
import sys

split_path, dataset_root, map_name = sys.argv[1], sys.argv[2], sys.argv[3]
entries = json.load(open(split_path))

print("=== 一条 split 记录长什么样（前 3 条）===")
for e in entries[:3]:
    print("  ", e, "  keys =", sorted(e.keys()))

by_traj = collections.defaultdict(list)
for e in entries:
    seq = e["json"].strip("/").split("/")[-2]
    by_traj[seq].append(e.get("frame"))

print("\n=== 记录数 vs 唯一轨迹 ===")
print("  split 记录数          :", len(entries))
print("  唯一轨迹数            :", len(by_traj))
print("  唯一 (轨迹, frame) 组合:", len({(e['json'], e.get('frame')) for e in entries}))
per = [len(v) for v in by_traj.values()]
print("  每条轨迹的记录数      : min %d / median %d / max %d"
      % (min(per), sorted(per)[len(per) // 2], max(per)))
print("  frame 取值范围        : min %d / max %d" % (min(min(v) for v in by_traj.values()),
                                                     max(max(v) for v in by_traj.values())))

print("\n=== 提示词来自哪里？===")
seq0 = sorted(by_traj)[0]
merged = json.load(open(os.path.join(dataset_root, map_name, seq0, "merged_data.json")))
conv = merged["conversations"]
print("  merged_data.json 的 conversations 条数:", len(conv), " roles:", [c.get("from") for c in conv])
print("  conversations[0] 前 140 字:", conv[0]["value"][:140].replace("\n", " | "))
print("  -> 提示词在 merged_data.json 里，按轨迹存一份；split 里只有 json + frame")

print("\n=== 磁盘上有多少轨迹 / 评测 split 引用了多少 ===")
on_disk = [d for d in os.listdir(os.path.join(dataset_root, map_name))
           if os.path.isdir(os.path.join(dataset_root, map_name, d))]
print("  disk 轨迹数           :", len(on_disk))
print("  split 引用的轨迹数    :", len(by_traj))
print("  split 是 disk 的子集? :", set(by_traj) <= set(on_disk))
print("  未被 split 引用       :", len(set(on_disk) - set(by_traj)))

print("\n=== 轨迹长度 vs split 里给出的 frame ===")
lens = []
for seq in list(sorted(by_traj))[:10]:
    m = json.load(open(os.path.join(dataset_root, map_name, seq, "merged_data.json")))
    lens.append((seq[:8], len(m["trajectory_raw_detailed"]), min(by_traj[seq]), max(by_traj[seq])))
print("  (seq, 轨迹帧数, frame_min, frame_max):")
for row in lens:
    print("   ", row)
