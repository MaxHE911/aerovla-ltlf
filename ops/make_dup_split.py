"""Build a tiny split whose records repeat the same trajectory with different `frame` values.

Used to check whether the baseline loader collapses such duplicates (it does: it only reads
`item["json"]`).
"""
import json
import sys

src, out_path, n_traj, n_frames = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
entries = json.load(open(src))

by_path = {}
for e in entries:
    by_path.setdefault(e["json"], []).append(e)

selected = list(by_path.items())[:n_traj]
out = []
for path, group in selected:
    out.extend(group[:n_frames])

json.dump(out, open(out_path, "w"))
print("wrote %s: %d records, %d unique trajectories, %d records per trajectory"
      % (out_path, len(out), len({e["json"] for e in out}), n_frames))
