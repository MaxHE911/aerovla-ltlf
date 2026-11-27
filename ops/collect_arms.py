"""Collect every finished evaluation arm into one JSON + one markdown table.

Wraps ``ops/analyze_results.py`` (which owns the metric definitions) so that the weekly report is
generated from the *saved logs*, never from hand-typed numbers.

Usage (repository root, on the server):

    python ops/collect_arms.py                # writes work/results/all_arms_v2.json
    python ops/collect_arms.py --markdown      # also prints the markdown table

Arms that are still empty are skipped, so the command is safe to run while other waves are running.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

R = "/mnt/kai_ckp/yiding/MaxHE"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (label, tag, family, mode)
#   family: which figure/table the arm belongs to
#   mode:   "fixed_order" / "stl" track the full chain; "long_instruction" / "original" only F(g)
ARMS = [
    # --- ModernCityMap, full RT set, idea 1 vs baseline+long ---
    ("mcm_rt35_ours", "w1-g5-rt35", "mcm_rt35", "fixed_order"),
    ("mcm_rt35_base_long", "w1-g4-rt35", "mcm_rt35", "long_instruction"),
    # --- idea 2 (STL: monitor + deadline + shield) vs idea 1, same episodes ---
    ("stl_ours", "stl-a", "stl", "stl"),
    ("stl_base_order", "stl-b", "stl", "fixed_order"),
    # --- safety-envelope ablation: altitude floor 6.0 m vs expert p01 ---
    ("stlalt_alt6", "stlalt-a", "stlalt", "stl"),
    ("stlband_band", "stlband-a", "stlband", "stl"),
    ("stlalt_p01", "stlalt-b", "stlalt", "stl"),
    # --- idea 3: Stage-A logic-conditioned SFT checkpoint vs original checkpoint ---
    ("sa_order_sft", "sa_order-a", "sa_order", "fixed_order"),
    ("sa_order_base", "sa_order-b", "sa_order", "fixed_order"),
    # retrain attempts (see weekly report 5.5/5.6): v3 = from base, v4 = continued from the SFT adapter
    ("sa3_from_base", "sa3-a", "sa_retrain", "fixed_order"),
    ("sa4_from_ckpt", "sa4-a", "sa_retrain", "fixed_order"),
    ("sa_single_sft", "sa_single-a", "sa_single", "single_stage"),
    ("sa_single_base", "sa_single-b", "sa_single", "single_stage"),
    # --- RefineFly post-trained weights vs AeroVLA SFT, same mode ---
    ("rf_refinefly", "rf-a", "refinefly", "original"),
    ("rf_aerovla", "rf-b", "refinefly", "original"),
    # --- ModularPark (unseen map, zero training samples), 40-step horizon ---
    ("mp40_ours", "mp40-a", "unseen_map", "fixed_order"),
    ("mp40_base_long", "mp40-b", "unseen_map", "long_instruction"),
    # --- other seen maps ---
    ("nyc_ours", "nyc-a", "nyc", "fixed_order"),
    ("nyc_base_long", "nyc-b", "nyc", "long_instruction"),
    ("ti_ours", "ti-a", "ti", "fixed_order"),
    ("ti_base_long", "ti-b", "ti", "long_instruction"),
]


def wilson(k: float, n: int, z: float = 1.96):
    """Wilson score interval for a proportion; k may be fractional because it comes from a mean."""
    if not n:
        return (0.0, 0.0)
    import math

    p = max(0.0, min(1.0, float(k) / n))
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/results/all_arms_v2.json")
    ap.add_argument("--tasks", default="work/tasks/tasks_all.json")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)

    os.chdir(REPO)
    cmd = [sys.executable, "ops/analyze_results.py"]
    single_goal = []
    used = []
    for label, tag, family, mode in ARMS:
        res_dir = os.path.join(R, "eval_results", "_ltlf", tag)
        log_dir = os.path.join(R, "ltlf_logs", tag)
        if not os.path.isdir(res_dir):
            continue
        n = len([d for d in os.listdir(res_dir) if os.path.isdir(os.path.join(res_dir, d))])
        if n == 0:
            continue
        cmd += ["--arm", "%s=%s" % (label, res_dir), "--ltlf-logs", "%s=%s" % (label, log_dir)]
        used.append((label, family, mode, n))
        if mode in ("long_instruction", "original"):
            single_goal.append(label)
    if not used:
        print("no finished arms yet")
        return 1
    cmd += ["--tasks", args.tasks, "--single-goal-arms", ",".join(single_goal), "--out", args.out]
    print("running:", " ".join(cmd[:6]), "... %d arms" % len(used))
    subprocess.check_call(cmd)

    with open(args.out) as f:
        data = json.load(f)
    fam_of = {label: family for label, family, _, _ in used}
    mode_of = {label: mode for label, _, mode, _ in used}
    n_of = {label: n for label, _, _, n in used}
    for label, agg in data["arms"].items():
        agg["family"] = fam_of.get(label, "")
        agg["mode"] = mode_of.get(label, "")
        agg["n_dirs"] = n_of.get(label, 0)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=1)

    if args.markdown:
        keys = ["episodes", "sr", "osr", "ne", "spl", "esr", "sr_and_esr", "stage_completion_rate"]
        print("\n| 臂 (arm) | group | mode | " + " | ".join(keys) + " | sr_95%CI |")
        print("|---|---|---|" + "---|" * len(keys) + "---|")
        for label, agg in data["arms"].items():
            vals = []
            for k in keys:
                v = agg.get(k)
                vals.append("%.3f" % v if isinstance(v, float) else ("-" if v is None else str(v)))
            ci = ""
            n = agg.get("episodes") or 0
            sr = agg.get("sr")
            if n and isinstance(sr, float):
                lo, hi = wilson(sr * n, n)
                ci = "[%.2f, %.2f]" % (lo, hi)
            print("| %s | %s | %s | %s | %s |" % (label, agg.get("family"), agg.get("mode"), " | ".join(vals), ci))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
