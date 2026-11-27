"""Figure 2: success rate of every finished arm with Wilson 95% intervals, grouped by experiment.

This is the "one slide" view of the week: it makes the small-sample overlaps obvious, which is the
point - none of the arms is separable from its control at these sample sizes.

Usage: python ops/plot_arms_sr.py --out work/results/fig_arms_sr.png
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = "/mnt/kai_ckp/yiding/MaxHE"

# (group label, [(arm label, tag), ...])
GROUPS = [
    ("idea1\nMCM RT", [("baseline+long", "w1-g4-rt35"), ("ours (LTLf)", "w1-g5-rt35")]),
    ("idea2\nonline", [("idea1 ctrl", "stl-b"), ("STL layer", "stl-a")]),
    ("idea2\naltitude", [("STL 1.7-33 m", "stl-a"), ("STL 6-33 m", "stlalt-a"), ("STL 7-12 m", "stlband-a")]),
    ("idea3\npost-train", [("orig weights", "rf-b"), ("Stage-A SFT", "sa_order-a")]),
    ("external\nown-ctl", [("AeroVLA", "rf-b"), ("RefineFly", "rf-a")]),
]


def wilson(k: float, n: int, z: float = 1.96):
    if not n:
        return (0.0, 0.0)
    p = max(0.0, min(1.0, k / n))
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def arm_stats(tag: str):
    files = glob.glob(os.path.join(R, "ltlf_logs", tag, "*.summary.json"))
    if not files:
        return None
    ok = 0
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        ok += bool(d.get("success"))
    n = len(files)
    lo, hi = wilson(ok, n)
    return ok / n, ok / n - lo, hi - ok / n, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/results/fig_arms_sr.png")
    args = ap.parse_args(argv)

    labels, rates, err_lo, err_hi, ns, colors, xticks, xlabels = [], [], [], [], [], [], [], []
    x = 0.0
    palette = {"baseline+long": "#8c8c8c", "ours (LTLf)": "#2b8cbe", "idea1 ctrl": "#8c8c8c",
               "STL layer": "#2b8cbe", "STL 1.7-33 m": "#8c8c8c", "STL 6-33 m": "#f0a202",
               "STL 7-12 m": "#de2d26", "orig weights": "#8c8c8c", "Stage-A SFT": "#de2d26",
               "AeroVLA": "#8c8c8c", "RefineFly": "#31a354"}
    for gname, arms in GROUPS:
        first = x
        for alabel, tag in arms:
            st = arm_stats(tag)
            if st is None:
                continue
            r, lo, hi, n = st
            labels.append(alabel)
            rates.append(r)
            err_lo.append(lo)
            err_hi.append(hi)
            ns.append(n)
            colors.append(palette.get(alabel, "#4d4d4d"))
            xlabels.append("%s\nn=%d" % (alabel, n))
            x += 1.0
        xticks.append((first + x - 1.0) / 2.0)
        x += 0.6  # gap between groups

    if not rates:
        print("no arms with results")
        return 1

    fig, ax = plt.subplots(figsize=(12.0, 4.4), dpi=160)
    xs = np.arange(len(rates), dtype=float)
    # re-space bars to keep the group gaps
    pos, cur, gi = [], 0.0, 0
    for i, g in enumerate(xlabels):
        if gi < len(GROUPS) and i >= sum(len(a) for _, a in GROUPS[: gi + 1]):
            gi += 1
            cur += 0.6
        pos.append(cur)
        cur += 1.0
    ax.bar(pos, rates, yerr=[err_lo, err_hi], capsize=5, color=colors, width=0.62, edgecolor="white")
    for p, r, n in zip(pos, rates, ns):
        ax.text(p, r + 0.055, "%.2f" % r, ha="center", fontsize=9)
    ax.set_xticks(pos)
    ax.set_xticklabels(xlabels, fontsize=7.5, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate (Wilson 95% CI)")
    ax.set_title("every arm is inside its control's confidence interval: the samples are too small to separate them",
                 fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.25)
    # group separators + labels
    bounds = []
    cur = 0.0
    for gname, arms in GROUPS:
        k = sum(1 for _, t in arms if arm_stats(t) is not None)
        if k == 0:
            continue
        lo, hi = cur, cur + k - 1
        bounds.append((lo, hi, gname))
        cur = hi + 0.6 + 1.0
    for lo, hi, gname in bounds:
        if lo > 0:
            ax.axvline(lo - 0.8, color="#cccccc", linewidth=0.8)
        ax.text((lo + hi) / 2, 0.04, gname, ha="center", fontsize=8, color="#444444")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)
    for l, r, n in zip(xlabels, rates, ns):
        print("  %-22s %.3f (n=%d)" % (l.replace("\n", " "), r, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
