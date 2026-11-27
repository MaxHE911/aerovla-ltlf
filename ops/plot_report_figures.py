"""Figures for the weekly report / group meeting.

Panel A: mean flight altitude of each episode, split by outcome (collision vs success) - the
         diagnostic that explains why the logic layer alone cannot lift the success rate.
Panel B: success rate per arm with Wilson 95% intervals - shows how weak the small-sample
         differences are.

Usage: python ops/plot_report_figures.py --out work/results/fig_altitude_sr.png
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

# (label, ltlf_log tag) - order matters for the figure
ARMS = [
    ("baseline\n+long instr.", "w1-g4-rt35"),
    ("ours\n(LTLf layer)", "w1-g5-rt35"),
]
ALT_ARMS = [("baseline + long instruction", "w1-g4-rt35"), ("ours (LTLf layer)", "w1-g5-rt35")]


def wilson(k: float, n: int, z: float = 1.96):
    if not n:
        return (0.0, 0.0)
    p = max(0.0, min(1.0, k / n))
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def episode_altitudes(tag: str):
    """(mean_altitude, collided) per finished episode."""
    out = []
    for summ in glob.glob(os.path.join(R, "ltlf_logs", tag, "*.summary.json")):
        try:
            with open(summ) as f:
                d = json.load(f)
        except Exception:
            continue
        seq = d.get("seq")
        jsonl = os.path.join(R, "ltlf_logs", tag, "%s.jsonl" % seq)
        if not os.path.exists(jsonl):
            continue
        alts = []
        try:
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    p = rec.get("position")
                    if p:
                        alts.append(-float(p[2]))
        except Exception:
            continue
        if not alts:
            continue
        out.append((float(np.mean(alts)), bool(d.get("collision")), bool(d.get("success"))))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/results/fig_altitude_sr.png")
    args = ap.parse_args(argv)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=160)

    # ---- panel A: altitude vs outcome ----
    ax = axes[0]
    rng = np.random.default_rng(0)
    ticks, labels = [], []
    for i, (label, tag) in enumerate(ALT_ARMS):
        eps = episode_altitudes(tag)
        if not eps:
            continue
        col = [a for a, c, s in eps if c]
        ok = [a for a, c, s in eps if not c]
        for j, (vals, color, name) in enumerate(
            [(ok, "#2b8cbe", "no collision"), (col, "#de2d26", "collided")]
        ):
            if not vals:
                continue
            x = i * 3 + j
            ax.scatter(
                x + rng.uniform(-0.18, 0.18, len(vals)),
                vals,
                s=46,
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.6,
                label=name if i == 0 else None,
                zorder=3,
            )
            ax.hlines(
                np.mean(vals), x - 0.3, x + 0.3, color=color, linewidth=2.4, zorder=4
            )
            ticks.append(x)
            labels.append("%s\n%sn=%d" % (label, name.replace(" ", "\n"), len(vals)))
    ax.axhline(9.84, color="#666666", linestyle="--", linewidth=1.1, zorder=2)
    ax.text(2.6, 10.4, "expert median 9.84 m", fontsize=8, color="#666666", ha="right")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("mean flight altitude per episode (m)")
    ax.set_title("A. episodes that collide fly much lower", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(fontsize=8, loc="upper right")

    # ---- panel B: success rate with Wilson CI ----
    ax = axes[1]
    names, rates, los, his, ns = [], [], [], [], []
    for label, tag in ARMS:
        vals = []
        for summ in glob.glob(os.path.join(R, "ltlf_logs", tag, "*.summary.json")):
            try:
                with open(summ) as f:
                    d = json.load(f)
            except Exception:
                continue
            vals.append(bool(d.get("success")))
        if not vals:
            continue
        k, n = sum(vals), len(vals)
        lo, hi = wilson(k, n)
        names.append(label)
        rates.append(k / n)
        los.append(k / n - lo)
        his.append(hi - k / n)
        ns.append(n)
    xs = np.arange(len(names))
    ax.bar(xs, rates, yerr=[los, his], capsize=6, color=["#7f7f7f", "#2b8cbe"][: len(names)],
           width=0.55, edgecolor="white")
    for x, r, n in zip(xs, rates, ns):
        ax.text(x, r + 0.045, "%.2f (n=%d)" % (r, n), ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("success rate")
    ax.set_title("B. on the full RT set the gap is inside the noise", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
