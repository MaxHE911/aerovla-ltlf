"""Paired comparison of two evaluation arms on the *same tasks*.

The simulator is not reproducible: the same split gives a different spawn pose on every reset, so two
arms never see an identical episode.  What IS shared is the task (instruction + ordered landmark
spec + final target), because both arms evaluate the same `seq`.  This script therefore pairs the two
arms **by seq** and reports the discordant counts with an exact McNemar test, which is the strongest
statement the protocol allows.

Usage:
    python ops/paired_compare.py --a stl-a --b stl-b --metric success
    python ops/paired_compare.py --a stl-a --b stl-b --metric esr_accepted --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

R = "/mnt/kai_ckp/yiding/MaxHE"


def load_arm(tag: str, metric: str):
    """{seq: bool} for one arm, read from the per-episode summaries."""
    out = {}
    for path in glob.glob(os.path.join(R, "ltlf_logs", tag, "*.summary.json")):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        seq = d.get("seq")
        if not seq:
            continue
        if metric == "esr_accepted":
            v = bool(d.get("esr_accepted"))
        elif metric == "sr_and_esr":
            v = bool(d.get("success")) and bool(d.get("esr_accepted"))
        elif metric == "collision":
            v = bool(d.get("collision"))
        elif metric == "stages_complete":
            stages = d.get("stages") or []
            v = bool(stages) and all(s.get("entered") for s in stages)
        else:
            v = bool(d.get(metric))
        out[seq] = v
    return out


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p-value for k successes out of n at p=0.5 (discordant pairs only)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i)
    return min(1.0, 2.0 * total / (2.0 ** n))


def compare(a: dict, b: dict, metric: str) -> dict:
    common = sorted(set(a) & set(b))
    only_a = [s for s in common if a[s] and not b[s]]
    only_b = [s for s in common if b[s] and not a[s]]
    both = [s for s in common if a[s] and b[s]]
    neither = [s for s in common if not a[s] and not b[s]]
    n_disc = len(only_a) + len(only_b)
    p = binom_two_sided(min(len(only_a), len(only_b)), n_disc)
    return {
        "metric": metric,
        "n_paired": len(common),
        "both": len(both),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "neither": len(neither),
        "rate_a": round((len(both) + len(only_a)) / len(common), 4) if common else None,
        "rate_b": round((len(both) + len(only_b)) / len(common), 4) if common else None,
        "mcnemar_exact_p": round(p, 4),
        "only_a_seqs": only_a,
        "only_b_seqs": only_b,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="tag of arm A (ltlf_logs/<tag>)")
    ap.add_argument("--b", required=True, help="tag of arm B")
    ap.add_argument("--metric", default="success",
                    choices=["success", "esr_accepted", "sr_and_esr", "collision", "stages_complete"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    a = load_arm(args.a, args.metric)
    b = load_arm(args.b, args.metric)
    if not a or not b:
        print("missing summaries: a=%d b=%d" % (len(a), len(b)))
        return 1
    res = compare(a, b, args.metric)
    res["arm_a"], res["arm_b"] = args.a, args.b
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_seqs")}, indent=1))
    print("A only: %s" % (res["only_a_seqs"] or "-"))
    print("B only: %s" % (res["only_b_seqs"] or "-"))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
