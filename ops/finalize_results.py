"""Produce every number the report needs, in one go, from the saved logs only.

Writes ``work/results/final_summary.md`` with four sections:

    A. every arm's metrics (delegated to ops/analyze_results.py, which owns the metric definitions)
    B. per-wave paired exact McNemar tests (delegated to ops/paired_compare.py)
    C. offline STL robustness of the recorded rollouts (delegated to ltlf_logic.stl_tasks)
    D. whether the STL mechanisms actually fired (shield / deadline hints per arm)

Everything is read from ``$R/ltlf_logs`` and ``$R/eval_results``, so it can be re-run at any time and
is safe to run while other arms are still evaluating.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = "/mnt/kai_ckp/yiding/MaxHE"
PY = sys.executable

# waves that were run as (a=A, b=B) pairs
WAVES = ["stl", "stlalt", "sa_order", "rf", "mp40", "nyc", "ti", "sa_single"]
# arms whose rollouts are on ModernCityMap with the full spec (for section C)
MCM_TAGS = ["stl-a", "stl-b", "stlalt-a", "stlalt-b", "sa_order-a", "sa_order-b", "w1-g5-rt35", "w1-g4-rt35"]
STL_TAGS = ["stl-a", "stl-b", "stlalt-a", "stlalt-b"]


def run(cmd: list) -> str:
    try:
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    except Exception as exc:  # pragma: no cover
        return "ERROR %s" % exc
    return out.stdout


def section_a() -> list:
    txt = run([PY, "ops/collect_arms.py", "--markdown"])
    lines = [ln for ln in txt.splitlines() if ln.startswith("| ")]
    # keep the header rows and the data rows of the second (labelled) table only
    if not lines:
        return ["(no finished arms)"]
    start = next((i for i, ln in enumerate(lines) if "臂 (arm)" in ln), 0)
    return lines[start:]


def section_b() -> list:
    rows = ["| 对比 | 指标 | 配对 n | 都成功 | 仅 A | 仅 B | 都不成 | p |", "|---|---|---|---|---|---|---|---|"]
    any_row = False
    for wave in WAVES:
        for metric in ("success", "esr_accepted"):
            js = os.path.join(REPO, "work/results", "pair_%s_%s.json" % (wave, metric))
            out = run([PY, "ops/paired_compare.py", "--a", "%s-a" % wave, "--b", "%s-b" % wave,
                       "--metric", metric, "--json", js])
            try:
                d = json.loads(out[out.index("{"): out.rindex("}") + 1])
            except Exception:
                continue
            if not d.get("n_paired"):
                continue
            any_row = True
            rows.append("| %s | %s | %d | %d | %d | %d | %d | %.3f |" % (
                wave, metric, d["n_paired"], d["both"], d["only_a"], d["only_b"], d["neither"],
                d["mcnemar_exact_p"]))
    return rows if any_row else ["(no paired arms finished yet)"]


def section_c() -> list:
    rows = ["| 臂 | n | chain 满足 | schedule 满足 | 包络满足 | ρ_chain 中位 | ρ_sched 中位 | 迟到率 | 从未到达率 | chain≡LTLf |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    any_row = False
    for tag in MCM_TAGS:
        log_dir = os.path.join(R, "ltlf_logs", tag)
        if not os.path.isdir(log_dir) or not glob.glob(os.path.join(log_dir, "*.jsonl")):
            continue
        out_path = os.path.join(REPO, "work/results", "stl_off2_%s.json" % tag)
        run([PY, "-m", "ltlf_logic.stl_tasks", "score", "--logs", log_dir,
             "--tasks", "work/tasks/tasks_ModernCityMap_seen_valset.json",
             "--dataset-root", "AeroVLA/dataset_raw/", "--out", out_path, "--label", tag])
        try:
            with open(out_path) as f:
                a = json.load(f)["aggregate"]
        except Exception:
            continue
        any_row = True

        def f(k):
            v = a.get(k)
            return "%.3f" % v if isinstance(v, (int, float)) else "-"

        rows.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            tag, a["n"], f("chain_sat_rate"), f("schedule_sat_rate"), f("safety_sat_rate"),
            f("rho_chain_median"), f("rho_schedule_median"), f("stage_late_rate"),
            f("stage_never_entered_rate"), f("chain_vs_ltlf_agreement")))
    return rows if any_row else ["(no ModernCityMap rollouts finished yet)"]


def section_d() -> list:
    rows = ["| 臂 | n | SR | ESR | 碰撞率 | shield 触发 | 截止期提示 | ρ_safety 中位 |",
            "|---|---|---|---|---|---|---|---|"]
    any_row = False
    for tag in STL_TAGS:
        files = glob.glob(os.path.join(R, "ltlf_logs", tag, "*.summary.json"))
        if not files:
            continue
        n = len(files)
        suc = esr = coll = sh = hints = 0
        rho = []
        for path in files:
            try:
                with open(path) as f:
                    d = json.load(f)
            except Exception:
                continue
            suc += bool(d.get("success"))
            esr += bool(d.get("esr_accepted"))
            coll += bool(d.get("collision"))
            st = d.get("stl") or {}
            sh += st.get("shield_fired") or 0
            hints += st.get("deadline_hints_injected") or 0
            if st.get("rho_safety") is not None:
                rho.append(st["rho_safety"])
        any_row = True
        med = sorted(rho)[len(rho) // 2] if rho else None
        rows.append("| %s | %d | %.3f | %.3f | %.3f | %d | %d | %s |" % (
            tag, n, suc / n, esr / n, coll / n, sh, hints, ("%.2f" % med) if med is not None else "-"))
    return rows if any_row else ["(no STL arms finished yet)"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/results/final_summary.md")
    args = ap.parse_args(argv)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)

    blocks = [
        ("# 自动生成的结果汇总（%s）" % os.popen("date '+%F %T'").read().strip(), []),
        ("## A. 全部臂（ops/collect_arms.py；指标口径见 ops/analyze_results.py）", section_a()),
        ("## B. 按任务配对的精确 McNemar 检验（ops/paired_compare.py）", section_b()),
        ("## C. STL 离线鲁棒度（ltlf_logic.stl_tasks，ModernCityMap）", section_c()),
        ("## D. STL 机制是否真的触发（shield / 截止期提示）", section_d()),
    ]
    with open(out_path, "w") as f:
        for title, lines in blocks:
            f.write("\n" + title + "\n\n")
            for ln in lines:
                f.write(ln + "\n")
    print("wrote", out_path)
    with open(out_path) as f:
        print(f.read()[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
