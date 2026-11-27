#!/usr/bin/env bash
# Produce every number the report needs, in one go, from the saved logs only.
# Writes work/results/final_summary.md (+ json files next to it).
#
#   bash ops/finalize_results.sh
set -u
REPO=/mnt/world_foundational_model/kai/MaxHE/ltlf
R=/mnt/kai_ckp/yiding/MaxHE
PY=$R/envs_py/aerovla_py310/bin/python
OUT=$REPO/work/results/final_summary.md
mkdir -p "$REPO/work/results"
cd "$REPO" || exit 1

{
  echo "# 自动生成的结果汇总（$(date '+%F %T')）"
  echo
  echo "## A. 全部臂（来自 ops/collect_arms.py，指标口径见 ops/analyze_results.py）"
  echo
  "$PY" ops/collect_arms.py --markdown 2>&1 | sed -n '/^| 臂/,$p'
  echo
  echo "## B. 按任务配对的精确 McNemar 检验（ops/paired_compare.py）"
  echo
  echo "| 对比 | 指标 | 配对 n | 都成功 | 仅 A | 仅 B | 都不成 | p |"
  echo "|---|---|---|---|---|---|---|---|"
  for wave in stl stlalt sa_order rf mp40 nyc ti sa_single; do
    for metric in success esr_accepted; do
      "$PY" ops/paired_compare.py --a "${wave}-a" --b "${wave}-b" --metric "$metric" \
        --json "work/results/pair_${wave}_${metric}.json" 2>/dev/null \
      | "$PY" -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    sys.exit(0)
if not d.get('n_paired'): sys.exit(0)
print('| %s | %s | %d | %d | %d | %d | %d | %.3f |' % (
  '${wave}', '${metric}', d['n_paired'], d['both'], d['only_a'], d['only_b'], d['neither'], d['mcnemar_exact_p']))
" 2>/dev/null
    done
  done
  echo
  echo "## C. STL 离线鲁棒度（对已跑完的 rollout 重打分，ModernCityMap）"
  echo
  echo "| 臂 | n | chain 满足 | schedule 满足 | 包络满足 | ρ_chain 中位 | ρ_sched 中位 | 迟到率 | 从未到达率 |"
  echo "|---|---|---|---|---|---|---|---|---|"
  for tag in stl-a stl-b stlalt-a stlalt-b sa_order-a sa_order-b w1-g5-rt35 w1-g4-rt35; do
    if [ -d "$R/ltlf_logs/$tag" ] && [ -n "$(ls -A "$R/ltlf_logs/$tag" 2>/dev/null)" ]; then
      "$PY" -m ltlf_logic.stl_tasks score --logs "$R/ltlf_logs/$tag" \
          --tasks work/tasks/tasks_ModernCityMap_seen_valset.json \
          --dataset-root AeroVLA/dataset_raw/ \
          --out "work/results/stl_off2_${tag}.json" --label "$tag" > /dev/null 2>&1
      "$PY" -c "
import json
try:
    d=json.load(open('work/results/stl_off2_${tag}.json')); a=d['aggregate']
except Exception:
    raise SystemExit(0)
f=lambda k: ('%.3f'%a[k]) if isinstance(a.get(k),(int,float)) else '-'
print('| %s | %d | %s | %s | %s | %s | %s | %s | %s |' % ('${tag}', a['n'],
  f('chain_sat_rate'), f('schedule_sat_rate'), f('safety_sat_rate'),
  f('rho_chain_median'), f('rho_schedule_median'), f('stage_late_rate'), f('stage_never_entered_rate')))
" 2>/dev/null
    fi
  done
  echo
  echo "## D. STL 机制使用统计（是否真的触发了）"
  echo
  echo "| 臂 | n | 成功 | ESR | 碰撞 | shield 触发次数 | 截止期提示次数 | ρ_safety 中位 |"
  echo "|---|---|---|---|---|---|---|---|"
  for tag in stl-a stl-b stlalt-a stlalt-b; do
    "$PY" -c "
import json,glob
fs=glob.glob('$R/ltlf_logs/${tag}/*.summary.json')
if not fs: raise SystemExit(0)
n=len(fs); suc=esr=coll=sh=hints=0; rho=[]
for f in fs:
    d=json.load(open(f)); suc+=bool(d.get('success')); esr+=bool(d.get('esr_accepted')); coll+=bool(d.get('collision'))
    st=d.get('stl') or {}
    sh+=st.get('shield_fired') or 0; hints+=st.get('deadline_hints_injected') or 0
    if st.get('rho_safety') is not None: rho.append(st['rho_safety'])
med=sorted(rho)[len(rho)//2] if rho else None
print('| ${tag} | %d | %.3f | %.3f | %.3f | %d | %d | %s |' % (n, suc/n, esr/n, coll/n, sh, hints,
  ('%.2f'%med) if med is not None else '-'))
" 2>/dev/null
  done
} > "$OUT"

echo "wrote $OUT"
wc -l "$OUT"
