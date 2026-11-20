# aerovla-ltlf — an LTLf task-logic layer on top of the AeroVLA baseline

This repository holds (a) a verbatim snapshot of the verified AeroVLA baseline (tag
`baseline-v0`) and (b) the v1 implementation of an **external LTLf task-logic layer**: the
baseline keeps its weights, prompt template and action head; the layer only decides *which
sub-goal and which object description* are handed to the model at each step, and when a LAND is
allowed.

The design this implementation follows is `work/idea1.md` of the project (five modules M0–M5);
the implementation log, the stage audits and the morning report live in `work/`.

## Layout

```
AeroVLA/                 baseline snapshot (upstream b87aaa7), never edited by this project
TravelUAV/               auxiliary snapshot (only its data-merge tool is used)
ops/                     runnable scripts (Xvfb + UE4 + eval drivers, offline replay, metrics)
ltlf_logic/              the logic layer (pure Python, no torch / no simulator imports except eval_loop)
  formula.py             LTLf representation, canonical normal form, progression, empty-trace accept
  semantics.py           independent finite-trace semantics (cross-check oracle)
  automaton.py           M1 residual automaton, accepting-path enumeration, swap predicate
  monitor.py             M4 event detection + residual update
  runner.py              M3/M5 helpers: phase plans, prompt injection, LAND gate, fallback push
  tasks.py               M0 offline spec construction from episode data
  offline.py             L0 offline replay + independent ESR judge
  eval_loop.py           evaluation loop wired to the baseline (imports the baseline)
ltlf_tests/              unittest suite (stdlib only)
work/                    implementation log, stage audits, task JSON, splits, results
```

## Quick start (server, from the repository root)

```bash
PY=/mnt/kai_ckp/yiding/MaxHE/envs_py/aerovla_py310/bin/python

# 1. unit tests (no GPU, no simulator)
$PY -m unittest discover -s ltlf_tests -t .

# 2. M0: build long-horizon tasks for ModernCityMap seen_valset
$PY -m ltlf_logic.tasks \
    --dataset-root AeroVLA/dataset_raw \
    --spawn-area-json AeroVLA/data/meta/map_spawnarea_info.json \
    --split-json AeroVLA/data/uav_dataset/seen_valset_splits/ModernCityMap.json \
    --out-dir work/tasks

# 3. L0: replay the expert trajectories through their own specs (must all accept)
$PY ops/run_offline_replay.py \
    --dataset-root AeroVLA/dataset_raw \
    --spawn-area-json AeroVLA/data/meta/map_spawnarea_info.json \
    --split-json AeroVLA/data/uav_dataset/seen_valset_splits/ModernCityMap.json \
    --tasks work/tasks/tasks_ModernCityMap_seen_valset.json \
    --out work/results/offline_replay_ModernCityMap.json

# 4. de-duplicated evaluation splits (the shipped splits repeat one trajectory per start frame)
$PY ops/make_split.py --split AeroVLA/data/uav_dataset/seen_valset_splits/ModernCityMap.json \
    --tasks work/tasks/tasks_ModernCityMap_seen_valset.json --select first:12 --out work/splits/stage3_12ep.json

# 5. closed-loop arms (each needs an Xvfb + a UE4 simulator; the script starts them)
TAG=g0-baseline GPU=0 PORT=30000 STEPS=70 SPLIT=work/splits/stage3_12ep.json bash ops/run_baseline_eval.sh
TAG=g1-ours GPU=1 PORT=30010 STEPS=70 SPLIT=work/splits/stage3_12ep.json bash ops/run_ltlf_eval.sh  # MODE=single_stage
TAG=g5-ours GPU=0 PORT=30000 STEPS=70 MODE=fixed_order SPLIT=work/splits/stage5_rt8.json \
    TASK_JSON=work/tasks/tasks_ModernCityMap_seen_valset.json bash ops/run_ltlf_eval.sh

# 6. independent offline metrics for every arm
$PY ops/analyze_results.py --arm g0=<result dir> --arm g1=<result dir> \
    --tasks work/tasks/tasks_ModernCityMap_seen_valset.json --single-goal-arms g0,g1 \
    --ltlf-logs g1=<ltlf log dir> --out work/results/stage3_metrics.json
```

`ops/run_ltlf_eval.sh` modes: `original`, `single_stage` (F(g) degradation, no gate, no
takeover), `long_instruction` (baseline + long instruction in the description slot),
`fixed_order` (the logic layer drives phase by phase, LAND gated on the residual formula).

## Environment facts that matter

* the simulator must run with an X display (`Xvfb :99`) and the UE4 Vulkan adapter index is **not**
  the `nvidia-smi` index — derive it with `vulkan_adapter.py`;
* **UE4 refuses to start as root** (`Refusing to run with the root privileges.`): the simulator server
  must be started with `LD_PRELOAD=ops/sim-runtime/lib/libfakerootuid.so` (source + build instructions in
  `ops/sim-runtime/`, rationale and pitfalls in `docs/ue4-root-refusal.md`); do **not** preload it into Xvfb;
* the baseline resolves the relative dataset path, so every arm runs with `cwd=AeroVLA` and
  `--dataset_path dataset_raw/`;
* the simulator returns the same post-step state five times per control step, so one control step
  carries exactly one new pose;
* closed-loop runs are **not** bit-reproducible (initial settle pose and rendered frames differ
  run to run), so arms are compared statistically, and interface equivalence is proven by unit
  tests instead.
