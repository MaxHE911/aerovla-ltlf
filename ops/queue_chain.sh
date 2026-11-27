#!/usr/bin/env bash
# Launch the remaining evaluation chain as dependent waiters (no cron, no manual timing).
# Each waiter blocks on the process patterns of the previous wave, then starts its arm.
#
# Order:  wave1 (already running)  ->  mp (unseen map)  ->  stl (idea 2 vs fixed_order)
#         -> rf (RefineFly post-trained weights vs AeroVLA SFT)  ->  nyc -> ti -> nycem
set -u
REPO=/mnt/world_foundational_model/kai/MaxHE/ltlf
R=/mnt/kai_ckp/yiding/MaxHE
cd "$REPO" || exit 1
mkdir -p "$R/logs"
TSK=work/tasks/tasks_ModernCityMap_seen_valset.json

arm(){ # arm <waits> <tag> <mode> <port> <split> <tasks> [extra_env]
  local waits=$1 tag=$2 mode=$3 port=$4 split=$5 tasks=$6 extra=${7:-}
  setsid nohup env WAIT="$waits" TAG="$tag" MODE="$mode" PORT="$port" SPLIT="$split" \
      TASK_JSON="$tasks" GPU=0 STEPS=80 EXTRA_ENV="$extra" \
      bash ops/run_next.sh > /dev/null 2>&1 < /dev/null &
  echo "queued $tag (mode=$mode port=$port waits=[$waits])"
}

# ---- wave 2: ModularPark, the official *unseen map* split ----
arm "w1-g5-rt35"  mp-g5  fixed_order      30200 work/splits/mp_rt14.json work/tasks/tasks_ModularPark_unseen_map_valset.json
arm "w1-g4-rt35"  mp-g4  long_instruction 30210 work/splits/mp_rt14.json work/tasks/tasks_ModularPark_unseen_map_valset.json

# ---- wave 3: idea 2 (STL) vs idea 1 (fixed order), same episodes ----
arm "mp-g5 mp-g4" stl-g5  stl             30300 work/splits/rt12.json "$TSK"
arm "mp-g5 mp-g4" stl-ctl fixed_order     30310 work/splits/rt12.json "$TSK"

# ---- wave 4: RefineFly post-trained adapter vs the AeroVLA SFT checkpoint ----
arm "stl-g5 stl-ctl" rf-rf   original 30400 work/splits/rt12.json - "MODEL_PATH=$R/models/refinefly"
arm "stl-g5 stl-ctl" rf-base original 30410 work/splits/rt12.json -

# ---- wave 5-7: the other seen maps ----
arm "rf-rf rf-base"      nyc-g5  fixed_order      30500 work/splits/NewYorkCity_rt12.json work/tasks/tasks_NewYorkCity_seen_valset.json
arm "rf-rf rf-base"      nyc-g4  long_instruction 30510 work/splits/NewYorkCity_rt12.json work/tasks/tasks_NewYorkCity_seen_valset.json
arm "nyc-g5 nyc-g4"      ti-g5   fixed_order      30600 work/splits/TropicalIsland_st12.json work/tasks/tasks_TropicalIsland_seen_valset.json
arm "nyc-g5 nyc-g4"      ti-g4   long_instruction 30610 work/splits/TropicalIsland_st12.json work/tasks/tasks_TropicalIsland_seen_valset.json
arm "ti-g5 ti-g4"        nycem-g5 fixed_order      30700 work/splits/NYCEnvironmentMegapa_rt12.json work/tasks/tasks_NYCEnvironmentMegapa_seen_valset.json
arm "ti-g5 ti-g4"        nycem-g4 long_instruction 30710 work/splits/NYCEnvironmentMegapa_rt12.json work/tasks/tasks_NYCEnvironmentMegapa_seen_valset.json
echo "all waiters queued; see $R/logs/waiter-<tag>.log"
