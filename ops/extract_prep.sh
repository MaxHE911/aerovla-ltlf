#!/bin/bash
# 解压仿真环境 + 数据集，删压缩包，生成 merged_data.json
set -u
R=/mnt/kai_ckp/yiding/MaxHE
PY=$R/envs_py/aerovla_py310/bin/python
GEN=/mnt/world_foundational_model/kai/MaxHE/TravelUAV/Model/LLaMA-UAV/tools/generate_merged_json.py
MAP=${1:-ModernCityMap}
log(){ echo "[$(date '+%F %T')] $*"; }

mkdir -p "$R/envs" "$R/dataset_raw"

if [ -d "$R/envs_archives" ] && ls "$R/envs_archives"/closeloop_envs.zip >/dev/null 2>&1; then
  log "解压 closeloop_envs -> $R/envs"
  ( cd "$R/envs_archives" && unzip -o -q closeloop_envs.zip -d "$R/envs" ) && log "closeloop_envs 解压完成"
else
  log "没有 closeloop_envs 压缩包，跳过"
fi

if [ -d "$R/dataset_archives" ] && ls "$R/dataset_archives/$MAP.zip" >/dev/null 2>&1; then
  log "解压 $MAP -> $R/dataset_raw"
  ( cd "$R/dataset_archives" && unzip -o -q "$MAP.zip" -d "$R/dataset_raw" ) && log "$MAP 解压完成"
else
  log "没有 $MAP 压缩包，跳过"
fi

log "删除压缩包（用户要求）"
rm -rf "$R/envs_archives" "$R/dataset_archives"

log "生成 merged_data.json ($MAP)"
cd "$R"
"$PY" "$GEN" --root_dir "$R/dataset_raw" --map_list "$MAP" 2>&1 | tail -5

log "=== 结果 ==="
echo "envs:";        ls "$R/envs" 2>/dev/null | head
echo "closeloop:";   ls "$R/envs/closeloop_envs" 2>/dev/null | head
echo "dataset_raw:"; ls "$R/dataset_raw" 2>/dev/null | head
echo "轨迹数: $(ls -d $R/dataset_raw/$MAP/*/ 2>/dev/null | wc -l)"
echo "merged_data.json 数: $(find $R/dataset_raw/$MAP -name merged_data.json 2>/dev/null | wc -l)"
echo "总大小:"; du -sh "$R/envs" "$R/dataset_raw" 2>/dev/null
