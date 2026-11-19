#!/bin/bash
# 按官方 troubleshooting 修 AirSim RPC 慢的问题
set -u
R=/mnt/kai_ckp/yiding/MaxHE
V=$R/envs_py/aerovla_py310
C=/mnt/world_foundational_model/kai/MaxHE/AeroVLA
PIP="https://pypi.tuna.tsinghua.edu.cn/simple"
log(){ echo "[$(date '+%F %T')] $*"; }

log "=== 1. 卸载冲突包 ==="
"$V/bin/pip" uninstall -y msgpack-python msgpack msgpack-rpc-python airsim tornado 2>&1 | tail -3

log "=== 2. 装 tornado==4.5.3 + 修复版 msgpack-rpc ==="
"$V/bin/pip" install -i "$PIP" tornado==4.5.3 2>&1 | tail -2
"$V/bin/pip" install "$C/msgpack-rpc-python-fix-msgpack-dep.zip" 2>&1 | tail -3

log "=== 3. 装 airsim==1.8.1 ==="
"$V/bin/pip" install -i "$PIP" airsim==1.8.1 --no-build-isolation 2>&1 | tail -3

log "=== 4. 强制重装 msgpack==1.1.2 ==="
"$V/bin/pip" uninstall -y msgpack-python 2>&1 | tail -1
"$V/bin/pip" install -i "$PIP" --force-reinstall msgpack==1.1.2 2>&1 | tail -3

log "=== 5. 打补丁：去掉 airsim client.py 里的 pack/unpack encoding ==="
"$V/bin/python" - <<'PYEOF'
import glob, io, re, sys
paths = glob.glob("/mnt/kai_ckp/yiding/MaxHE/envs_py/aerovla_py310/lib/python3.*/site-packages/airsim/client.py")
if not paths:
    print("找不到 airsim/client.py"); sys.exit(1)
p = paths[0]
s = io.open(p, encoding="utf-8").read()
orig = s
# 去掉 pack_encoding / unpack_encoding 参数
s = re.sub(r",\s*pack_encoding\s*=\s*'[^']*'", "", s)
s = re.sub(r",\s*unpack_encoding\s*=\s*'[^']*'", "", s)
s = re.sub(r",\s*pack_encoding\s*=\s*\"[^\"]*\"", "", s)
s = re.sub(r",\s*unpack_encoding\s*=\s*\"[^\"]*\"", "", s)
if s == orig:
    print("未找到 encoding 参数（可能已经是干净的）")
else:
    io.open(p, "w", encoding="utf-8").write(s)
    print("已打补丁:", p)
PYEOF

log "=== 6. 校验 ==="
"$V/bin/python" -c "
import airsim, msgpack, msgpackrpc, tornado
print('airsim', getattr(airsim,'__version__','?'))
print('msgpack', msgpack.version)
print('tornado', tornado.version)
"
grep -n "msgpackrpc.Client(" "$V"/lib/python3.*/site-packages/airsim/client.py | head -3
