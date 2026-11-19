#!/usr/bin/env python3
"""物理 GPU 索引 -> UE4 该用的 Vulkan 适配器索引（按 deviceUUID 匹配 nvidia-smi）。"""
import re, subprocess, sys

def die(m):
    sys.stderr.write(m + "\n"); sys.exit(1)

gpu = str(sys.argv[1]).strip()
smi = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", gpu],
                     capture_output=True, text=True)
if smi.returncode != 0 or not smi.stdout.strip():
    die("nvidia-smi failed: " + smi.stderr)
target = smi.stdout.strip().splitlines()[0].strip().replace("GPU-", "").lower()

vi = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True)
if vi.returncode != 0:
    die("vulkaninfo failed: " + vi.stderr[-500:])

blocks, cur = {}, None
for line in vi.stdout.splitlines():
    m = re.match(r"\s*GPU(\d+)\s*:\s*$", line)
    if m:
        cur = int(m.group(1)); continue
    m = re.match(r"\s*deviceUUID\s*=\s*([0-9a-fA-F-]+)\s*$", line)
    if m and cur is not None:
        blocks[cur] = m.group(1).lower()
for idx in sorted(blocks):
    if blocks[idx] == target:
        print(idx); sys.exit(0)
die("no adapter for gpu %s (uuid %s); found %s" % (gpu, target, blocks))
