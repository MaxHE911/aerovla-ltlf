#!/usr/bin/env python3
"""从 hf-mirror 下载指定文件，带断点续传 + 429/断流重试 + 大小校验。"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_ENDPOINT = "https://hf-mirror.com"
FILE_ENDPOINTS = ["https://hf-api.gitee.com", "https://hf-mirror.com"]
UA = {"User-Agent": "Mozilla/5.0 hfget/1.0"}


def api_tree(repo, kind):
    url = "%s/api/%s/%s/tree/main?recursive=true" % (
        API_ENDPOINT, "datasets" if kind == "dataset" else "models", repo)
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.load(urllib.request.urlopen(req, timeout=90))
        except Exception as e:
            print("  [api] 重试 %d: %s" % (attempt + 1, e), flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("api failed: " + repo)


def size_of(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def fetch(url, out, expected):
    """下载到 out，支持续传；返回 True 表示大小正确。"""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(40):
        have = size_of(out)
        if expected and have == expected:
            return True
        if expected and have > expected:
            os.remove(out)
            have = 0
        req = urllib.request.Request(url, headers=dict(UA))
        if have:
            req.add_header("Range", "bytes=%d-" % have)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(out, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
            if not expected or size_of(out) >= expected:
                return True
            print("    断流，已有 %d/%d，续传" % (size_of(out), expected), flush=True)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(30 + 30 * attempt, 300)
                print("    429 限流，等 %ds" % wait, flush=True)
                time.sleep(wait)
            elif e.code == 404:
                print("    HTTP 404（路径不对），跳过", flush=True)
                return False
            elif e.code in (416,):
                return bool(expected and size_of(out) == expected)
            else:
                print("    HTTP %s，重试" % e.code, flush=True)
                time.sleep(10)
        except Exception as e:
            print("    %s，5s 后重试" % type(e).__name__, flush=True)
            time.sleep(5)
    return False


def main():
    spec = json.load(open(sys.argv[1]))
    for job in spec:
        repo, kind, dest, patterns = job["repo"], job["type"], job["dest"], job["include"]
        print("=" * 70, flush=True)
        print("[%s] %s -> %s" % (kind, repo, dest), flush=True)
        files = [f for f in api_tree(repo, kind) if f.get("type") == "file"]
        if patterns:
            files = [f for f in files
                     if any(f["path"] == p or f["path"].startswith(p)
                            for p in patterns)]
        total = sum(f["size"] for f in files)
        print("  选中 %d 个文件，合计 %.2f GB" % (len(files), total / 1e9), flush=True)
        done = 0
        t0 = time.time()
        for i, f in enumerate(files, 1):
            rel = ("datasets/" + repo) if kind == "dataset" else repo
            url = None
            out = os.path.join(dest, f["path"])
            if size_of(out) == f["size"]:
                done += f["size"]
                print("  [%d/%d] 跳过(已完整) %s" % (i, len(files), f["path"]), flush=True)
                continue
            print("  [%d/%d] %.2f GB  %s" % (i, len(files), f["size"] / 1e9, f["path"]), flush=True)
            ok = False
            for mbase in FILE_ENDPOINTS:
                url = "%s/%s/resolve/main/%s" % (mbase, rel, f["path"])
                print("        via %s" % mbase, flush=True)
                if fetch(url, out, f["size"]):
                    ok = True
                    break
                print("        换镜像", flush=True)
            if ok:
                done += f["size"]
                el = time.time() - t0
                print("       OK  累计 %.1f/%.1f GB  平均 %.1f MB/s"
                      % (done / 1e9, total / 1e9, done / el / 1e6 if el else 0), flush=True)
            else:
                print("       FAILED %s" % f["path"], flush=True)
        print("  [完成] %s  用时 %.0f 分钟" % (repo, (time.time() - t0) / 60), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
