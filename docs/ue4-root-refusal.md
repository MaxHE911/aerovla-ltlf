## 为什么必须 `LD_PRELOAD` 一个"假 uid"库（UE4 拒绝 root 运行）

跑仿真时如果看到

```
Refusing to run with the root privileges.
```

**这不是本项目的代码报错，也不是 AirSim/Python 的问题，而是 UE4 游戏二进制自己的启动检查**。
现场复现（2026-11-20，本机 `envs/closeloop_envs/`）：

| 命令 | 结果 |
|---|---|
| `sh ModernCityMap.sh -RenderOffscreen -NoSound -NoVSync` | 打印 `Refusing to run with the root privileges.` 后立即退出 |
| 同上 + `LD_PRELOAD=.../libfakerootuid.so` | 正常初始化并挂载 `ModernCityDowntown-LinuxNoEditor.pak` |

定位过程：

```bash
E=/mnt/kai_ckp/yiding/MaxHE/envs/closeloop_envs
head -6 $E/ModernCityMap.sh                      # 启动器只是 chmod +x 后 exec 二进制，没有 root 判断
grep -a -m1 -o "Refusing to run with the root privileges" \
     $E/ModernCityDowntown/Binaries/Linux/ModernCityDowntown   # 字符串在 UE4 二进制里
```

即：`<Scene>.sh` → `Binaries/Linux/<Scene>`，检查在二进制内部（`geteuid() == 0` 就拒绝）。
本机只有 root 账号，所以必须让这个进程"以为"自己不是 root。

### 解决办法：预加载 uid 伪造库

`ops/sim-runtime/` 下是这个库的**源码 + 构建产物**（原来只有数据盘里一个来路不明的 `.so`，
现在可审计、可重建）：

```bash
cd ops/sim-runtime
gcc -shared -fPIC -O2 -o lib/libfakerootuid.so fakerootuid.c     # 只覆盖 getuid/geteuid/getgid/getegid
LD_PRELOAD=$PWD/lib/libfakerootuid.so python3 -c "import os;print(os.geteuid())"   # -> 1000
```

正确用法：**挂在仿真服务端**（它是 UE4 的父进程，环境变量会被子进程继承），不要挂在 Xvfb 上：

```bash
unset LD_PRELOAD                                      # 起 Xvfb 前必须清掉，否则 Xvfb 起不来
Xvfb :99 -screen 0 1280x720x24 -nolisten tcp &
export DISPLAY=:99
cd AeroVLA/airsim_plugin
LD_PRELOAD=/abs/path/lib/libfakerootuid.so \
  python -u AirVLNSimulatorServerTool.py --port 30000 --gpus 0 --root_path /mnt/kai_ckp/yiding/MaxHE/envs
```

`ops/run_ltlf_eval.sh`、`ops/run_baseline_eval.sh`、`ops/smoke_baseline.sh` 已经自动做这件事
（优先用仓库内 `ops/sim-runtime/lib/libfakerootuid.so`，找不到再退回数据盘的那份，并在 driver log 里打印
`fakeroot shim: <path>`）。

### 三个常见的"明明加了还是报错"原因

1. **只给评测进程加了、没给服务端加**：UE4 是 `AirVLNSimulatorServerTool.py` 的子进程，
   必须挂在服务端那条命令上（或 `export LD_PRELOAD` 后再启动服务端）。
2. **`LD_PRELOAD` 被 `sudo`/`su -`/登录 shell 清掉**：不需要 sudo（本来就是 root），别用会重置环境的包装。
3. **把它全局 export 到 `~/.bashrc`**：Xvfb 会因 uid 变成 1000 而起不来。只在启动服务端那一行加。

> 注意：这个库**只伪造返回值**，不会真的降低权限。它不会改变文件属主、capability 或任何实际访问控制。

---

## 附：同日发现的另一件事（与 root 无关，但会让"卡住"很像环境问题）

`AeroVLA/src/model_wrapper/aerovla_wrapper_ui.py:22` 把 `base_model_path` 写死为 `"./openvla-7b"`，
**`--model_base` 传什么都不生效**（`--model_path` 才生效，管 442 MB 的 adapter）。
该路径是 pfs 上的软链接，pfs 拥塞时单次模型加载可卡 8 分钟以上（进程处于 `cxiWaitEventWait`）。

`ops/sim-runtime/with_shm_model.sh` 在**私有 mount namespace** 里把 RAM 副本
（`/dev/shm/openvla-7b`）bind-mount 到软链接的解析目标，从而让 `./openvla-7b` 走内存；
磁盘上不创建/删除/修改任何文件（`git status -- AeroVLA/` 保持干净），也不影响其他用户。
三个 driver 都已接上。
