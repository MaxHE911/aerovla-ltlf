# baseline-v0 — verified AeroVLA baseline snapshot

This tag is a verbatim snapshot of the working baseline directories on the lab server
(`aibox-r155db332cb6-669f64bffd-4fnw6`, taken 2026-11-20).

| item | value |
|---|---|
| source (code) | `/mnt/world_foundational_model/kai/MaxHE/AeroVLA` @ upstream git `b87aaa7` |
| source (aux)  | `/mnt/world_foundational_model/kai/MaxHE/TravelUAV` |
| verbatim copies | `AeroVLA/`, `TravelUAV/`, `ops/` |
| removed from snapshot | nested `.git` directories |
| data/model dirs | NOT copied; they are symlinks into `/mnt/kai_ckp/yiding/MaxHE/` and stay that way |

The original directories are never modified by this project; all development happens in
this repository on branch `dev`.

`ops/` holds copies of the helper scripts that live on the data disk
(`smoke_test.sh`, `vulkan_adapter.py`, `hfget.py`, `setup_rpc.sh`, `extract_prep.sh`).
