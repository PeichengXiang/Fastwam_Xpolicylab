# 0812 XPolicyLab bench audit

- Date: 2026-08-12 (Asia/Shanghai)
- Upstream XPolicyLab base: `8b74924181eb877bb43a5da0ec2593740e0d4ebf`
- Raw data: `/personal/tjy/spark0_bench`
- Tasks: `collect_objects`, `dual_bottles_pick`, `hammer_beat`, `insert_block`, `retrieve_gap`, `stack_bowls`
- Raw episodes: 100 per task, 600 total
- Robot: `tianji_marvin_wuji` (`arm_dim=[7,7]`, `ee_dim=[20,20]`)
- Official action mode for H-RDT, RDT-1B and FastWAM adapters: `joint`
- Required run settings: 8 GPUs, per-device batch 8, global batch 64, 100000 train steps, checkpoint every 5000 steps, W&B enabled

## GPU audit at 2026-08-12 14:55 CST

- A800-12 (`192.168.156.44`): 8/8 GPUs free; assigned H-RDT.
- A800-9 (`192.168.81.241`): 8/8 GPUs free; assigned RDT-1B.
- A800-7 (`192.168.80.234`): occupied by UniT on all 8 GPUs; untouched.
- A800-8 (`192.168.81.238`): occupied by Pi-0.5 on all 8 GPUs; untouched.

## Change boundary

Data preparation and command-line/config overrides are permitted. No upstream or adapter training-code change has been made. Any required training-code edit must be approved by the user first.
