# FastWAM training run: 2026-08-12 `/mnt` joint54 50k

- Machine: A800-8 (`192.168.81.238`), eight A800 80 GB GPUs
- Started: 2026-08-12 21:14 CST
- Source: 12 LeRobot v2.1 shards converted from `/mnt/xspark-data/tjy/spark0_bench`
- Mode: official XPolicyLab FastWAM joint training; Action DiT trainable, other model components frozen as upstream
- Source episodes/frames: 600 / 151,410 across six tasks
- Official split: 148,575 training frames / 2,835 validation frames
- Batch: 8 per GPU, 64 global, gradient accumulation 1
- Optimizer steps: 50,000; checkpoint period: 5,000
- Seed: 42; mixed precision: bf16; DeepSpeed ZeRO stage 1
- Output: `chpt/20260812_mnt_joint_bs64_s42_50k`
- Log: `logs/train_20260812_mnt_joint_bs64_s42_50k.log`
- W&B: `https://wandb.ai/peichengxiang773-hkust/fast-wam/runs/pfbi60be`

Startup validation observed all eight GPU workers, the 12 new dataset shards, the combined statistics and six-prompt text cache, `max_steps=50000`, and live finite action/video losses.

The previous run and complete `step_005000` checkpoint remain under `chpt/20260812_joint_bs64_s42`.
