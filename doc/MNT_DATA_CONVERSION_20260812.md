# FastWAM `/mnt` data conversion (2026-08-12)

## Source and output

- Source: `/mnt/xspark-data/tjy/spark0_bench`
- Format: standard LeRobot v2.1, joint state/action (54 dimensions), RGB videos
- Output: 12 datasets under `/personal/xiangpc/0812_Xpolicylab_bench/FastWAM/data`
- Sharding: two 50-episode shards for each of six tasks (`ep000_049`, `ep050_099`)
- Total: 600 episodes, 151,410 frames, 1,800 videos

Tasks: `collect_objects`, `dual_bottles_pick`, `hammer_beat`, `insert_block`, `retrieve_gap`, and `stack_bowls`.

## Validation

- Every state and action row was compared exactly with the source HDF5 data.
- Every video was checked for frame count and RGB ordering using its first, middle, and last frames.
- Official FastWAM statistics were generated at `data/spark0_mnt20260812_fastwam_6tasks_v21_joint54/dataset_stats.json`.
- The official per-dataset 1% validation split selects one episode from each shard: 588 training episodes / 148,575 frames and 12 validation episodes / 2,835 frames.
- Six unique task prompts were encoded in the official text embedding cache.
- The official Hydra `RobotVideoDataset` train and validation loaders sampled all 12 shards successfully. Observed sample shapes: video `(3, 9, 384, 320)`, action `(32, 54)`, proprioception `(32, 54)`.

Logs:

- `logs/wait_validate_calc_stats_shards_20260812.log`
- `logs/precompute_mnt_fastwam_text_cache_20260812.log`
- `logs/validate_mnt_fastwam_official_loader_20260812.log`

## Training configuration

The new official joint-mode run uses eight GPUs, per-GPU batch size 8 (global batch size 64), 50,000 optimizer steps, and checkpoints every 5,000 steps. Its output directory and W&B run name include `20260812_mnt_joint_bs64_s42_50k`.

The earlier converted datasets, checkpoints, and canonical data links are retained for rollback.
