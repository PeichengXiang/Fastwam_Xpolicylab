# FastWAM training monitor (2026-08-13)

## 18:39 CST snapshot

- Machine: A800_8 (`192.168.81.238`), 8/8 GPUs.
- Run: Spark0 joint54 training resumed from step 25,000.
- Progress: step 38,260 / 80,000.
- Configuration: batch size 8/GPU, global batch size 64, checkpoint every 10,000 steps.
- Latest periodic checkpoint: step 30,000 under
  `/personal/xiangpc/0812_Xpolicylab_bench/FastWAM/chpt/20260812_mnt_joint_bs64_s42_80k`.
- Recent 200 logged points:
  - total loss: min 0.0533, median 0.07765, max 0.1143;
  - action loss: min 0.0044, median 0.0080, max 0.0137;
  - video loss: min 0.0447, median 0.0695, max 0.1062.
- Every reported loss was finite.
- GPU state: all eight ranks active, about 52.7 GB/GPU, 87-94% utilization.
- No traceback, CUDA OOM, or child-process failure was found.
- W&B: `https://wandb.ai/peichengxiang773-hkust/fast-wam/runs/gwm4yc0p`

Conclusion: FastWAM is training normally. The next expected periodic checkpoint
is step 40,000.

## Dataset split and normalization audit

- The official `robotwin.yaml` uses `val_set_proportion: 0.01` for both train
  and validation, with `is_training_set: true/false` respectively.
- `BaseLerobotDataset` applies the split independently to every input shard.
  Each current shard has 50 episodes, so `int(50 * 0.99) = 49` episodes are
  assigned to training and one to validation.
- With seed 42, the held-out local episode index is 8 in each of the 12 shards.
  This gives exactly 588 train episodes and 12 validation episodes.
- The 12 held-out episode lengths sum to 2,835 frames; the remaining episodes
  sum to 148,575 frames. The training log independently reports
  `Train/val dataset size: 148575/2835`, and the total is exactly all 151,410
  converted Spark0 frames.
- Per task, the two held-out lengths are: collect 243+290, dual-bottles
  151+138, hammer 96+64, insert 260+319, retrieve 332+304, and stack 415+223.
  Every task therefore retains 98 train episodes and two validation episodes.
- The active checkpoint configuration points to the intended joint54
  `dataset_stats.json`. All state arrays have shape 54 and all action arrays
  have shape `(32, 54)`; every value is finite and all expected dimensions are
  populated.

Conclusion: the 588 count is the official validation split, not data loss or a
short-episode filter. No FastWAM code or configuration change is needed.

## 19:36 CST heartbeat and 40k checkpoint gate

- Training reached step 40,000 and validation completed with
  `val_loss=0.283158`, `infer_psnr=28.1651`, `infer_ssim=0.9063`,
  `action_l2=0.0044`, and `action_l1=0.0292`.
- The dated run directory contains a complete `step_040000` resumable state:
  eight optimizer shards, eight random-state files, model state, scheduler,
  trainer state, `zero_to_fp32.py`, and the `latest` marker.
- The exported `step_040000.pt` is 12,042,305,017 bytes, matching the expected
  weight artifact size. The model-state file is 13,452,725,897 bytes and all
  eight optimizer shards are about 9.03 GB each.
- Training resumed after saving and reached step 40,010 with finite
  `loss=0.05801`, `loss_action=0.007835`, and `loss_video=0.05017`.
- W&B independently reports the run as `running` at step 40,010 and contains
  the same finite metrics. All eight GPUs remain allocated at about 52.7 GB.
- No traceback, CUDA OOM, NCCL error, child-process failure, or non-finite loss
  was found.

Conclusion: the FastWAM 40k validation, resumable state, exported weight, and
post-save continuation gates all passed. The next required checkpoint is 50k.

## 20:04 CST heartbeat

- Live training reached step 40,890. W&B independently reported `running` at
  the same step with finite `loss=0.105782`, `loss_action=0.008371`,
  `loss_video=0.097410`, and learning rate `3.4726e-5`.
- The complete 40k state and 12,042,305,017-byte exported weight remain present.
- Eight GPU compute processes were present; all cards retained about 52.7 GB
  and sampled at 84-96% utilization.
- No traceback, OOM, NCCL error, child failure, or non-finite value was found.

Conclusion: FastWAM remains healthy after the 40k checkpoint and continues
toward 50k.

## 23:42 CST status and inference-consistency warning audit

- Live training reached step 47,520; W&B reported `running` at step 47,610 with
  finite total/action/video losses `0.083881/0.006886/0.076994` and learning
  rate `2.5765e-5`.
- All eight GPUs were active at 100% in the snapshot, each holding about
  52.7 GB. The complete checkpoint series through 40k remains intact; 50k is
  the next required save.
- The log contains one occurrence in the entire run of the official validation
  warning that joint inference and action-only inference differ. The maximum
  absolute difference was `0.015625` at step 47,500.
- Source inspection shows this is an `allclose(atol=1e-2, rtol=1e-2)` diagnostic
  between two inference paths, not the training target or optimizer. Training
  resumed normally, validation was finite (`val_loss=0.2472`, action L2
  `0.0095`), and no second occurrence was found. No source change was made.
- No traceback, OOM, NCCL error, child failure, or non-finite value was found.

Conclusion: FastWAM training remains healthy. The isolated validation-path
precision warning is documented and will remain under observation.
