# FastWAM 80k extension (2026-08-13)

- User-requested total optimizer steps: 80,000.
- New checkpoint interval: 10,000 optimizer steps.
- Resume source: latest complete full-state checkpoint (`step_025000` at switch time), including model, optimizer, scheduler, DeepSpeed, dataloader position and RNG state.
- Output directory renamed to `chpt/20260812_mnt_joint_bs64_s42_80k`; historical 5k checkpoints are retained, and subsequent periodic checkpoints follow the new 10k interval.
- The cosine scheduler state is adjusted only to extend its terminal step from 50k to 80k while preserving the current learning rate; no training-source change is made.
- Official FastWAM trainable/frozen module selection and joint54 data remain unchanged; batch remains 8/GPU and 64 global.
- W&B remains online under a resume-specific 80k run name.

## Startup verification

- Restored `step_025000` model, optimizer, scheduler, dataloader sampler and RNG state successfully.
- The saved cosine scheduler `T_max` was extended from 47,500 to 77,500; resumed LR remained continuous at approximately `5.45e-5` and now decays to its minimum at global step 80k.
- Verified all eight ranks, DeepSpeed ZeRO-1, per-GPU batch 8 and global batch 64.
- First observed resumed progress: 25,050/80,000, loss `0.1002`, action loss `0.0115`, video loss `0.0887`.
- W&B: `https://wandb.ai/peichengxiang773-hkust/fast-wam/runs/gwm4yc0p`
- Log: `logs/train_20260813_mnt_joint_bs64_s42_80k_resume25k.log`
