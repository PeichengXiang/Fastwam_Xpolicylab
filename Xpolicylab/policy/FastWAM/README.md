# FastWAM

**Contributor:** RoboDojo Team | **Paper:** Fast-WAM: Do World Action Models Need Test-time Future Imagination? | **arXiv:** https://arxiv.org/abs/2603.16666 | **Original code:** https://github.com/yuantianyuan01/FastWAM

`FastWAM` adapts the Fast-WAM world-action model to XPolicyLab/RoboDojo. Integration scripts live at this directory level; the vendored upstream implementation lives in `FastWAM/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Read `INSTALLATION.md` before first use; it covers setup that `install.sh` cannot fully express, such as external checkpoints, system packages, manual fallback steps, and multi-environment runtime notes.

```bash
cd XPolicyLab/policy/FastWAM
bash install.sh
conda activate <policy_env>  # e.g. fastwam
```

## Data Processing

No top-level `process_data.sh`. Training consumes a prepared LeRobot v2.1 dataset directly (see Training below); use the upstream FastWAM tooling under `FastWAM/` for data preparation.

## Training

```bash
cd XPolicyLab/policy/FastWAM
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]

# Example: train a cotrain run on four GPUs (the upstream model is large; multi-GPU is recommended)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name, the full run-directory name, or a path to a checkpoint directory. Training expects a prepared LeRobot v2.1 dataset under `data/<dataset_id>/lerobot/` with `data/<dataset_id>/dataset_stats.json`, a matching T5 text embedding cache under `FastWAM/data/text_embeds_cache/xpolicylab/<dataset_id>/`, and the ActionDiT backbone at `FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`. `<dataset_id>` defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>` (override with `FASTWAM_DATASET_ID`), and `train.sh` prints the upstream commands that generate the text cache and backbone when they are missing. The process count is inferred from a comma-separated `gpu_id` unless passed as the optional 7th argument `num_gpus`. The wrapper defaults to `FASTWAM_BATCH_SIZE=8`; you may still need to lower it or increase the GPU count depending on available memory. `train.sh` sets `DIFFSYNTH_MODEL_BASE_PATH` to `FastWAM/checkpoints` automatically.

## Evaluation

```bash
cd XPolicyLab/policy/FastWAM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `action_dim`, `checkpoint_path`, `dataset_stats_path`, `sim_cfg_name`, `sim_task`, `device`, `mixed_precision`, `action_horizon`, `replan_steps`, `num_inference_steps`, `sigma_shift`.

Optional policy-specific environment overrides used by the scripts: `FASTWAM_DATASET_ID`, `FASTWAM_BATCH_SIZE`, `FASTWAM_GRADIENT_ACCUMULATION_STEPS`, `FASTWAM_NUM_WORKERS`, `FASTWAM_NUM_EPOCHS`, `FASTWAM_CKPT_SETTING`, `FASTWAM_CKPT_ROOT`, `FASTWAM_CHECKPOINT_PATH`, `FASTWAM_DATASET_STATS_PATH`, `FASTWAM_ALLOW_DUMMY_POLICY`.

## Notes

- Use the same `action_type` for training and evaluation. The reference FastWAM path follows XPolicyLab's `pack_robot_state` / `unpack_robot_state` helpers directly and does not add policy-local `ee` pose conversion.

## EgoVLA 38-D evaluation

The converter intentionally excludes exactly 100 Deprecated episodes: the 50
Insert-Cans/Left_Right_Deprecated files and the 50
Sort-Cans/Single_hand_first_deprecated files. They are absent from both
the converted dataset and the training split.

The EgoVLA run uses the active (non-Deprecated) episodes only: two 7-DoF arms
and two 12-DoF Inspire hands (`env_cfg_type=ego_h1_inspire`, packed action
dimension 38).  FastWAM currently supports `action_type=joint` only.  The
server consumes RGB arrays after XPolicyLab's websocket decoder and maps them
to FastWAM's head/left-wrist/right-wrist composite without decoding in
`model.py`.

For EgoVLA, the adapter fails closed unless the selected weight has a
checkpoint-owned `fastwam_inference_contract.json`. The contract binds the
weight, saved training config, normalization statistics, data manifests,
38-D layout, action horizon, image geometry, and per-task camera provenance.
It also forces inference to load the checkpoint's saved model/processor config
instead of the mutable SParkArena defaults in `deploy.yml`.

`Insert-And-Unload-Cans` uses genuine wrist cameras. The other eleven tasks
are single-view, so both wrist inputs are black frames. Do not copy the head
image into those slots. Native images follow the official
EgoVLA 384x384 boundary and then the checkpoint's validation transforms.

The training output is outside this policy's legacy `checkpoints/` directory.
Pass the run directory (or an explicit `step_XXXXXX.pt`) and the converted
dataset statistics when evaluating:

```bash
export FASTWAM_CHECKPOINT_PATH=/personal/xiangpc/0812_Xpolicylab_bench/FastWAM/chpt/20260902_egovla_joint38_bs64_s42_80k/checkpoints/weights/step_080000.pt
export FASTWAM_DATASET_STATS_PATH=/personal/xiangpc/0812_Xpolicylab_bench/FastWAM/data/EgoVLA_benchmark_fastwam_v21_joint38_cmd/dataset_stats.json
export FASTWAM_MODEL_BASE_PATH=/personal/xiangpc/0812_Xpolicylab_bench/FastWAM/pretrain_model
export EVAL_MAIN_ROOT="/personal/xiangpc/EgoVLA benchmark"  # simulator/env-config root
cd /personal/xiangpc/0812_Xpolicylab_bench/FastWAM/Xpolicylab/policy/FastWAM
bash eval.sh EgoVLA Humanoid-Push-Box-v0 \
  20260902_egovla_joint38_bs64_s42_80k ego_h1_inspire joint 0 \
  0 1 fastwam /personal/miniconda3/envs/fastwam
```

For a wiring-only check before a checkpoint exists, set
`EVAL_ENV_TYPE=debug FASTWAM_ALLOW_DUMMY_POLICY=true`.  The dummy path checks
the 38-D action dictionary and websocket lifecycle; it is not a quality
evaluation.  Set `FASTWAM_EVAL_BATCH=true` to exercise the batch RPC, and set
`DEBUG_OBS_ENCODED=1` to exercise server-side JPEG decoding.  A real checkpoint
must be available before claiming model inference or task success.

The setup script resolves checkpoints with the shared XPolicyLab resolver. It
searches an explicit `FASTWAM_CHECKPOINT_PATH`, then the formal model-root
`chpt/` tree, then the policy-local `checkpoints/` tree. `FASTWAM_CKPT_ROOT`
can point at another root; `FASTWAM_CHECKPOINT_NUM=10000` pins a numbered step.
If a run directory is supplied, the newest numeric `step_*.pt` is selected.
`FASTWAM_DATASET_STATS_PATH` should point at the matching 38-D
`dataset_stats.json` (the converted dataset path above is the fallback).

Exact offline smoke commands (no simulator and no checkpoint load):

```bash
# single-env/plain observation path
EVAL_ENV_TYPE=debug FASTWAM_ALLOW_DUMMY_POLICY=true \
  FASTWAM_EVAL_BATCH=false DEBUG_OBS_ENCODED=0 \
  bash eval.sh EgoVLA debug smoke ego_h1_inspire joint 42 "" "" fastwam fastwam

# batch RPC + JPEG/bytes decode path
EVAL_ENV_TYPE=debug FASTWAM_ALLOW_DUMMY_POLICY=true DEBUG_OBS_ENCODED=1 \
  bash eval.sh EgoVLA debug smoke ego_h1_inspire joint 42 "" "" fastwam fastwam
```

The batch adapter receives indices as `model_client.call(..., obs=env_idx_list)`
and returns one equally sized action chunk per active environment. `reset()`
and each single-env update clear the previous batch cache, preventing stale
state when the evaluator changes modes between episodes.

For a new EgoVLA training run, the dedicated task config defaults to
`train_sampling=task_uniform`: each task has equal draw probability and frames
are uniform within that task. Historical behavior remains available as
`train_sampling=frame_uniform`. A saved optimizer state records the sampling
contract and refuses a silent mode/data/world-size change on resume; start a
new run when changing the mode. `data_scripts/launch_egovla_fastwam_8gpu.sh`
defaults to a new output name and accepts `SAMPLING_MODE=task_uniform` or
`SAMPLING_MODE=frame_uniform`.
