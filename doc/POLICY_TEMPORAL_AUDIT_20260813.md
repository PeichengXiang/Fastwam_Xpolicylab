# Four-policy temporal and dimensional audit (2026-08-13)

| Policy | Source-to-training time contract | Dimension result | Status |
|---|---|---|---|
| H-RDT | Official reader was one frame late; corrected at the data boundary so image/state are at `t` and first target is source `action[t]` | 54/54 retained | Corrected and fresh 8-GPU run started |
| FastWAM | LeRobot stores `state[t]`, `action[t]`; official loader uses video/state `t..t+32` and actions `t..t+31`, so each action controls the following transition | 54/54 retained in dataset, processor, config, and action expert | No temporal defect found; current run unchanged |
| RDT-1B | Reader uses state/image `t` and source action chunk starting at `t`; no one-frame shift | Corrected mapping retains 54/54; train/deploy round trip max error 0 | Old invalid run stopped; corrected fresh 8-GPU run active |
| RDP | Converter stores observations/actions at identical source indices; upstream sequence sampler keeps the same window origin | Converted action width 54, but policy requires tactile inputs absent from this dataset | Training remains stopped as requested |

## FastWAM evidence

- Conversion validation compared all 151,410 numeric rows across 600 episodes to source HDF5.
- Packed order is `left arm 7 + left hand 20 + right arm 7 + right hand 20` for both `observation.state` and `action`.
- Active run config has `raw_shape=54`, transformed shape 54, `action_output_dim=54`, and `proprio_output_dim=54` for train and validation.
- The official loader explicitly documents action `[num_frames-1]` starting at `t0` and proprio/video `[num_frames]` starting at `t0`; this matches the Spark0 contract `action[t] -> state[t+1]`.
- The 12 current LeRobot shards were revalidated against all 600 HDF5 episodes
  (151,410 frames), with exact numeric equality for both state and action.
- The official loader was instantiated against all 12 shards and returned
  finite action/proprio tensors of shape `(32,54)` for every shard.
- The active run was at step 28,290/80,000 with action loss about 0.009-0.014,
  all eight GPUs active, and no traceback/OOM/NCCL failure.

## Responsibility summary

- H-RDT's Spark0-incompatible index arithmetic and RDT-1B's lossy/14-D
  mappings are upstream XPolicyLab code from commit `3e6cd252...`; the local
  setup did not introduce those lines.
- The local work nevertheless failed to detect these semantic defects before expensive training. Shape-only smoke tests and falling losses were not sufficient. Future starts must pass temporal identity, dimension coverage, and train/deploy round-trip gates.
