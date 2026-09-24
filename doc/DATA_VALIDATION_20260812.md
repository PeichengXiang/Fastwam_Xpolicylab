# FastWAM data validation (2026-08-12)

`data_scripts/validate_lerobot_joint54.py` compared every numeric row of all
600 LeRobot v2.1 episodes against the six-task source HDF5 collection.

Result:

```text
FASTWAM_LEROBOT_VALIDATION_OK episodes=600 frames=151410 tasks=6 mode=joint action_dim=54 state_dim=54 fps=25
```

Verified invariants:

- exactly 100 episodes per task and 151,410 total frames;
- `observation.state` and `action` are float32[54];
- every state/action row exactly equals source HDF5 packed as
  left arm(7), left hand(20), right arm(7), right hand(20);
- episode/frame/task/global indices, lengths, timestamps, and 25 Hz frequency;
- the three official FastWAM camera fields are present.

The HDF5 fourth view (`cam_third_view`) is not included because the official
XPolicyLab FastWAM RoboTwin task is explicitly a three-camera configuration.

