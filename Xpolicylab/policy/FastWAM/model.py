import os
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


POLICY_DIR = Path(__file__).resolve().parent
FASTWAM_ROOT = POLICY_DIR / "FastWAM"
FASTWAM_SRC = FASTWAM_ROOT / "src"


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _standardize_rgb(image: np.ndarray) -> np.ndarray:
    # The WS server has already decoded encoded colors.  ``decode_obs_images``
    # may hand us a read-only view, so only make a contiguous copy at the
    # boundary when needed; do not decode again in the adapter.
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] != (240, 320):
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    if image.shape != (240, 320, 3):
        raise ValueError(f"Expected standardized RGB shape (240, 320, 3), got {image.shape}")
    return np.ascontiguousarray(image)


def _get_instruction(obs: dict, fallback: str) -> str:
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else fallback
    if value is None:
        return fallback
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip()
    return text if text else fallback


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg.get("action_type") or "joint").strip().lower()
        if self.action_type != "joint":
            raise ValueError(
                "FastWAM EgoVLA deployment supports action_type='joint' only; "
                "EE actions require a separate IK/action-head contract."
            )
        self.env_cfg_type = self.model_cfg["env_cfg_type"]
        self.action_horizon = 1
        self.replan_steps = int(self.model_cfg.get("replan_steps") or 24)
        self.default_instruction = str(
            self.model_cfg.get("default_instruction")
            or self.model_cfg.get("prompt")
            or "follow the instruction"
        )
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self._batch_obs = {}
        self._batch_instruction = {}
        self.model = None
        self.ego_camera_mode = None
        self.runtime_summary = None
        self.action_dim = int(
            sum(self.robot_action_dim_info["arm_dim"])
            + sum(self.robot_action_dim_info["ee_dim"])
        )
        configured_action_dim = self.model_cfg.get("action_dim")
        if not _is_none_like(configured_action_dim):
            try:
                configured_action_dim = int(configured_action_dim)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"FastWAM action_dim must be an integer, got {configured_action_dim!r}"
                ) from exc
            if configured_action_dim != self.action_dim:
                raise ValueError(
                    "FastWAM action_dim does not match the registered robot schema: "
                    f"config={configured_action_dim}, schema={self.action_dim}, "
                    f"env_cfg_type={self.env_cfg_type}"
                )

        self.allow_dummy_policy = _is_true(self.model_cfg.get("allow_dummy_policy", False))
        checkpoint_path = self.model_cfg.get("checkpoint_path") or self.model_cfg.get("ckpt_setting")
        dataset_stats_path = self.model_cfg.get("dataset_stats_path")

        if self.allow_dummy_policy:
            print("[FastWAM] allow_dummy_policy=true; real checkpoint loading is skipped for debug flow only.")
            return

        if _is_none_like(checkpoint_path):
            raise FileNotFoundError("FastWAM requires checkpoint_path/ckpt_setting for real deployment.")
        if _is_none_like(dataset_stats_path):
            raise FileNotFoundError("FastWAM requires dataset_stats_path for real deployment.")

        for path in (str(FASTWAM_ROOT), str(FASTWAM_SRC)):
            if path not in sys.path:
                sys.path.insert(0, path)

        from experiments.robotwin.fastwam_policy.deploy_policy import get_model

        upstream_cfg = dict(self.model_cfg)
        upstream_cfg["ckpt_setting"] = str(Path(checkpoint_path).expanduser().resolve())
        upstream_cfg["dataset_stats_path"] = str(Path(dataset_stats_path).expanduser().resolve())
        upstream_cfg.setdefault("sim_cfg_name", "sim_robotwin.yaml")
        upstream_cfg.setdefault("sim_task", "robotwin_uncond_3cam_384_1e-4")
        if self.env_cfg_type == "ego_h1_inspire":
            from .checkpoint_runtime import resolve_runtime
            from .exact_egovla_image import build_egovla_image_tensor

            checkpoint_cfg, self.ego_camera_mode, self.runtime_summary = resolve_runtime(
                upstream_cfg["ckpt_setting"], upstream_cfg["dataset_stats_path"],
                self.model_cfg.get("task_name"), self.action_dim,
            )
            upstream_cfg["checkpoint_config"] = checkpoint_cfg
            upstream_cfg["image_tensor_builder"] = build_egovla_image_tensor
            requested_horizon = self.model_cfg.get("action_horizon")
            if not _is_none_like(requested_horizon) and int(requested_horizon) != self.runtime_summary["action_horizon"]:
                raise ValueError("EgoVLA action_horizon must match the checkpoint training horizon")
            print("[FastWAM runtime contract] " + json.dumps(self.runtime_summary, sort_keys=True), flush=True)
        self.model = get_model(upstream_cfg)
        self.action_horizon = int(self.model.action_horizon)
        self.replan_steps = int(self.model.replan_steps)

    def _encode_obs_for_fastwam(self, obs: dict) -> dict:
        if not isinstance(obs, dict):
            raise TypeError(f"FastWAM expects an observation dict, got {type(obs).__name__}")
        vision = obs.get("vision")
        if not isinstance(vision, dict):
            raise KeyError("FastWAM observation must contain a 'vision' mapping")

        def camera_rgb(name: str) -> np.ndarray:
            camera = vision.get(name)
            if not isinstance(camera, dict) or "color" not in camera:
                raise KeyError(f"FastWAM observation is missing vision[{name!r}]['color']")
            image = np.asarray(camera["color"])
            if self.ego_camera_mode is not None:
                if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError(f"EgoVLA expects decoded uint8 HWC RGB, got {image.shape}/{image.dtype}")
                return np.ascontiguousarray(image)
            return _standardize_rgb(image)

        vector = pack_robot_state(
            obs,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
            state_type="state",
        ).astype(np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.action_dim:
            raise ValueError(
                "FastWAM packed state shape mismatch: "
                f"expected ({self.action_dim},), got {vector.shape}"
            )
        head = camera_rgb("cam_head")
        # EgoVLA is single-view except tasks that recorded real hand cameras.
        # Missing wrists are black frames, matching the training videos.
        if self.ego_camera_mode == "black_wrist":
            left = np.zeros_like(head)
            right = np.zeros_like(head)
        else:
            left, right = camera_rgb("cam_left_wrist"), camera_rgb("cam_right_wrist")
        adapted = {
            "observation": {
                "head_camera": {"rgb": head},
                "left_camera": {"rgb": left},
                "right_camera": {"rgb": right},
            },
            "joint_action": {"vector": vector},
        }
        return adapted

    def update_obs(self, obs):
        self.last_obs = self._encode_obs_for_fastwam(obs)
        self.last_instruction = _get_instruction(obs, self.default_instruction)
        # A single-env call starts a fresh stream; discard any stale batch cache.
        self._batch_obs.clear()
        self._batch_instruction.clear()

    def update_obs_batch(self, obs_list):
        if obs_list is None or not isinstance(obs_list, (list, tuple)) or not obs_list:
            raise ValueError("update_obs_batch received an empty observation list.")
        batch_obs = {}
        batch_instruction = {}
        for obs in obs_list:
            if not isinstance(obs, dict) or "env_idx" not in obs:
                raise ValueError("Each FastWAM batch observation must be a dict with env_idx.")
            env_idx = int(obs["env_idx"])
            if env_idx in batch_obs:
                raise ValueError(f"Duplicate env_idx in FastWAM batch: {env_idx}")
            batch_obs[env_idx] = self._encode_obs_for_fastwam(obs)
            batch_instruction[env_idx] = _get_instruction(obs, self.default_instruction)
        self._batch_obs = batch_obs
        self._batch_instruction = batch_instruction
        self.last_obs = None
        self.last_instruction = self.default_instruction

    def _zero_actions(self):
        dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        zeros = np.zeros((self.replan_steps, dim), dtype=np.float32)
        return unpack_robot_state(zeros, self.action_type, self.robot_action_dim_info, source_type="obs")

    def _infer_actions(self, obs, instruction):
        if self.allow_dummy_policy:
            return self._zero_actions()
        if obs is None:
            raise ValueError("No observation is available. Call update_obs() before get_action().")
        action_chunk = self.model._infer_action_chunk(obs, instruction)
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        if action_chunk.ndim != 2 or action_chunk.shape[1] != self.action_dim:
            raise ValueError(
                "FastWAM checkpoint returned an invalid action shape: "
                f"expected [T,{self.action_dim}], got {action_chunk.shape}"
            )
        if action_chunk.shape[0] == 0:
            raise ValueError("FastWAM checkpoint returned an empty action chunk")
        if not np.isfinite(action_chunk).all():
            raise ValueError("FastWAM checkpoint returned non-finite action values")
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        action_chunk = action_chunk[:n_exec]
        return unpack_robot_state(action_chunk, self.action_type, self.robot_action_dim_info, source_type="obs")

    def get_action(self):
        return self._infer_actions(self.last_obs, self.last_instruction)

    def get_action_batch(self, env_idx_list=None):
        if not self._batch_obs:
            raise ValueError("No batch observation is available. Call update_obs_batch() first.")
        if env_idx_list is None:
            env_idx_list = sorted(self._batch_obs)
        if not isinstance(env_idx_list, (list, tuple, np.ndarray)):
            raise TypeError(
                "get_action_batch expects env_idx_list as a sequence or None, "
                f"got {type(env_idx_list).__name__}"
            )
        env_idx_list = [int(env_idx) for env_idx in env_idx_list]
        if not env_idx_list:
            raise ValueError("get_action_batch received an empty environment index list")
        if len(set(env_idx_list)) != len(env_idx_list):
            raise ValueError(f"get_action_batch received duplicate env indices: {env_idx_list}")
        missing = [env_idx for env_idx in env_idx_list if env_idx not in self._batch_obs]
        if missing:
            raise KeyError(f"No batch observation stored for env indices: {missing}")
        return [
            self._infer_actions(self._batch_obs[env_idx], self._batch_instruction[env_idx])
            for env_idx in env_idx_list
        ]

    def reset(self):
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self._batch_obs.clear()
        self._batch_instruction.clear()
        if self.model is not None:
            self.model.reset()
