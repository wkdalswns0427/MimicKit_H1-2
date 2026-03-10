#!/usr/bin/env python3
"""
Generate a simple static H1-2 standing motion in MimicKit format.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "mimickit"))

from anim.motion import LoopMode, Motion
from anim.mjcf_char_model import MJCFCharModel
from util import torch_util


def _build_joint_dof_map(char_model: MJCFCharModel) -> Dict[str, Tuple[int, int]]:
    mapping: Dict[str, Tuple[int, int]] = {}
    for joint_id in range(1, char_model.get_num_joints()):
        joint = char_model.get_joint(joint_id)
        dof_dim = joint.get_dof_dim()
        if dof_dim > 0:
            mapping[joint.name] = (joint.dof_idx, dof_dim)
    return mapping


def _auto_ground_root_height(
    model: MJCFCharModel,
    root_pos: np.ndarray,
    root_rot_exp: np.ndarray,
    dof_pos: np.ndarray,
    ground_bodies,
    ground_target_z: float,
) -> np.ndarray:
    valid_body_ids = []
    body_names = set(model.get_body_names())
    for body_name in ground_bodies:
        if body_name in body_names:
            valid_body_ids.append(model.get_body_id(body_name))

    if not valid_body_ids:
        raise ValueError(f"No valid ground bodies found in target model: {ground_bodies}")

    root_pos_t = torch.tensor(root_pos, dtype=torch.float32)
    root_rot_t = torch_util.exp_map_to_quat(torch.tensor(root_rot_exp, dtype=torch.float32))
    dof_t = torch.tensor(dof_pos, dtype=torch.float32)
    joint_rot_t = model.dof_to_rot(dof_t)
    body_pos_t, _ = model.forward_kinematics(root_pos_t, root_rot_t, joint_rot_t)

    foot_z = body_pos_t[:, valid_body_ids, 2]
    min_foot_z = torch.min(foot_z, dim=1).values.cpu().numpy().astype(np.float32)
    root_pos[:, 2] += np.float32(ground_target_z) - min_foot_z
    return root_pos


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a static H1-2 standing motion.")
    parser.add_argument("--char_file", default="data/assets/h1_2_official/h1_2.xml")
    parser.add_argument("--out_motion_file", default="data/motions/h1-2/h1-2_stand.pkl")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--root_x", type=float, default=0.0)
    parser.add_argument("--root_y", type=float, default=0.0)
    parser.add_argument("--root_z", type=float, default=1.03)
    parser.add_argument("--ground_target_z", type=float, default=0.074)
    parser.add_argument(
        "--ground_bodies",
        nargs="+",
        default=[
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_ankle_pitch_link",
            "right_ankle_pitch_link",
        ],
    )
    parser.add_argument("--torso_joint", type=float, default=0.0)
    parser.add_argument("--hip_pitch", type=float, default=0.0)
    parser.add_argument("--knee", type=float, default=0.0)
    parser.add_argument("--ankle_pitch", type=float, default=0.0)
    args = parser.parse_args()

    model = MJCFCharModel(torch.device("cpu"))
    model.load(args.char_file)
    dof_size = model.get_dof_size()
    joint_map = _build_joint_dof_map(model)

    num_frames = int(round(args.duration * args.fps)) + 1
    dof_pos = np.zeros((num_frames, dof_size), dtype=np.float32)

    symmetric_joint_values = {
        "torso_joint": args.torso_joint,
        "left_hip_pitch_joint": args.hip_pitch,
        "right_hip_pitch_joint": args.hip_pitch,
        "left_knee_joint": args.knee,
        "right_knee_joint": args.knee,
        "left_ankle_pitch_joint": args.ankle_pitch,
        "right_ankle_pitch_joint": args.ankle_pitch,
    }

    for joint_name, value in symmetric_joint_values.items():
        if joint_name in joint_map:
            dof_idx, dof_dim = joint_map[joint_name]
            if dof_dim == 1:
                dof_pos[:, dof_idx] = np.float32(value)

    root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    root_pos[:, 0] = np.float32(args.root_x)
    root_pos[:, 1] = np.float32(args.root_y)
    root_pos[:, 2] = np.float32(args.root_z)
    root_rot_exp = np.zeros((num_frames, 3), dtype=np.float32)

    root_pos = _auto_ground_root_height(
        model=model,
        root_pos=root_pos,
        root_rot_exp=root_rot_exp,
        dof_pos=dof_pos,
        ground_bodies=args.ground_bodies,
        ground_target_z=args.ground_target_z,
    )

    frames = np.concatenate([root_pos, root_rot_exp, dof_pos], axis=1)
    motion = Motion(loop_mode=LoopMode.WRAP, fps=args.fps, frames=frames)

    out_dir = os.path.dirname(args.out_motion_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    motion.save(args.out_motion_file)

    print("=" * 70)
    print("Standing motion generated")
    print("=" * 70)
    print(f"Character:   {args.char_file}")
    print(f"Output:      {args.out_motion_file}")
    print(f"Frames:      {num_frames}")
    print(f"Frame width: {frames.shape[1]}")
    print(f"FPS:         {args.fps}")
    print(f"Duration:    {args.duration:.3f}s")
    print(f"Root z:      {root_pos[0, 2]:.4f}")


if __name__ == "__main__":
    main()
