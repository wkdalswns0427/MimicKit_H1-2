#!/usr/bin/env python3
"""
Validate a MimicKit motion against an MJCF character model.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "mimickit"))

from anim.motion import load_motion
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


def _parse_mjcf_joint_limits(char_file: str) -> Dict[str, Tuple[float, float]]:
    limits: Dict[str, Tuple[float, float]] = {}
    tree = ET.parse(char_file)
    root = tree.getroot()
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name")
        range_str = joint.attrib.get("range")
        if name is None or range_str is None:
            continue
        vals = np.fromstring(range_str, sep=" ", dtype=np.float32)
        if vals.shape[0] != 2:
            continue
        limits[name] = (float(min(vals[0], vals[1])), float(max(vals[0], vals[1])))
    return limits


def _foot_slip_stats(body_pos: np.ndarray, body_names: List[str], fps: int, ground_bodies: List[str], contact_height: float) -> Tuple[float, float]:
    valid_ids = [body_names.index(name) for name in ground_bodies if name in body_names]
    if not valid_ids or body_pos.shape[0] < 2:
        return 0.0, 0.0

    pos_xy = body_pos[:, valid_ids, :2]
    pos_z = body_pos[:, valid_ids, 2]
    vel_xy = np.linalg.norm(np.diff(pos_xy, axis=0), axis=-1) * fps
    contact_mask = pos_z[:-1] <= contact_height
    if not np.any(contact_mask):
        return 0.0, 0.0

    contact_speeds = vel_xy[contact_mask]
    return float(np.mean(contact_speeds)), float(np.max(contact_speeds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a MimicKit motion for feasibility issues.")
    parser.add_argument("--char_file", required=True)
    parser.add_argument("--motion_file", required=True)
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
    parser.add_argument("--contact_height", type=float, default=0.12)
    parser.add_argument("--joint_limit_margin", type=float, default=0.05)
    args = parser.parse_args()

    model = MJCFCharModel(torch.device("cpu"))
    model.load(args.char_file)
    motion = load_motion(args.motion_file)
    frames = motion.frames

    expected_width = 6 + model.get_dof_size()
    if frames.ndim != 2 or frames.shape[1] != expected_width:
        raise ValueError(f"Invalid motion shape {frames.shape}, expected [T, {expected_width}]")

    root_pos = torch.tensor(frames[:, 0:3], dtype=torch.float32)
    root_rot = torch_util.exp_map_to_quat(torch.tensor(frames[:, 3:6], dtype=torch.float32))
    dof_pos = torch.tensor(frames[:, 6:], dtype=torch.float32)
    joint_rot = model.dof_to_rot(dof_pos)
    body_pos_t, _ = model.forward_kinematics(root_pos, root_rot, joint_rot)
    body_pos = body_pos_t.cpu().numpy()
    body_names = model.get_body_names()

    joint_map = _build_joint_dof_map(model)
    joint_limits = _parse_mjcf_joint_limits(args.char_file)

    print("=" * 70)
    print("Motion validation")
    print("=" * 70)
    print(f"Character:     {args.char_file}")
    print(f"Motion:        {args.motion_file}")
    print(f"Frames:        {frames.shape[0]}")
    print(f"FPS:           {motion.fps}")
    print(f"Duration:      {(frames.shape[0] - 1) / motion.fps:.3f}s")
    print(f"Frame width:   {frames.shape[1]}")
    print(f"Root z min:    {float(frames[:, 2].min()):.4f}")
    print(f"Root z mean:   {float(frames[:, 2].mean()):.4f}")
    print(f"Root z max:    {float(frames[:, 2].max()):.4f}")
    print(f"Body z min:    {float(body_pos[..., 2].min()):.4f}")

    valid_ground_names = [name for name in args.ground_bodies if name in body_names]
    for name in valid_ground_names:
        body_id = body_names.index(name)
        z = body_pos[:, body_id, 2]
        print(f"{name:24s} z min/mean/max: {float(z.min()):.4f} / {float(z.mean()):.4f} / {float(z.max()):.4f}")

    slip_mean, slip_max = _foot_slip_stats(
        body_pos=body_pos,
        body_names=body_names,
        fps=motion.fps,
        ground_bodies=args.ground_bodies,
        contact_height=args.contact_height,
    )
    print(f"Foot slip mean: {slip_mean:.4f} m/s")
    print(f"Foot slip max:  {slip_max:.4f} m/s")

    total_limited_samples = 0
    total_samples = 0
    saturated_joints = []
    dof_np = dof_pos.cpu().numpy()
    for joint_name, (dof_idx, dof_dim) in joint_map.items():
        if dof_dim != 1 or joint_name not in joint_limits:
            continue
        lo, hi = joint_limits[joint_name]
        vals = dof_np[:, dof_idx]
        motion_span = float(vals.max() - vals.min())
        if motion_span < 1e-4:
            continue
        margin_hits = np.logical_or(vals <= lo + args.joint_limit_margin, vals >= hi - args.joint_limit_margin)
        hit_frac = float(np.mean(margin_hits))
        total_limited_samples += int(np.count_nonzero(margin_hits))
        total_samples += vals.shape[0]
        if hit_frac > 0.01:
            saturated_joints.append((joint_name, hit_frac, lo, hi, float(vals.min()), float(vals.max())))

    if total_samples > 0:
        print(f"Joint-limit margin hit rate: {total_limited_samples / total_samples:.4%}")
    if saturated_joints:
        print("-" * 70)
        print("Joints frequently near limits")
        for joint_name, hit_frac, lo, hi, vmin, vmax in saturated_joints:
            print(
                f"{joint_name:24s} hit={hit_frac:6.2%} "
                f"range=[{lo:.3f}, {hi:.3f}] "
                f"used=[{vmin:.3f}, {vmax:.3f}]"
            )

    print("-" * 70)
    issues = []
    if float(body_pos[..., 2].min()) < -0.01:
        issues.append("body penetrates below ground")
    if slip_mean > 0.5:
        issues.append("high mean foot slip while near ground")
    if slip_max > 1.0:
        issues.append("high max foot slip while near ground")
    if total_samples > 0 and (total_limited_samples / total_samples) > 0.05:
        issues.append("motion spends too much time near joint limits")

    if issues:
        print("Warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No major issues detected by heuristic checks.")


if __name__ == "__main__":
    main()
