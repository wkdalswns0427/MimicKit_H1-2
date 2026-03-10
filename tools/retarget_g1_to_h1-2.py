#!/usr/bin/env python3
"""
Retarget a G1 MimicKit motion to H1-2 MimicKit motion format.

This tool maps source DOFs to target DOFs by joint name using MJCF character files.
It is intended for hinge-only G1/H1-2 models used in this repository.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "mimickit"))

from anim.motion import LoopMode, Motion, load_motion
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


def _pick_loop_mode(src_mode: LoopMode, override: str) -> LoopMode:
    if override == "keep":
        return src_mode
    if override == "wrap":
        return LoopMode.WRAP
    if override == "clamp":
        return LoopMode.CLAMP
    raise ValueError(f"Unsupported loop mode override: {override}")


def _parse_mjcf_joint_limits(char_file: str) -> Dict[str, Tuple[float, float]]:
    limits: Dict[str, Tuple[float, float]] = {}
    if not char_file.endswith(".xml"):
        return limits

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

        lo = float(min(vals[0], vals[1]))
        hi = float(max(vals[0], vals[1]))
        limits[name] = (lo, hi)
    return limits


def _smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = int(window)
    if window % 2 == 0:
        window += 1

    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    y = np.convolve(x_pad, kernel, mode="valid")
    return y.astype(np.float32)


def _auto_ground_root_height(
    tgt_model: MJCFCharModel,
    root_pos: np.ndarray,
    root_rot_exp: np.ndarray,
    tgt_dof: np.ndarray,
    ground_bodies: List[str],
    ground_target_z: float,
    ground_smooth_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    valid_body_ids: List[int] = []
    body_names = set(tgt_model.get_body_names())
    for body_name in ground_bodies:
        if body_name in body_names:
            valid_body_ids.append(tgt_model.get_body_id(body_name))

    if not valid_body_ids:
        raise ValueError(
            "No valid ground bodies found in target model. "
            f"Requested: {ground_bodies}"
        )

    root_pos_t = torch.tensor(root_pos, dtype=torch.float32)
    root_rot_t = torch_util.exp_map_to_quat(torch.tensor(root_rot_exp, dtype=torch.float32))
    dof_t = torch.tensor(tgt_dof, dtype=torch.float32)
    joint_rot_t = tgt_model.dof_to_rot(dof_t)
    body_pos_t, _ = tgt_model.forward_kinematics(root_pos_t, root_rot_t, joint_rot_t)

    foot_z = body_pos_t[:, valid_body_ids, 2]
    min_foot_z = torch.min(foot_z, dim=1).values.cpu().numpy().astype(np.float32)

    per_frame_offset = np.float32(ground_target_z) - min_foot_z
    per_frame_offset = _smooth_1d(per_frame_offset, ground_smooth_window)

    out_root_pos = root_pos.copy()
    out_root_pos[:, 2] += per_frame_offset
    return out_root_pos, per_frame_offset


def retarget(
    src_char_file: str,
    tgt_char_file: str,
    src_motion_file: str,
    out_motion_file: str,
    torso_source: str,
    output_fps: int,
    loop_mode: str,
    root_height_offset: float,
    clamp_joint_limits: bool,
    auto_ground_feet: bool,
    ground_bodies: List[str],
    ground_target_z: float,
    ground_smooth_window: int,
) -> None:
    src_model = MJCFCharModel(torch.device("cpu"))
    tgt_model = MJCFCharModel(torch.device("cpu"))
    src_model.load(src_char_file)
    tgt_model.load(tgt_char_file)

    src_motion = load_motion(src_motion_file)
    src_frames = src_motion.frames

    if src_frames.ndim != 2:
        raise ValueError(f"Expected source frames to be 2D, got shape={src_frames.shape}")
    if src_frames.shape[1] != 6 + src_model.get_dof_size():
        raise ValueError(
            "Source frame width mismatch: "
            f"got {src_frames.shape[1]}, expected {6 + src_model.get_dof_size()}"
        )

    src_root_pos = src_frames[:, 0:3].copy()
    src_root_rot = src_frames[:, 3:6].copy()
    src_dof = src_frames[:, 6:].copy()

    if root_height_offset != 0.0:
        src_root_pos[:, 2] += root_height_offset

    tgt_dof_size = tgt_model.get_dof_size()
    tgt_dof = np.zeros((src_frames.shape[0], tgt_dof_size), dtype=np.float32)

    src_joint_map = _build_joint_dof_map(src_model)
    tgt_joint_map = _build_joint_dof_map(tgt_model)
    tgt_joint_limits = _parse_mjcf_joint_limits(tgt_char_file) if clamp_joint_limits else {}

    manual_alias = {}
    if torso_source != "none":
        manual_alias["torso_joint"] = torso_source

    mapped: List[str] = []
    unmapped: List[str] = []

    for tgt_name, (tgt_idx, tgt_dim) in tgt_joint_map.items():
        src_name = manual_alias.get(tgt_name, tgt_name)
        if src_name not in src_joint_map:
            unmapped.append(tgt_name)
            continue

        src_idx, src_dim = src_joint_map[src_name]
        if src_dim != tgt_dim:
            unmapped.append(tgt_name)
            continue

        tgt_dof[:, tgt_idx : tgt_idx + tgt_dim] = src_dof[:, src_idx : src_idx + src_dim]
        mapped.append(f"{tgt_name} <- {src_name}")

    num_clamped = 0
    if clamp_joint_limits and tgt_joint_limits:
        for tgt_name, (tgt_idx, tgt_dim) in tgt_joint_map.items():
            if tgt_dim != 1:
                continue
            if tgt_name not in tgt_joint_limits:
                continue

            lo, hi = tgt_joint_limits[tgt_name]
            before = tgt_dof[:, tgt_idx].copy()
            tgt_dof[:, tgt_idx] = np.clip(tgt_dof[:, tgt_idx], lo, hi)
            num_clamped += int(np.count_nonzero(before != tgt_dof[:, tgt_idx]))

    grounding_offset = np.zeros(src_frames.shape[0], dtype=np.float32)
    if auto_ground_feet:
        src_root_pos, grounding_offset = _auto_ground_root_height(
            tgt_model=tgt_model,
            root_pos=src_root_pos,
            root_rot_exp=src_root_rot,
            tgt_dof=tgt_dof,
            ground_bodies=ground_bodies,
            ground_target_z=ground_target_z,
            ground_smooth_window=ground_smooth_window,
        )

    out_frames = np.concatenate([src_root_pos, src_root_rot, tgt_dof], axis=1)
    out_fps = src_motion.fps if output_fps <= 0 else output_fps
    out_loop_mode = _pick_loop_mode(src_motion.loop_mode, loop_mode)
    out_motion = Motion(loop_mode=out_loop_mode, fps=out_fps, frames=out_frames)

    out_dir = os.path.dirname(out_motion_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_motion.save(out_motion_file)

    print("=" * 70)
    print("Retarget complete")
    print("=" * 70)
    print(f"Source char:   {src_char_file}")
    print(f"Target char:   {tgt_char_file}")
    print(f"Source motion: {src_motion_file}")
    print(f"Output motion: {out_motion_file}")
    print(f"Frames:        {out_frames.shape[0]}")
    print(f"Frame width:   {out_frames.shape[1]} (expected {6 + tgt_dof_size})")
    print(f"FPS:           {out_fps}")
    print(f"Loop mode:     {out_loop_mode.name}")
    print(f"Mapped joints: {len(mapped)}")
    print(f"Unmapped:      {len(unmapped)}")
    print(f"Joint clamps:  {num_clamped} samples")
    if auto_ground_feet:
        print(
            "Ground shift z (min/mean/max): "
            f"{grounding_offset.min():.4f} / {grounding_offset.mean():.4f} / {grounding_offset.max():.4f}"
        )

    if mapped:
        print("-" * 70)
        print("Mapped joint aliases")
        for item in mapped:
            print(f"  {item}")

    if unmapped:
        print("-" * 70)
        print("Unmapped target joints (left at zero)")
        for name in unmapped:
            print(f"  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retarget G1 MimicKit motion to H1-2 motion.")
    parser.add_argument(
        "--src_char_file",
        default="data/assets/g1/g1.xml",
        help="Source character MJCF file (default: G1 XML).",
    )
    parser.add_argument(
        "--tgt_char_file",
        default="data/assets/h1_2_official/h1_2.xml",
        help="Target character MJCF file (default: H1-2 XML).",
    )
    parser.add_argument(
        "--src_motion_file",
        required=True,
        help="Input source MimicKit motion .pkl file.",
    )
    parser.add_argument(
        "--out_motion_file",
        required=True,
        help="Output target MimicKit motion .pkl file.",
    )
    parser.add_argument(
        "--torso_source",
        default="waist_yaw_joint",
        choices=["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint", "none"],
        help="Source joint used for target torso_joint.",
    )
    parser.add_argument(
        "--output_fps",
        type=int,
        default=-1,
        help="Output FPS. Use -1 to keep source FPS.",
    )
    parser.add_argument(
        "--loop_mode",
        default="keep",
        choices=["keep", "wrap", "clamp"],
        help="Output loop mode. keep=inherit source loop mode.",
    )
    parser.add_argument(
        "--root_height_offset",
        type=float,
        default=0.0,
        help="Constant offset added to root z in meters.",
    )
    parser.add_argument(
        "--clamp_joint_limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp mapped target DOFs to target MJCF joint ranges (default: enabled).",
    )
    parser.add_argument(
        "--auto_ground_feet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-shift root z so selected foot bodies track ground height (default: enabled).",
    )
    parser.add_argument(
        "--ground_bodies",
        nargs="+",
        default=[
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_ankle_pitch_link",
            "right_ankle_pitch_link",
        ],
        help="Target body names used to estimate foot height for auto-grounding.",
    )
    parser.add_argument(
        "--ground_target_z",
        type=float,
        default=0.0,
        help="Desired minimum z of ground bodies after grounding.",
    )
    parser.add_argument(
        "--ground_smooth_window",
        type=int,
        default=7,
        help="Moving-average window for per-frame root z grounding offset.",
    )

    args = parser.parse_args()
    retarget(
        src_char_file=args.src_char_file,
        tgt_char_file=args.tgt_char_file,
        src_motion_file=args.src_motion_file,
        out_motion_file=args.out_motion_file,
        torso_source=args.torso_source,
        output_fps=args.output_fps,
        loop_mode=args.loop_mode,
        root_height_offset=args.root_height_offset,
        clamp_joint_limits=args.clamp_joint_limits,
        auto_ground_feet=args.auto_ground_feet,
        ground_bodies=args.ground_bodies,
        ground_target_z=args.ground_target_z,
        ground_smooth_window=args.ground_smooth_window,
    )


if __name__ == "__main__":
    main()
