#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NUM_ENVS="${NUM_ENVS:-512}"
VISUALIZE="${VISUALIZE:-false}"
OUT_DIR="${OUT_DIR:-output/}"
PRESET="${PRESET:-walk}"

case "$PRESET" in
  walk)
    ENV_CONFIG="data/envs/deepmimic_h1-2_env.yaml"
    MOTION_FILE="data/motions/h1-2/h1-2_from_g1_walk_v2_ground.pkl"
    SRC_MOTION_FILE="data/motions/g1/g1_walk.pkl"
    ;;
  stand)
    ENV_CONFIG="data/envs/deepmimic_h1-2_stand_env.yaml"
    MOTION_FILE="data/motions/h1-2/h1-2_stand.pkl"
    SRC_MOTION_FILE=""
    ;;
  *)
    echo "Unsupported PRESET=$PRESET. Use PRESET=walk or PRESET=stand."
    exit 1
    ;;
esac

if [[ "$VISUALIZE" == "true" && "${FORCE_LARGE_VIS:-0}" != "1" && "$NUM_ENVS" -gt 128 ]]; then
  echo "VISUALIZE=true with NUM_ENVS=$NUM_ENVS is likely to be OOM-killed on Isaac Sim."
  echo "Clamping NUM_ENVS to 128. Set FORCE_LARGE_VIS=1 to override."
  NUM_ENVS=128
fi

if [[ ! -f "$MOTION_FILE" ]]; then
  if [[ "$PRESET" == "walk" ]]; then
    echo "Missing $MOTION_FILE, generating from $SRC_MOTION_FILE..."
    python3 tools/retarget_g1_to_h1-2.py \
      --src_motion_file "$SRC_MOTION_FILE" \
      --out_motion_file "$MOTION_FILE" \
      --ground_target_z 0.074 \
      --ground_smooth_window 7 \
      --torso_source waist_pitch_joint
  else
    echo "Missing $MOTION_FILE, generating standing bootstrap motion..."
    python3 tools/make_h1-2_standing_motion.py \
      --out_motion_file "$MOTION_FILE"
  fi
fi

STDCPP="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
GCCS="/lib/x86_64-linux-gnu/libgcc_s.so.1"
if [[ -f "$STDCPP" && -f "$GCCS" ]]; then
  export LD_PRELOAD="$STDCPP:$GCCS${LD_PRELOAD:+:$LD_PRELOAD}"
fi

python3 mimickit/run.py \
  --mode train \
  --num_envs "$NUM_ENVS" \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config "$ENV_CONFIG" \
  --agent_config data/agents/deepmimic_h1-2_ppo_agent.yaml \
  --visualize "$VISUALIZE" \
  --out_dir "$OUT_DIR"
