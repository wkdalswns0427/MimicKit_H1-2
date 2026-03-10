#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NUM_ENVS="${NUM_ENVS:-64}"
VISUALIZE="${VISUALIZE:-false}"
OUT_DIR="${OUT_DIR:-output/h1-2_walk_from_retarget}"
ENGINE_CONFIG="${ENGINE_CONFIG:-data/engines/isaac_lab_engine.yaml}"
BASE_ENV_CONFIG="${BASE_ENV_CONFIG:-data/envs/deepmimic_h1-2_env.yaml}"
AGENT_CONFIG="${AGENT_CONFIG:-data/agents/deepmimic_h1-2_ppo_agent.yaml}"

SRC_MOTION_FILE="${SRC_MOTION_FILE:-data/motions/g1/g1_walk.pkl}"
RETARGET_MOTION_FILE="${RETARGET_MOTION_FILE:-data/motions/h1-2/h1-2_from_g1_walk_v3_train.pkl}"

GROUND_TARGET_Z="${GROUND_TARGET_Z:-0.09}"
GROUND_SMOOTH_WINDOW="${GROUND_SMOOTH_WINDOW:-11}"
TORSO_SOURCE="${TORSO_SOURCE:-waist_pitch_joint}"
FORCE_RETARGET="${FORCE_RETARGET:-0}"

MAX_SLIP_MEAN="${MAX_SLIP_MEAN:-1.2}"
MAX_SLIP_MAX="${MAX_SLIP_MAX:-5.0}"
ALLOW_HIGH_SLIP="${ALLOW_HIGH_SLIP:-0}"

if [[ "$VISUALIZE" == "true" && "${FORCE_LARGE_VIS:-0}" != "1" && "$NUM_ENVS" -gt 128 ]]; then
  echo "VISUALIZE=true with NUM_ENVS=$NUM_ENVS may be OOM-killed on Isaac Sim."
  echo "Clamping NUM_ENVS to 128. Set FORCE_LARGE_VIS=1 to override."
  NUM_ENVS=128
fi

if [[ ! -f "$RETARGET_MOTION_FILE" || "$FORCE_RETARGET" == "1" ]]; then
  echo "Retargeting motion -> $RETARGET_MOTION_FILE"
  python3 tools/retarget_g1_to_h1-2.py \
    --src_motion_file "$SRC_MOTION_FILE" \
    --out_motion_file "$RETARGET_MOTION_FILE" \
    --ground_target_z "$GROUND_TARGET_Z" \
    --ground_smooth_window "$GROUND_SMOOTH_WINDOW" \
    --torso_source "$TORSO_SOURCE"
fi

# Validate motion quality before training.
VALIDATE_LOG="$(mktemp /tmp/h1_2_validate.XXXXXX.log)"
python3 tools/validate_motion.py \
  --char_file data/assets/h1_2_official/h1_2.xml \
  --motion_file "$RETARGET_MOTION_FILE" | tee "$VALIDATE_LOG"

SLIP_MEAN="$(awk '/Foot slip mean:/ {print $4}' "$VALIDATE_LOG" | tail -n1)"
SLIP_MAX="$(awk '/Foot slip max:/ {print $4}' "$VALIDATE_LOG" | tail -n1)"
rm -f "$VALIDATE_LOG"

if [[ -n "$SLIP_MEAN" && -n "$SLIP_MAX" ]]; then
  TOO_HIGH="$(python3 - <<PY
sm=float("$SLIP_MEAN")
sx=float("$SLIP_MAX")
print(int((sm > float("$MAX_SLIP_MEAN")) or (sx > float("$MAX_SLIP_MAX"))))
PY
)"
  if [[ "$TOO_HIGH" == "1" && "$ALLOW_HIGH_SLIP" != "1" ]]; then
    echo "Slip is high (mean=$SLIP_MEAN, max=$SLIP_MAX)."
    echo "Aborting. Set ALLOW_HIGH_SLIP=1 to train anyway."
    exit 2
  fi
fi

# Use a temporary env config pointing to the chosen retargeted motion.
TMP_ENV_CONFIG="$(mktemp /tmp/deepmimic_h1-2_walk.XXXXXX.yaml)"
cp "$BASE_ENV_CONFIG" "$TMP_ENV_CONFIG"
sed -i "s|^motion_file:.*|motion_file: \"$RETARGET_MOTION_FILE\"|" "$TMP_ENV_CONFIG"

STDCPP="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
GCCS="/lib/x86_64-linux-gnu/libgcc_s.so.1"
if [[ -f "$STDCPP" && -f "$GCCS" ]]; then
  export LD_PRELOAD="$STDCPP:$GCCS${LD_PRELOAD:+:$LD_PRELOAD}"
fi

echo "Training with env config: $TMP_ENV_CONFIG"
python3 mimickit/run.py \
  --mode train \
  --num_envs "$NUM_ENVS" \
  --engine_config "$ENGINE_CONFIG" \
  --env_config "$TMP_ENV_CONFIG" \
  --agent_config "$AGENT_CONFIG" \
  --visualize "$VISUALIZE" \
  --out_dir "$OUT_DIR"
