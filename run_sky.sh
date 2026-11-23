#!/bin/bash
# Part 2: base pitch estimation -> sky band dataset -> sky model training.
#   bash run_sky.sh <scene_dir>
# Run after <scene_dir>/panorama.png has been outpainted and upscaled.

set -euo pipefail
[ $# -eq 1 ] || { echo "Usage: bash $0 <scene_dir>" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")}"
export PYTHONPATH=$(pwd)

SCENE=$(realpath "$1")
MODEL=$SCENE
PANO=$SCENE/panorama.png

[ -f "$PANO" ] || { echo "Panorama not found: $PANO" >&2; exit 1; }
[ -f "$SCENE/panorama_eye.json" ] || { echo "Missing panorama_eye.json (run run_panorama.sh first)" >&2; exit 1; }

echo "=== [1/3] Estimating base pitches"
python scripts/sky_pipeline/estimate_base_pitches.py --model_path "$MODEL" --bg_panorama "$PANO"

echo "=== [2/3] Creating sky band dataset"
python scripts/sky_pipeline/render_camera_band.py --model_path "$MODEL" --bg_panorama "$PANO"

echo "=== [3/3] Training sky model"
python train_sky.py -s "$SCENE/sky_band" -m "$SCENE/out_FINAL"

echo "Done. Model: $SCENE/out_FINAL"