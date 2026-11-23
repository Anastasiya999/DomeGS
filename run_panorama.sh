#!/bin/bash
# Part 1: render the 360° panorama from a trained 3DGS model.
#   bash run_panorama.sh <scene_dir>
# Expects <scene_dir> (point_cloud/ + cameras.json).
# Then follow "Preparing the panorama" in README.md before running run_sky.sh.

set -euo pipefail
[ $# -eq 1 ] || { echo "Usage: bash $0 <scene_dir>" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")}"
export PYTHONPATH=$(pwd)

SCENE=$(realpath "$1")
MODEL=$SCENE
PANO=$SCENE/panorama.png

[ -f "$MODEL/cameras.json" ] || { echo "No trained 3DGS model in $MODEL (cameras.json missing)" >&2; exit 1; }

python scripts/sky_pipeline/render_panorama.py --model_path "$MODEL"
cp "$PANO" "$SCENE/panorama_raw.png"      # keep the original render

echo "Saved: $PANO (copy: panorama_raw.png)"
echo "Next: mask, outpaint and upscale it (see README), save the result as $PANO,"
echo "then run: bash run_sky.sh $1"

