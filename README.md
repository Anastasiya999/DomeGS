## Installation

The training and rendering code is based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and has the same requirements:

- Linux, NVIDIA GPU with Compute Capability 7.0+
- Conda
- CUDA toolkit 11.8 (used to compile the rasterizer extensions)

Clone the repository with its submodules:

```bash
git clone --recursive <repo-url>
cd DomeGS
```

If you already cloned it without `--recursive`:

```bash
git submodule update --init --recursive
```

Create the environment. This also compiles `diff-gaussian-rasterization` and `simple-knn`:

```bash
conda env create --file environment.yml
conda activate gs_sky
```

Check the install:

```bash
python -c "import torch, diff_gaussian_rasterization, simple_knn; print(torch.cuda.is_available())"
```

Common issues:
- **Submodule folders are empty:** the repository was cloned without `--recursive`; run the `git submodule` command above.
- **Errors while compiling the submodules:** the CUDA toolkit version must match the one PyTorch was built with (11.8 here). See the [3DGS installation notes](https://github.com/graphdeco-inria/gaussian-splatting#setup) for platform-specific fixes.

Panorama outpainting uses a separate environment with newer packages; see [`panorama_outpainting/README.md`](panorama_outpainting/README.md).

## Pipeline

The pipeline has two automatic parts with manual panorama preparation in between.

### Input

A scene folder with a trained 3DGS model:

```
<scene_dir>/                       # any trained 3DGS model folder
├── cameras.json
└── point_cloud/iteration_30000/point_cloud.ply

```

### Part 1: render the panorama

```bash
bash run_panorama.sh <scene_dir>
```

This writes:
- `panorama.png`: the 360° equirectangular render from the centre of the cameras
- `panorama_eye.json`: the viewpoint and alignment used for the render (needed by Part 2)
- `panorama_raw.png`: an untouched copy of the render

Regions the 3DGS model never saw (mostly the sky and areas far from the cameras) stay empty (black).

### Preparing the panorama

1. **Create a mask.** Make a grayscale image with the same size as `panorama.png`: **white = areas to generate**, black = areas to keep. Any image editor works (e.g. GIMP or Photoshop). Cover the empty regions plus a small margin into the blurry or broken edges around them.

2. **Outpaint with Flux.** Fill the masked regions with FLUX.2 [klein] and the 360° ERP outpainting LoRA. Installation, hardware requirements and all options are described in [`panorama_outpainting/README.md`](panorama_outpainting/README.md). Basic run:

   ```bash
   python panorama_outpainting/outpaint_panorama.py \
       --image  <scene_dir>/panorama_raw.png \
       --mask   <scene_dir>/panorama_mask.png \
       --output <scene_dir>/panorama_outpainted.png
   ```

   The result is 2048×1024.

3. **Upscale.** Upscale the outpainted panorama, for example with the [Upscaler Space](https://huggingface.co/spaces/Phips/Upscaler). Use at least 2× so it is back at the render's resolution (4096×2048), and keep the 2:1 aspect ratio.

4. **Save the result as `<scene_dir>/panorama.png`**, overwriting the render.

   > Keep the exact name `panorama.png`. Part 2 finds `panorama_eye.json` by the panorama's file name, so a different name will make it fail.

### Part 2: sky dataset and training

```bash
bash run_sky.sh <scene_dir>
```

This runs in order:
1. `scripts/sky_pipeline/estimate_base_pitches.py`: estimates the camera pitch angles → `base_pitches.json`
2. `scripts/sky_pipeline/render_camera_band_clean.py`: renders the training views from the 3DGS model and the panorama → `sky_band/`
3. `train_sky.py`: trains the sky model → `out_FINAL/`

### Output

```
<scene_dir>/
├── cameras.json           # input
├── point_cloud/           # input model
├── panorama_raw.png       # original render
├── panorama.png           # final outpainted + upscaled panorama
├── panorama_eye.json
├── base_pitches.json
├── sky_band/              # training dataset
└── out_FINAL/             # trained sky model
```

Render the trained model:

```bash
python scripts/render.py -m <scene_dir>/out_FINAL --gs_type gs_sky
```