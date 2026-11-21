# Mini-SfM: Simplified Professor Demo

**Computer Vision theme:** 3D Reconstruction (also includes Feature Matching). This is a **simplified educational reimplementation based on the ideas in** [the original project](https://github.com/YashAG17/3D_modeling). It does **not** claim to reproduce the original implementation or its published results exactly.

## Core files (only 3 to present)

- `main.py`: entry point, command-line options and output summary
- `sfm_core.py`: SIFT features, mutual descriptor matching, essential matrix, relative camera poses, triangulation, incremental camera registration, *optional* subset bundle adjustment
- `visualize.py`: colored point cloud PLY export, matches picture, reconstruction figure and error histogram

Other: `make_demo_dataset.py` is **optional** and only makes synthetic input images for smoke testing. `requirements.txt` has Python packages; `images/` is where your own photographs go; `output/` will receive results. `VIVA_GUIDE.md` documents each important function.

## Quick start (Windows PowerShell / CMD)

Open this folder in VS Code terminal, then:

```bash
python -m pip install -r requirements.txt
python make_demo_dataset.py --out demo_images
python main.py --images demo_images --out output_demo --focal 850
```

Use Python 3.10+ (ideally Python 3.11 or 3.12). Run the commands from the folder containing `main.py`.

## Using your OWN photographs

1. Put **15–30 overlapping photos** of one still, textured scene inside `images/`.
2. Name them in shooting order: `001.jpg`, `002.jpg`, etc.
3. Run:

```bash
python main.py --images images --out output
```

Optional: `python main.py --images images --out output --bundle-adjust`

**Capture tips:** use a static scene with textured objects at visibly different depths; move the camera *sideways* in small steps (60–80% overlap), keep the same zoom/camera, avoid reflective/blank surfaces and moving people. Use good lighting. Avoid big jumps between consecutive photos. For smartphone images, copy photos at their original resolution, then the script scales them consistently. Approximate focal length is used if unknown; if you can determine focal length in **resized-image pixels**, pass `--focal PIXELS`.

## Output files

- `points.ply`: colored sparse 3D point cloud; view using MeshLab/CloudCompare
- `reconstruction.png`: 3D scatter plot and estimated camera centers
- `feature_matches.jpg`: matches between a photo pair
- `reprojection_error.png`: histogram of image reprojection error
- `report.json`: number of registered cameras, 3D points, pixel error
- `model.npz`: numerical point cloud and intrinsic matrix

A **sparse cloud is the intended result**: it will look like points, not a photorealistic 3D solid object. Absolute 3D scale is unknown. Camera calibration, texture, parallax, overlap, and initialization strongly affect quality. The demo is deliberately small and lacks loop closure, dense reconstruction, robust global point merging and full-scale global bundle adjustment. Some photo sequences may not register fully.

## Troubleshooting

- `No valid initial image pair`: improve overlap and parallax; start with adjacent photos of textured surfaces; try an appropriate `--focal`.
- Few registered photos: take smaller steps; avoid motion blur; use consistent camera intrinsics.
- Flat or warped point cloud: include foreground **and** background details; move sideways instead of turning in place; improve focal calibration.
- `module not found`: run `python -m pip install -r requirements.txt` in the same Python environment used for `python main.py`.
- A small or poor-looking synthetic point cloud is **not** evidence of real-photo performance.

## Authorship and attribution

Simplified teaching implementation assembled for this classroom demo; original project: https://github.com/YashAG17/3D_modeling . This package is not a verbatim reduced copy of the original repository. In your presentation, distinguish your original team's work from this adapted demonstration code.
