# Professor / Viva Guide (simple English)

## One-minute project explanation

"Our project is Python-based **Structure-from-Motion**, within **3D Reconstruction** and **Feature Matching**. We input overlapping photographs of the same static scene. SIFT finds keypoints, descriptor matching finds the same features in different views, the essential matrix helps estimate how the camera moved, triangulation converts matching 2D points to 3D, and PnP estimates later camera positions. We export a *sparse* point cloud as a PLY file. We also calculate image reprojection error to evaluate consistency."

## File-by-file walkthrough

### 1. main.py (start here)

- `main()`: parses CLI options, loads image data, calls the SfM pipeline, saves images/PLY/results and prints summary.
- Explain: "This is our control file; it does not contain the complex CV maths."

### 2. sfm_core.py (main computer vision algorithms)

- `load_images()`: loads and resizes ordered image files.
- `camera_matrix()`: creates intrinsic matrix **K** from focal length and image center.
- `extract_features()`: finds SIFT interest points and their descriptors.
- `match_features()`: matches SIFT descriptors; Lowe ratio and mutual tests reject ambiguous matches.
- `relative_pose()`: finds essential matrix E with RANSAC, then uses it to estimate rotation R and translation direction t.
- `triangulate()`: uses two cameras and corresponding image points to estimate 3D point coordinates; checks depth and reprojection error.
- `register_pnp()`: estimates the position/orientation of another camera using existing 3D points matched to 2D image locations.
- `reconstruct()`: links all the stages: choose seed pair, triangulate new 3D points, add cameras, repeat.
- `reprojection_errors()`: checks where reconstructed 3D points project onto the input images and compares to measured keypoints.
- `bundle_adjust()`: optional **small subset** least-squares refinement, not full global BA; optimizes some 3D points and camera poses jointly.

### 3. visualize.py (output)

- `save_matches()`: image showing feature correspondences between photos.
- `save_ply()`: writes 3D points + RGB color to a PLY file.
- `save_reconstruction()`: draws point cloud and estimated camera positions.
- `save_error_plot()`: plots reprojection errors.

### 4. make_demo_dataset.py (optional testing utility)

- `make_texture()`: creates recognizable random textures to match across images.
- `main()`: renders a toy scene from different viewpoints; useful for checking that the code runs. **Do not pass synthetic results off as real-camera measurements.**

## What to show the professor

1. Show three sample overlapping photos from `images/`.
2. Show `feature_matches.jpg` and explain how SIFT identified shared points.
3. Show `reconstruction.png` and open `points.ply` using a point-cloud viewer.
4. Open `report.json`: explain registered image count, 3D points and error. These are from the current run, not hardcoded.
5. Discuss the limitations: output is sparse; metric scale unknown; images must overlap; imperfect camera calibration may warp geometry.

## Likely viva questions

**Q: Why is this a CV project?** A: It extracts image features, matches them, estimates camera geometry and reconstructs 3D points from 2D pictures.

**Q: Why SIFT?** A: It detects repeatable local patterns and describes them in a form useful for matching changes in viewpoint and scale.

**Q: What is the essential matrix?** A: It encodes the epipolar relationship between two calibrated camera views.

**Q: What is triangulation?** A: It estimates the location of a 3D point from the same point seen by two camera views.

**Q: What is PnP?** A: Perspective-n-Point estimates a new camera pose from known 3D points and their 2D image coordinates.

**Q: What is bundle adjustment?** A: It jointly adjusts camera poses and 3D point positions to reduce reprojection error. This package demonstrates it only on a small subset, optionally.

**Q: Why does the cloud have scattered points?** A: This is sparse SfM: reconstructed feature points, not dense surface reconstruction.

**Q: Why can the output be distorted?** A: Insufficient parallax, wrong camera intrinsics, wrong matches or low scene texture.

**Q: What is one limitation in this simplified version?** A: It has no global loop closure and only limited multiview track association; it can fail on difficult scenes.

**Q: What is the evaluation metric?** A: Pixel reprojection error and proportion of registered images; these don't by themselves measure absolute 3D shape accuracy.
