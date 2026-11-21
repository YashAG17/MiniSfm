"""The core of a compact educational incremental Structure-from-Motion pipeline.

Method: SIFT -> descriptor matching -> essential matrix -> relative pose ->
triangulation -> 2D/3D PnP -> optional bundle adjustment.
Not a dense reconstruction or a replacement for production SfM (COLMAP).
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


@dataclass
class Model:
    points: np.ndarray
    colors: np.ndarray
    cameras: dict
    keypoints: list
    observations: list
    K: np.ndarray
    registered: list
    diagnostics: dict


def load_images(folder, max_size=1200):
    """Read and consistently resize all photos; return RGB images and paths."""
    paths = sorted([p for p in Path(folder).iterdir()
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    if len(paths) < 3:
        raise ValueError('Need >=3 overlapping photographs in the images folder.')
    images = []
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f'Could not read image: {path}')
        h, w = bgr.shape[:2]
        if max(h, w) > max_size:
            s = max_size / max(h, w)
            bgr = cv2.resize(bgr, (round(w*s), round(h*s)), interpolation=cv2.INTER_AREA)
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if len(set((im.shape[1], im.shape[0]) for im in images)) != 1:
        raise ValueError('Use images with the same aspect ratio and resolution after resizing.')
    return images, paths


def camera_matrix(image, focal=None):
    """Approximate pinhole intrinsics: use a known focal length if supplied."""
    h, w = image.shape[:2]
    f = float(focal) if focal else 1.2 * max(w, h)
    if f <= 0:
        raise ValueError('Focal length must be positive')
    return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1.]], float)


def extract_features(images, nfeatures=3500):
    """Detect SIFT keypoints and compute descriptors for each photograph."""
    detector = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = [], []
    for image in images:
        k, d = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
        keypoints.append(np.array([x.pt for x in k], dtype=np.float64).reshape(-1, 2))
        descriptors.append(d)
    return keypoints, descriptors


def match_features(desc_a, desc_b, ratio=0.73):
    """Two-nearest-neighbours Lowe ratio test followed by mutual matching."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return np.empty((0, 2), dtype=int)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(desc_a, desc_b, k=2)
    backward = matcher.knnMatch(desc_b, desc_a, k=2)
    good_ab = {m.queryIdx: m.trainIdx for m, n in forward if m.distance < ratio*n.distance}
    good_ba = {m.queryIdx: m.trainIdx for m, n in backward if m.distance < ratio*n.distance}
    return np.array([(a, b) for a, b in good_ab.items() if good_ba.get(b) == a], dtype=int).reshape(-1, 2)


def relative_pose(x1, x2, K, threshold=1.5):
    """RANSAC essential matrix + cheirality: camera-2 pose relative to camera-1."""
    if len(x1) < 12:
        return None
    E, inlier_mask = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC,
                                         prob=0.999, threshold=threshold)
    if E is None or inlier_mask is None:
        return None
    # OpenCV may return multiple essential matrices stacked vertically.
    if E.shape != (3, 3):
        E = E[:3, :3]
    count, R, t, pose_mask = cv2.recoverPose(E, x1, x2, K, mask=inlier_mask)
    keep = pose_mask.ravel() > 0
    if count < 20:
        return None
    return R, t.reshape(3), keep


def triangulate(x1, x2, pose1, pose2, K, min_angle_deg=0.0):
    """Convert matching image coordinates into 3D positions with two views."""
    R1, t1 = pose1
    R2, t2 = pose2
    P1 = K @ np.column_stack([R1, t1])
    P2 = K @ np.column_stack([R2, t2])
    hom = cv2.triangulatePoints(P1, P2, x1.T, x2.T)
    xyz = (hom[:3] / np.where(np.abs(hom[3]) < 1e-12, 1e-12, hom[3])).T
    z1 = (R1 @ xyz.T + t1[:, None])[2]
    z2 = (R2 @ xyz.T + t2[:, None])[2]
    def project(P, pts):
        q = (P @ np.column_stack([pts, np.ones(len(pts))]).T).T
        return q[:, :2] / np.maximum(np.abs(q[:, 2:]), 1e-12)
    err = (np.linalg.norm(project(P1, xyz)-x1, axis=1) +
           np.linalg.norm(project(P2, xyz)-x2, axis=1))/2
    good = np.isfinite(xyz).all(axis=1) & (z1 > 0) & (z2 > 0) & (err < 3.0)
    if min_angle_deg > 0:
        c1 = -R1.T @ t1
        c2 = -R2.T @ t2
        v1 = xyz - c1
        v2 = xyz - c2
        cosine = np.einsum('ij,ij->i', v1, v2) / np.maximum(
            np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1), 1e-12)
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        good &= angle >= min_angle_deg
    return xyz, good, err


def register_pnp(points_3d, points_2d, K):
    """Recover a new camera pose using correspondences to existing 3D points."""
    if len(points_3d) < 8:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(points_3d, np.float64), np.asarray(points_2d, np.float64),
        K, None, iterationsCount=1000, reprojectionError=4.0, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 8:
        return None
    ix = inliers.ravel()
    if len(ix) >= 6:
        rvec, tvec = cv2.solvePnPRefineLM(np.asarray(points_3d)[ix],
                                          np.asarray(points_2d)[ix], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return (R, tvec.ravel()), ix


def reconstruct(images, K, max_features=3500, min_matches=25, verbose=True,
                adaptive=False, candidate_window=10, min_parallax_deg=1.0):
    """SfM baseline or adaptive geometry-aware image pair selection.

    adaptive=False deliberately preserves the old algorithm and parameters.
    adaptive=True increases candidate reach, evaluates seed geometry, and retries
    registration against additional already-registered neighbours.
    """
    kps, desc = extract_features(images, max_features)
    n = len(images)
    # Precompute matches for each neighbour and a small window of earlier images.
    pairs = {}
    window = max(3, int(candidate_window)) if adaptive else 3
    for i in range(n):
        for j in range(i+1, min(n, i+window+1)):
            matches = match_features(desc[i], desc[j])
            # Adaptive only: pre-filter geometrically inconsistent feature matches.
            # Original mode is deliberately left unchanged for a fair baseline.
            if adaptive and len(matches) >= 12:
                p1, p2 = kps[i][matches[:, 0]], kps[j][matches[:, 1]]
                F, inliers = cv2.findFundamentalMat(
                    p1, p2, cv2.FM_RANSAC, 1.5, 0.999)
                if F is None or inliers is None:
                    matches = np.empty((0, 2), dtype=int)
                else:
                    matches = matches[inliers.ravel().astype(bool)]
            pairs[i, j] = matches
    # Choose a strong seed among the first few nearby pairs, not arbitrary endpoints.
    seed_options = []
    seed_diagnostics = []
    # For adaptive mode evaluate promising pairs near the beginning of the sequence.
    # Restrict candidate geometry checks to avoid costly O(N^2) processing.
    candidate_pairs = [((i, j), matches) for (i, j), matches in pairs.items()
                       if (i < 15 if adaptive else j-i <= 2)
                       and len(matches) >= min_matches]
    if adaptive:
        candidate_pairs = sorted(candidate_pairs, key=lambda item: len(item[1]), reverse=True)[:65]
    for (i, j), matches in candidate_pairs:
        pose = relative_pose(kps[i][matches[:, 0]], kps[j][matches[:, 1]], K)
        if pose is None:
            continue
        R, t, mask = pose
        if mask.sum() < min_matches:
            continue
        inliers = matches[mask]
        if adaptive:
            trial_poses = ((np.eye(3), np.zeros(3)), (R, t))
            trial_xyz, valid, trial_err = triangulate(
                kps[i][inliers[:, 0]], kps[j][inliers[:, 1]],
                trial_poses[0], trial_poses[1], K)
            if valid.sum() < min_matches:
                continue
            # Rays from each camera centre to the same triangulated 3D point.
            c1 = np.zeros(3)
            c2 = -R.T @ t
            va = trial_xyz[valid] - c1
            vb = trial_xyz[valid] - c2
            cosang = np.sum(va * vb, axis=1) / np.maximum(
                np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-12)
            angles = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            median_angle = float(np.median(angles))
            median_error = float(np.median(trial_err[valid]))
            # A configurable scoring experiment: reward useful triangulation
            # geometry and inliers while penalizing poor reprojection.
            parallax_score = min(median_angle / 2.0, 1.0)
            if median_angle > 20.0:
                parallax_score *= 20.0 / median_angle
            quality = (float(valid.sum()) ** 0.5) * parallax_score / (1.0 + median_error)
            seed_options.append((quality, i, j, R, t, inliers))
            seed_diagnostics.append({'image_a':i+1, 'image_b':j+1,
                'ransac_pose_inliers':int(mask.sum()), 'valid_points':int(valid.sum()),
                'median_parallax_deg':round(median_angle, 4),
                'median_reprojection_px':round(median_error, 4),
                'quality_score':round(quality, 4)})
        else:
            seed_options.append((int(mask.sum()), i, j, R, t, inliers))
    if not seed_options:
        raise RuntimeError('No valid initial image pair. Try a more textured static scene, better overlap, or --focal.')
    # Prioritise beginning of sequence, to maintain an interpretable walkthrough.
    if adaptive:
        seed_options.sort(key=lambda v: v[0], reverse=True)
    else:
        seed_options.sort(key=lambda v: (v[1] <= 1, v[0]), reverse=True)
    _, a, b, R, t, seed_matches = seed_options[0]
    poses = {a: (np.eye(3), np.zeros(3)), b: (R, t)}
    xyz, valid, _ = triangulate(kps[a][seed_matches[:, 0]],
                               kps[b][seed_matches[:, 1]], poses[a], poses[b], K,
                               min_angle_deg=min_parallax_deg if adaptive else 0.0)
    points, colors, observations = [], [], []
    track_to_id = {}
    def add_point(p, image_index, kp_index, other_index, other_kp):
        point_id = len(points)
        points.append(p)
        xy = np.round(kps[image_index][kp_index]).astype(int)
        h, w = images[image_index].shape[:2]
        colors.append(images[image_index][np.clip(xy[1], 0, h-1), np.clip(xy[0], 0, w-1)])
        observations.append({image_index: int(kp_index), other_index: int(other_kp)})
        track_to_id[image_index, int(kp_index)] = point_id
        track_to_id[other_index, int(other_kp)] = point_id
    for pair, pt, keep in zip(seed_matches, xyz, valid):
        if keep:
            add_point(pt, a, int(pair[0]), b, int(pair[1]))
    if verbose:
        print(f'Seed images {a+1}, {b+1}: {len(points)} valid 3D points')
    remaining = [i for i in range(n) if i not in poses]
    registration_log = []
    # Repeated attempts permit images neighbouring both sides of the seed to register.
    for _ in range(6 if adaptive else 2):
        progressed = False
        # The GitHub SfM driver ranks unregistered views by the number of
        # available 2D-to-3D correspondences. Do the same only in adaptive
        # mode; preserve the original registration order as a control.
        def registration_evidence(j):
            evidence = set()
            for i in poses:
                key = (min(i, j), max(i, j))
                if key not in pairs:
                    continue
                match = pairs[key] if i < j else pairs[key][:, ::-1]
                for ik, jk in match:
                    if (i, int(ik)) in track_to_id:
                        evidence.add(int(jk))
            return len(evidence)
        order = (sorted(remaining, key=registration_evidence, reverse=True)
                 if adaptive else list(remaining))
        for j in order:
            related = [i for i in poses if (min(i,j),max(i,j)) in pairs]
            # Adaptive: require consistent 2D-to-3D track evidence across references.
            # Original mode preserves the previous last-write-wins behaviour.
            candidates = {}
            proposals = {}
            for i in related:
                match = pairs[min(i,j), max(i,j)]
                if i > j:
                    match = match[:, ::-1]
                for ik, jk in match:
                    point_id = track_to_id.get((i, int(ik)))
                    if point_id is not None:
                        if adaptive:
                            proposals.setdefault(int(jk), []).append(point_id)
                        else:
                            candidates[int(jk)] = point_id
            if adaptive:
                # Reject ambiguous proposals (several 3D points for one 2D keypoint).
                # Also enforce a one-to-one mapping from 3D tracks to 2D keypoints.
                used_points = set()
                for jk0, pts in sorted(proposals.items(), key=lambda e: -len(e[1])):
                    if len(set(pts)) == 1 and pts[0] not in used_points:
                        candidates[jk0] = pts[0]
                        used_points.add(pts[0])
            if len(candidates) < 12:
                continue
            jk, pid = zip(*candidates.items())
            registration = register_pnp(np.array(points)[list(pid)], kps[j][list(jk)], K)
            if registration is None:
                continue
            poses[j], inlier_ids = registration
            # Record the observations validated by PnP.
            for ix in inlier_ids:
                ki, pi = jk[ix], pid[ix]
                track_to_id[j, int(ki)] = pi
                observations[pi][j] = int(ki)
            # Triangulate previously unseen tracks against already registered cameras.
            added = 0
            for i in sorted(related, key=lambda i: abs(i-j)):
                match = pairs[min(i,j), max(i,j)]
                if i > j:
                    match = match[:, ::-1]
                available = [(int(ki),int(kj)) for ki,kj in match
                             if (i,int(ki)) not in track_to_id and (j,int(kj)) not in track_to_id]
                if len(available) < 2:
                    continue
                arr = np.asarray(available, int)
                coords, good, _ = triangulate(
                    kps[i][arr[:,0]], kps[j][arr[:,1]], poses[i], poses[j], K,
                    min_angle_deg=min_parallax_deg if adaptive else 0.0)
                for pair, pt, keep in zip(arr, coords, good):
                    if keep:
                        add_point(pt, i, int(pair[0]), j, int(pair[1]))
                        added += 1
            registration_log.append({'image': j+1,
                'pnp_inliers': int(len(inlier_ids)),
                'new_3d_points': int(added),
                'reference_images': int(len(related))})
            remaining.remove(j)
            progressed = True
            if verbose:
                print(f'Image {j+1} registered: PnP inliers={len(inlier_ids)}, new points={added}')
        if not progressed:
            break
    model = Model(np.array(points, float).reshape(-1,3),
                  np.array(colors, np.uint8).reshape(-1,3), poses, kps,
                  observations, K, sorted(poses),
                  {'total_images':n, 'seed_images': [a+1,b+1],
                   'unregistered_images':[i+1 for i in remaining],
                   'adaptive':bool(adaptive), 'candidate_window':window,
                    'min_parallax_deg':min_parallax_deg if adaptive else 0.0,
                   'seed_candidates':seed_diagnostics, 'registration_log':registration_log})
    return model, pairs


def reprojection_errors(model):
    """Return pixel errors between observed feature locations and projected 3D points."""
    errors = []
    for p, obs in enumerate(model.observations):
        for idx, key_id in obs.items():
            if idx not in model.cameras:
                continue
            R,t = model.cameras[idx]
            q = model.K @ (R @ model.points[p] + t)
            if q[2] > 1e-8:
                errors.append(np.linalg.norm(q[:2]/q[2] - model.keypoints[idx][key_id]))
    return np.asarray(errors)


def bundle_adjust(model, max_points=180, max_nfev=25):
    """Optional SMALL sparse BA demo. Refines selected points and camera poses only.

    Cap keeps class demos practical. The first registered camera is fixed to remove
    coordinate-frame ambiguity; monocular global scale remains unconstrained.
    """
    cam_ids = model.registered
    if len(cam_ids) < 2 or len(model.points) < 12:
        return False
    # Prefer points with multiple observations.
    selected = sorted(range(len(model.points)),
                      key=lambda i: len(model.observations[i]), reverse=True)[:max_points]
    ptmap = {orig:k for k, orig in enumerate(selected)}
    active = cam_ids[1:]
    cammap = {id:k for k,id in enumerate(active)}
    obs_cam,obs_pt, obs_xy = [],[],[]
    for orig in selected:
        for ci,ki in model.observations[orig].items():
            if ci in model.cameras:
                obs_cam.append(ci); obs_pt.append(ptmap[orig]); obs_xy.append(model.keypoints[ci][ki])
    if len(obs_cam) < 30:
        return False
    obs_xy = np.asarray(obs_xy)
    x0 = []
    for ci in active:
        R,t = model.cameras[ci]
        rv,_ = cv2.Rodrigues(R)
        x0.extend(rv.ravel()); x0.extend(t.ravel())
    x0 = np.r_[np.asarray(x0), model.points[selected].ravel()]
    ncam = len(active)
    def residual(v):
        xyz = v[6*ncam:].reshape(-1,3)
        camera_params = {}
        for ci in cam_ids:
            if ci == cam_ids[0]:
                camera_params[ci] = model.cameras[ci]
            else:
                k = cammap[ci]
                R,_ = cv2.Rodrigues(v[6*k:6*k+3])
                camera_params[ci]=(R,v[6*k+3:6*k+6])
        result = []
        for ci,pi,xy in zip(obs_cam,obs_pt,obs_xy):
            R,t = camera_params[ci]
            q = model.K @ (R@xyz[pi]+t)
            uv = q[:2] / (q[2] if abs(q[2]) > 1e-8 else 1e-8)
            result.extend(uv-xy)
        return np.asarray(result)
    pattern = lil_matrix((2*len(obs_pt), len(x0)), dtype=np.int8)
    for oi,(ci,pi) in enumerate(zip(obs_cam, obs_pt)):
        if ci in cammap:
            pattern[2*oi:2*oi+2, 6*cammap[ci]:6*cammap[ci]+6] = 1
        pattern[2*oi:2*oi+2, 6*ncam+3*pi:6*ncam+3*pi+3] = 1
    before = np.median(np.abs(residual(x0)))
    result = least_squares(residual, x0, jac_sparsity=pattern, method='trf',
                           x_scale='jac', loss='soft_l1', f_scale=1.,
                           max_nfev=max_nfev, ftol=1e-3)
    after = np.median(np.abs(residual(result.x)))
    if not np.isfinite(after) or after >= before:
        return False
    for ci in active:
        k=cammap[ci]
        R,_ = cv2.Rodrigues(result.x[6*k:6*k+3])
        model.cameras[ci] = (R,result.x[6*k+3:6*k+6])
    model.points[selected] = result.x[6*ncam:].reshape(-1,3)
    return True


def refine_points_fixed_cameras(model, max_points=1500, max_nfev=30):
    """Point-only refinement that NEVER changes camera poses.

    This prevents the previous partial BA problem: moving all cameras using
    a tiny subset of points while leaving most points fixed.  It does NOT fix
    camera drift. Each point must have observations in >= 2 registered cameras.
    Returns number of accepted point updates.
    """
    selected = sorted(range(len(model.points)),
                      key=lambda i: len(model.observations[i]), reverse=True)[:max_points]
    accepted = 0
    for p in selected:
        obs = [(ci, model.keypoints[ci][ki]) for ci, ki in model.observations[p].items()
               if ci in model.cameras]
        if len(obs) < 2:
            continue
        original = model.points[p].copy()
        def residual(x):
            values = []
            for ci, xy in obs:
                R, t = model.cameras[ci]
                cam = R @ x + t
                if cam[2] <= 1e-6:
                    return np.full(2 * len(obs), 1e5)
                q = model.K @ cam
                values.extend(q[:2]/q[2] - xy)
            return np.asarray(values)
        before = residual(original)
        if not np.isfinite(before).all():
            continue
        try:
            fit = least_squares(residual, original, loss='soft_l1',
                                f_scale=1., max_nfev=max_nfev)
        except (ValueError, np.linalg.LinAlgError):
            continue
        after = residual(fit.x)
        if np.isfinite(after).all() and np.mean(after**2) < np.mean(before**2):
            model.points[p] = fit.x
            accepted += 1
    return accepted
