"""Geometry-Aware Adaptive Image Selection for incremental SfM (our implementation).

Built ON TOP of the primitives in sfm_core.py (matching, essential matrix,
triangulation, PnP). sfm_core.reconstruct(adaptive=False) is NOT modified, so the
Original baseline stays reproducible. Nothing here is claimed as novel research:
it combines well-known ideas (retrieval-based candidate pairs, next-best-view by
2D-3D evidence, baseline-aware triangulation, local BA) and measures each one.

Five switchable components (see Options / --ablate):
  retrieval : candidate pairs come from index window  +  coarse all-pairs visual
              similarity (top-k most similar images), so images that REVISIT a place can be linked
              even when their file index is far apart (loop closure).
  seed      : seed pair scored by inliers, parallax, reprojection AND connectivity.
  order     : next image = the one with the most reliable 2D-3D correspondences.
  vote      : a keypoint seen by several cameras takes the MAJORITY 3D point instead
              of being thrown away when its proposals disagree (track consistency).
  baseline  : new points are triangulated against the reference camera with the
              LARGEST baseline first (so the parallax filter does not starve tracks).
  localba   : sliding-window local bundle adjustment with an acceptance test.
"""
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from sfm_core import (Model, match_features, relative_pose,
                      triangulate, register_pnp)

ALL_COMPONENTS = ('retrieval', 'seed', 'order', 'vote', 'baseline', 'localba')


@dataclass
class Options:
    """Which adaptive components are on, plus their (shared, documented) thresholds."""
    retrieval: bool = True
    seed: bool = True
    order: bool = True
    vote: bool = True
    baseline: bool = True
    localba: bool = True
    window: int = 3            # index window (same default as Original)
    topk: int = 8              # retrieval candidates per image
    min_matches: int = 25      # seed pair minimum inliers (same as Original)
    min_pair_matches: int = 15  # keep a verified pair if it has >= this many inliers
    min_loop_matches: int = 30  # stricter bar for retrieval (non-window) pairs
    min_parallax_deg: float = 1.0
    min_corr: int = 12         # minimum 2D-3D correspondences before trying PnP
    min_pnp_inliers: int = 12
    ba_window: int = 8         # cameras freed in a local BA
    ba_every: int = 5          # run local BA after every k registrations
    ba_max_nfev: int = 15

    @staticmethod
    def from_ablate(spec):
        """spec=None -> all on; 'none' -> all off; 'a,b' -> only those on."""
        if spec is None:
            return Options()
        on = set() if spec.strip().lower() == 'none' else {s.strip() for s in spec.split(',') if s.strip()}
        bad = on - set(ALL_COMPONENTS)
        if bad:
            raise ValueError(f'Unknown component(s) {sorted(bad)}; choose from {ALL_COMPONENTS}')
        return Options(**{c: (c in on) for c in ALL_COMPONENTS})


# --------------------------------------------------------------------------- #
# 1. View graph: which image pairs deserve matching?
# --------------------------------------------------------------------------- #
def extract_features_ranked(images, nfeatures=3500):
    """Same SIFT detector/settings as sfm_core.extract_features, but keypoints are sorted
    by detector response (strongest first). The feature SET is identical; only the index
    order differs, so the coarse retrieval step can cheaply use the strongest ones."""
    det = cv2.SIFT_create(nfeatures=nfeatures)
    kps, desc = [], []
    for im in images:
        k, d = det.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None)
        order = np.argsort([-x.response for x in k])
        kps.append(np.array([k[i].pt for i in order], np.float64).reshape(-1, 2))
        desc.append(d[order] if d is not None else None)
    return kps, desc


def view_similarity(desc, n_coarse=1000, ratio=0.8):
    """Coarse all-pairs image similarity = number of Lowe-ratio matches between the
    `n_coarse` first (strongest) descriptors of every image pair, symmetrised.

    Cheap (a few seconds for ~130 images) and exact enough to find images that look at
    the same place even when they are far apart in the file order. For thousands of
    images one would replace this by a vocabulary tree / global descriptor."""
    n = len(desc)
    D = [None if d is None else d[:n_coarse].astype(np.float32) for d in desc]
    sq = [None if d is None else (d ** 2).sum(1) for d in D]
    S = np.zeros((n, n), np.float32)
    for i in range(n):
        if D[i] is None or len(D[i]) < 2:
            continue
        for j in range(i + 1, n):
            if D[j] is None or len(D[j]) < 2:
                continue
            d2 = sq[i][:, None] + sq[j][None] - 2 * D[i] @ D[j].T
            two = np.partition(d2, 1, axis=1)[:, :2]
            good_ij = np.sqrt(np.maximum(two[:, 0], 0)) < ratio * np.sqrt(np.maximum(two[:, 1], 1e-12))
            two = np.partition(d2, 1, axis=0)[:2]
            good_ji = np.sqrt(np.maximum(two[0], 0)) < ratio * np.sqrt(np.maximum(two[1], 1e-12))
            S[i, j] = S[j, i] = 0.5 * (good_ij.sum() + good_ji.sum())
    return S


def build_view_graph(kps, desc, opt, verbose=True):
    """Match + geometrically verify candidate pairs. Returns {(i,j): Nx2 matches}, i<j."""
    n = len(kps)
    cand, loop = set(), set()
    for i in range(n):
        for j in range(i + 1, min(n, i + opt.window + 1)):
            cand.add((i, j))
    if opt.retrieval and n > 2:
        sim = view_similarity(desc)
        np.fill_diagonal(sim, -1)
        for i in range(n):
            added = 0
            for j in np.argsort(-sim[i]):
                if abs(int(j) - i) <= opt.window:
                    continue
                key = (min(i, int(j)), max(i, int(j)))
                cand.add(key)
                loop.add(key)
                added += 1
                if added >= opt.topk:
                    break
    pairs = {}
    for (i, j) in sorted(cand):
        m = match_features(desc[i], desc[j])
        if len(m) < 12:
            continue
        F, inl = cv2.findFundamentalMat(kps[i][m[:, 0]], kps[j][m[:, 1]], cv2.FM_RANSAC, 1.5, 0.999)
        if F is None or inl is None:
            continue
        m = m[inl.ravel().astype(bool)]
        need = opt.min_loop_matches if ((i, j) in loop and j - i > opt.window) else opt.min_pair_matches
        if len(m) >= need:
            pairs[i, j] = m
    if verbose:
        n_loop = sum(1 for k in pairs if k[1] - k[0] > opt.window)
        print(f'View graph: {len(cand)} candidate pairs -> {len(pairs)} verified '
              f'({n_loop} non-sequential)')
    return pairs


def pair_matches(pairs, i, j):
    """Matches between images i and j, oriented as (keypoint in i, keypoint in j)."""
    if i < j:
        return pairs.get((i, j))
    m = pairs.get((j, i))
    return None if m is None else m[:, ::-1]


# --------------------------------------------------------------------------- #
# 2. Geometry helpers
# --------------------------------------------------------------------------- #
def camera_center(pose):
    return -pose[0].T @ pose[1]


def triangulation_angle(xyz, pose1, pose2):
    """Angle (deg) at each 3D point between the rays to the two camera centres."""
    v1, v2 = xyz - camera_center(pose1), xyz - camera_center(pose2)
    cosang = np.einsum('ij,ij->i', v1, v2) / np.maximum(
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def select_seed(kps, pairs, K, opt, n_eval=40):
    """Choose the initial pair. Returns (a, b, R, t, inlier_matches, diagnostics).

    adaptive (opt.seed): score = sqrt(#valid points) * parallax_score * log(1+connectivity)
                         / (1 + median reprojection error), over the best `n_eval` pairs.
    otherwise          : Original-style - most pose inliers among pairs with |i-j| <= 2."""
    nb = {}
    for (i, j) in pairs:
        nb.setdefault(i, set()).add(j)
        nb.setdefault(j, set()).add(i)
    ranked = sorted(((k, m) for k, m in pairs.items() if len(m) >= opt.min_matches),
                    key=lambda it: -len(it[1]))
    if not opt.seed:
        ranked = [it for it in ranked if it[0][1] - it[0][0] <= 2]
    best, diag = None, []
    for (i, j), m in ranked[:n_eval if opt.seed else 10 ** 9]:
        pose = relative_pose(kps[i][m[:, 0]], kps[j][m[:, 1]], K)
        if pose is None:
            continue
        R, t, mask = pose
        if mask.sum() < opt.min_matches:
            continue
        inl = m[mask]
        if not opt.seed:
            score = float(mask.sum())
        else:
            xyz, valid, err = triangulate(kps[i][inl[:, 0]], kps[j][inl[:, 1]],
                                          (np.eye(3), np.zeros(3)), (R, t), K)
            if valid.sum() < opt.min_matches:
                continue
            ang = float(np.median(triangulation_angle(xyz[valid], (np.eye(3), np.zeros(3)), (R, t))))
            par = min(ang / 2.0, 1.0) * (20.0 / ang if ang > 20 else 1.0)
            conn = len(nb.get(i, set()) | nb.get(j, set()))
            score = (valid.sum() ** 0.5) * par * np.log1p(conn) / (1.0 + float(np.median(err[valid])))
            diag.append({'image_a': i + 1, 'image_b': j + 1, 'pose_inliers': int(mask.sum()),
                         'valid_points': int(valid.sum()), 'median_parallax_deg': round(ang, 3),
                         'connectivity': conn, 'score': round(float(score), 3)})
        if best is None or score > best[0]:
            best = (score, i, j, R, t, inl)
    if best is None:
        raise RuntimeError('No valid initial image pair. Check overlap / --focal.')
    return best[1], best[2], best[3], best[4], best[5], diag


# --------------------------------------------------------------------------- #
# 3. Track voting (vectorised)
# --------------------------------------------------------------------------- #
def vote_correspondences(jk_all, pid_all, n_kp, n_pts, use_vote=True, ratio=0.6):
    """Turn raw proposals (keypoint jk -> 3D point pid, one per reference camera) into a
    consistent one-to-one 2D-3D assignment.

    use_vote=True : keep the majority 3D point if it has >= `ratio` of the votes.
    use_vote=False: keep only unanimous proposals (Original adaptive-v1 behaviour).
    Returns (keypoints, point_ids, n_ambiguous_dropped)."""
    e = np.empty(0, np.int64)
    if len(jk_all) == 0:
        return e, e, 0
    jk, pid = np.asarray(jk_all, np.int64), np.asarray(pid_all, np.int64)
    total = np.bincount(jk, minlength=n_kp)
    key = jk * (n_pts + 1) + pid
    uk, cnt = np.unique(key, return_counts=True)
    kj, kp = uk // (n_pts + 1), uk % (n_pts + 1)
    o = np.lexsort((-cnt, kj))
    kj, kp, cnt = kj[o], kp[o], cnt[o]
    first = np.r_[True, kj[1:] != kj[:-1]]
    kj, kp, cnt = kj[first], kp[first], cnt[first]
    ok = cnt >= (ratio * total[kj] if use_vote else total[kj])
    dropped = int((~ok).sum())
    kj, kp, cnt = kj[ok], kp[ok], cnt[ok]
    o = np.argsort(-cnt, kind='stable')           # one keypoint per 3D point
    kj, kp = kj[o], kp[o]
    _, keep = np.unique(kp, return_index=True)
    return kj[keep], kp[keep], dropped


# --------------------------------------------------------------------------- #
# 4. Local bundle adjustment (vectorised, sparse Jacobian, acceptance test)
# --------------------------------------------------------------------------- #
def _rotate(rv, P):
    """Rodrigues rotation of points P (N,3) by rotation vectors rv (N,3)."""
    th = np.linalg.norm(rv, axis=1, keepdims=True)
    k = rv / np.maximum(th, 1e-12)
    c, s = np.cos(th), np.sin(th)
    return P * c + np.cross(k, P) * s + k * np.sum(k * P, axis=1, keepdims=True) * (1 - c)


def local_bundle_adjust(poses, points, observations, kps, K, free, max_nfev=15):
    """Refine the cameras in `free` AND every 3D point they observe; all other cameras
    are held fixed (this fixes gauge and scale, and keeps the rest of the model
    consistent). Robust Huber loss; result is applied ONLY if the mean pixel error on the
    very same observations decreases and the median does not get worse.
    Returns a log dict. This is local BA, not a global solve - drift in old cameras
    is not corrected."""
    free = [c for c in free if c in poses]
    fset = set(free)
    pids = [p for p, obs in enumerate(observations)
            if len(obs) >= 2 and any(c in fset for c in obs)]
    if not free or len(pids) < 20:
        return {'ran': False}
    pmap = {p: k for k, p in enumerate(pids)}
    oc, op, oxy = [], [], []
    for p in pids:
        for c, k in observations[p].items():
            if c in poses:
                oc.append(c); op.append(pmap[p]); oxy.append(kps[c][k])
    oc, op, oxy = np.array(oc), np.array(op), np.array(oxy)
    used = sorted(set(oc.tolist()))
    u = {c: k for k, c in enumerate(used)}
    free = [c for c in free if c in u]
    fi_map = {c: k for k, c in enumerate(free)}
    nf = len(free)
    rv_u = np.array([cv2.Rodrigues(poses[c][0])[0].ravel() for c in used])
    t_u = np.array([poses[c][1] for c in used])
    ou = np.array([u[c] for c in oc])
    ofi = np.array([fi_map.get(c, -1) for c in oc])
    X0 = np.array([points[p] for p in pids])
    x0 = np.concatenate([np.concatenate([rv_u[u[c]], t_u[u[c]]]) for c in free] + [X0.ravel()])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    N = len(op)

    def residual(x):
        cams, P = x[:6 * nf].reshape(nf, 6), x[6 * nf:].reshape(-1, 3)
        rv, t = rv_u[ou].copy(), t_u[ou].copy()
        m = ofi >= 0
        rv[m], t[m] = cams[ofi[m], :3], cams[ofi[m], 3:]
        Xc = _rotate(rv, P[op]) + t
        z = np.where(Xc[:, 2] < 1e-6, 1e-6, Xc[:, 2])        # behind camera -> huge residual
        uv = np.column_stack([fx * Xc[:, 0] / z + cx, fy * Xc[:, 1] / z + cy])
        return (uv - oxy).ravel()

    o = np.arange(N)
    m = ofi >= 0
    rows, cols = [], []
    for d in (0, 1):
        r = 2 * o + d
        for k in range(3):
            rows.append(r); cols.append(6 * nf + 3 * op + k)
        for k in range(6):
            rows.append(r[m]); cols.append(6 * ofi[m] + k)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(2 * N, len(x0))).tocsr()

    before = np.linalg.norm(residual(x0).reshape(-1, 2), axis=1)
    fit = least_squares(residual, x0, jac_sparsity=A, method='trf', x_scale='jac',
                        loss='huber', f_scale=1.0, max_nfev=max_nfev, ftol=1e-4)
    after = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
    accepted = bool(np.isfinite(after).all() and after.mean() < before.mean()
                    and np.median(after) <= 1.05 * np.median(before))
    if accepted:
        for c in free:
            k = fi_map[c]
            poses[c] = (cv2.Rodrigues(fit.x[6 * k:6 * k + 3])[0], fit.x[6 * k + 3:6 * k + 6].copy())
        P = fit.x[6 * nf:].reshape(-1, 3)
        for p, k in pmap.items():
            points[p] = P[k]
    return {'ran': True, 'accepted': accepted, 'free_cameras': nf, 'points': len(pids),
            'observations': N, 'mean_before_px': round(float(before.mean()), 4),
            'mean_after_px': round(float(after.mean()), 4),
            'median_before_px': round(float(np.median(before)), 4),
            'median_after_px': round(float(np.median(after)), 4)}


# --------------------------------------------------------------------------- #
# 5. Incremental reconstruction with next-best-view selection
# --------------------------------------------------------------------------- #
def incremental_sfm(kps, pairs, K, opt, images=None, verbose=True):
    """Incremental SfM on a prepared view graph. Returns a sfm_core.Model."""
    n = len(kps)
    a, b, R, t, seed_m, seed_diag = select_seed(kps, pairs, K, opt)
    poses = {a: (np.eye(3), np.zeros(3)), b: (R, t)}
    points, colors, observations = [], [], []
    kp2pid = [np.full(len(k), -1, np.int64) for k in kps]       # keypoint -> 3D point id
    stats = {'ambiguous_dropped': 0, 'parallax_rejected': 0, 'pnp_failures': 0}

    def add_points(i, ik, j, jk, xyz):
        for p, ki, kj in zip(xyz, ik, jk):
            pid = len(points)
            points.append(p)
            if images is not None:
                h, w = images[i].shape[:2]
                x, y = np.round(kps[i][ki]).astype(int)
                colors.append(images[i][np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)])
            else:
                colors.append(np.zeros(3, np.uint8))
            observations.append({i: int(ki), j: int(kj)})
            kp2pid[i][ki] = pid
            kp2pid[j][kj] = pid

    def triangulate_pair(i, j, min_par):
        m = pair_matches(pairs, i, j)
        if m is None:
            return 0
        free = (kp2pid[i][m[:, 0]] < 0) & (kp2pid[j][m[:, 1]] < 0)
        m = m[free]
        if len(m) < 2:
            return 0
        xyz, good, _ = triangulate(kps[i][m[:, 0]], kps[j][m[:, 1]], poses[i], poses[j], K)
        ang = triangulation_angle(xyz, poses[i], poses[j])
        keep = good & (ang >= min_par)
        stats['parallax_rejected'] += int((good & ~keep).sum())
        add_points(i, m[keep, 0], j, m[keep, 1], xyz[keep])
        return int(keep.sum())

    # seed points (parallax filter applies to the seed too)
    xyz, good, _ = triangulate(kps[a][seed_m[:, 0]], kps[b][seed_m[:, 1]], poses[a], poses[b], K)
    keep = good & (triangulation_angle(xyz, poses[a], poses[b]) >= opt.min_parallax_deg)
    add_points(a, seed_m[keep, 0], b, seed_m[keep, 1], xyz[keep])
    if verbose:
        print(f'Seed images {a + 1}, {b + 1}: {len(points)} valid 3D points')

    def correspondences(j):
        jk_all, pid_all = [], []
        for i in poses:
            m = pair_matches(pairs, i, j)
            if m is None:
                continue
            pid = kp2pid[i][m[:, 0]]
            ok = pid >= 0
            jk_all.append(m[ok, 1]); pid_all.append(pid[ok])
        if not jk_all:
            return np.empty(0, np.int64), np.empty(0, np.int64), 0
        return vote_correspondences(np.concatenate(jk_all), np.concatenate(pid_all),
                                    len(kps[j]), len(points), opt.vote)

    remaining = [i for i in range(n) if i not in poses]
    reg_order, reg_log, ba_log = [a, b], [], []
    failed = {}                       # image -> number of cameras when PnP last failed
    since_ba = 0
    while remaining:
        scored = []
        for j in remaining:
            if failed.get(j) == len(poses):
                continue
            jk, pid, amb = correspondences(j)
            if len(jk) >= opt.min_corr:
                scored.append((len(jk), j, jk, pid, amb))
        if not scored:
            break
        scored.sort(key=lambda s: -s[0] if opt.order else s[1])
        progressed = False
        for nc, j, jk, pid, amb in scored:
            reg = register_pnp(np.array(points)[pid], kps[j][jk], K)
            if reg is None or len(reg[1]) < opt.min_pnp_inliers:
                failed[j] = len(poses)
                stats['pnp_failures'] += 1
                continue
            poses[j], ix = reg
            stats['ambiguous_dropped'] += amb
            for k, p in zip(jk[ix], pid[ix]):
                kp2pid[j][k] = p
                observations[p][j] = int(k)
            related = [i for i in poses if i != j and pair_matches(pairs, i, j) is not None]
            if opt.baseline:
                cj = camera_center(poses[j])
                related.sort(key=lambda i: -np.linalg.norm(camera_center(poses[i]) - cj))
            else:
                related.sort(key=lambda i: abs(i - j))
            added = sum(triangulate_pair(i, j, opt.min_parallax_deg) for i in related)
            reg_log.append({'image': j + 1, 'corr_2d3d': int(nc), 'pnp_inliers': int(len(ix)),
                            'confidence': round(len(ix) / nc, 3), 'new_3d_points': int(added),
                            'reference_images': len(related)})
            reg_order.append(j)
            remaining.remove(j)
            since_ba += 1
            progressed = True
            if verbose:
                print(f'Image {j + 1} registered: 2D-3D={nc} PnP inliers={len(ix)} new points={added}')
            if opt.localba and since_ba >= opt.ba_every:
                free = [c for c in reg_order[-opt.ba_window:] if c not in (a, b)]
                ba_log.append(local_bundle_adjust(poses, points, observations, kps, K, free,
                                                  opt.ba_max_nfev))
                ba_log[-1]['after_image'] = j + 1
                since_ba = 0
                if verbose and ba_log[-1].get('ran'):
                    L = ba_log[-1]
                    print(f"  local BA: mean {L['mean_before_px']:.3f} -> {L['mean_after_px']:.3f} px "
                          f"({'accepted' if L['accepted'] else 'rejected'})")
            break                     # re-score every remaining image after each registration
        if not progressed:
            break
    if opt.localba and since_ba > 0:
        free = [c for c in reg_order[-opt.ba_window:] if c not in (a, b)]
        ba_log.append(local_bundle_adjust(poses, points, observations, kps, K, free, opt.ba_max_nfev))
        ba_log[-1]['after_image'] = 'final'

    diagnostics = {'total_images': n, 'seed_images': [a + 1, b + 1],
                   'unregistered_images': [i + 1 for i in remaining], 'adaptive': True,
                   'candidate_window': opt.window, 'min_parallax_deg': opt.min_parallax_deg,
                   'seed_candidates': seed_diag, 'registration_log': reg_log,
                   'track_stats': stats, 'local_ba_log': ba_log,
                   'components': {c: getattr(opt, c) for c in ALL_COMPONENTS}}
    model = Model(np.array(points, float).reshape(-1, 3), np.array(colors, np.uint8).reshape(-1, 3),
                  poses, kps, observations, K, sorted(poses), diagnostics)
    return model


def reconstruct_adaptive(images, K, max_features=3500, opt=None, verbose=True):
    """Full adaptive pipeline: SIFT -> view graph -> seed -> next-best-view -> local BA."""
    opt = opt or Options()
    kps, desc = extract_features_ranked(images, max_features)
    pairs = build_view_graph(kps, desc, opt, verbose)
    return incremental_sfm(kps, pairs, K, opt, images, verbose), pairs


# --------------------------------------------------------------------------- #
# 6. Validation utilities (used by compare_three.py and tests)
# --------------------------------------------------------------------------- #
def umeyama_sim3(src, dst):
    """Least-squares similarity transform dst ~= s * R @ src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    sgn = np.sign(np.linalg.det(U @ Vt))
    E = np.diag([1, 1, sgn])
    R = U @ E @ Vt
    s = (sig * np.diag(E)).sum() / ((S ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


def ate_after_sim3(est_centres, ref_centres):
    """Absolute trajectory error of camera centres after Sim(3) alignment.
    Returns (rmse, median, scale, n)."""
    s, R, t = umeyama_sim3(est_centres, ref_centres)
    err = np.linalg.norm((s * (R @ est_centres.T)).T + t - ref_centres, axis=1)
    return float(np.sqrt((err ** 2).mean())), float(np.median(err)), float(s), len(err)


def check_track_consistency(model):
    """Return a dict of integrity violations (all zeros == consistent)."""
    seen, dup_kp, bad_cam, short = {}, 0, 0, 0
    for pid, obs in enumerate(model.observations):
        if len(obs) < 2:
            short += 1
        for c, k in obs.items():
            if c not in model.cameras:
                bad_cam += 1
            if (c, k) in seen:
                dup_kp += 1               # same keypoint claimed by two 3D points
            seen[c, k] = pid
    return {'keypoint_in_two_points': dup_kp, 'obs_in_unregistered_camera': bad_cam,
            'points_with_lt2_obs': short}
