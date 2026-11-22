"""Print a fair, auditable SfM comparison from measured outputs.

Run COLMAP on the same input files and export its sparse model to TXT first (run_colmap.py does it).

Metric honesty
  DIRECTLY comparable : registered images, registration %, runtime (same machine), pose error vs COLMAP.
  NOT strictly equal  : reprojection error (Python = mean over observations of OUR model;
                        COLMAP = mean of its per-point ERROR field over ITS points).
  Pose error (ATE)    : camera centres of images registered by BOTH methods, aligned with a
                        Sim(3) (Umeyama) because monocular SfM has arbitrary scale/frame. COLMAP is
                        the reference, NOT ground truth, so ATE measures agreement with COLMAP.

Usage:
  python compare_three.py --original outputs/original --adaptive outputs/adaptive \\
         --colmap-txt outputs/colmap_final_txt --extra "no-BA=outputs/abl_noba" --extra "retrieval only=outputs/abl_ret"
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from adaptive_sfm import ate_after_sim3


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def read_python_report(path):
    r = json.loads((path / 'report.json').read_text())
    out = {'input_images': r['images_total'], 'registered': r['images_registered'],
           'points': r['3d_points'], 'error': r['reprojection_error_mean_px'],
           'median_error': r.get('reprojection_error_median_px'),
           'runtime': r.get('runtime_seconds'),
           'image_files': set(r.get('input_image_files', [])), 'source': str(path / 'report.json'),
           'centres': {}}
    npz = path / 'model.npz'
    names = r.get('input_image_files', [])
    if npz.is_file() and names:
        z = np.load(npz)
        if 'cam_R' in z.files:                         # older runs did not store poses
            for idx, R, t in zip(z['camera_ids'], z['cam_R'], z['cam_t']):
                out['centres'][names[int(idx)]] = -R.T @ t
    return out


def read_colmap_txt(path):
    images_path, points_path = path / 'images.txt', path / 'points3D.txt'
    if not images_path.is_file() or not points_path.is_file():
        raise FileNotFoundError('COLMAP export requires images.txt and points3D.txt')
    # COLMAP writes TWO lines per image: the pose, then its 2D points (which is EMPTY for an
    # image with no observations). Blank lines are therefore data and must NOT be dropped.
    lines = [x for x in images_path.read_text().splitlines() if not x.startswith('#')]
    centres = {}
    for i in range(0, len(lines) - 1, 2):
        if not lines[i].strip():
            continue
        p = lines[i].split(maxsplit=9)
        R = quat_to_R(np.array(p[1:5], float))
        centres[p[9].strip()] = -R.T @ np.array(p[5:8], float)
    errs = []
    for line in points_path.read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 8:
                errs.append(float(parts[7]))
    return {'input_images': None, 'registered': len(centres), 'points': len(errs),
            'error': float(np.mean(errs)) if errs else None,
            'median_error': float(np.median(errs)) if errs else None, 'runtime': None,
            'image_files': set(centres), 'source': str(path), 'centres': centres}


def pose_error(run, colmap, only=None):
    """ATE (RMSE, median) of camera centres vs COLMAP after Sim(3); also % of trajectory extent.
    `only` restricts the evaluation to a given set of image names (same-image comparison)."""
    common = sorted(set(run['centres']) & set(colmap['centres']) & (set(only) if only is not None else set(run['centres'])))
    if len(common) < 4:
        return None
    A = np.array([run['centres'][k] for k in common])
    B = np.array([colmap['centres'][k] for k in common])
    rmse, med, scale, n = ate_after_sim3(A, B)
    extent = float(np.linalg.norm(B.max(0) - B.min(0)))
    return {'n': n, 'rmse': rmse, 'median': med, 'rmse_pct_extent': 100 * rmse / max(extent, 1e-12)}


def fmt(v, digits=3):
    if v is None:
        return 'N/A'
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    return f'{v:.{digits}f}' if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, default=Path('outputs/original'))
    p.add_argument('--adaptive', type=Path, default=Path('outputs/adaptive'))
    p.add_argument('--colmap-txt', type=Path, default=None)
    p.add_argument('--extra', action='append', default=[], metavar='LABEL=DIR',
                   help='additional Python run(s), e.g. ablation variants')
    p.add_argument('--csv', type=Path, default=Path('outputs/comparison.csv'))
    args = p.parse_args()
    runs = [('Original Python SfM', read_python_report(args.original)),
            ('Adaptive Python SfM', read_python_report(args.adaptive))]
    for e in args.extra:
        label, _, d = e.partition('=')
        runs.append((label, read_python_report(Path(d))))
    base = runs[0][1]
    for name, r in runs[1:]:
        if base['input_images'] != r['input_images'] or (
                base['image_files'] and r['image_files'] and base['image_files'] != r['image_files']):
            raise ValueError(f'{name}: must use EXACTLY the same photos as the Original run')
    colmap = None
    if args.colmap_txt:
        colmap = read_colmap_txt(args.colmap_txt)
        if base['image_files'] and not colmap['image_files'].issubset(base['image_files']):
            raise ValueError('COLMAP model includes photos absent from the Python dataset')
        colmap['input_images'] = base['input_images']
        runs.append(('COLMAP', colmap))

    # Fair accuracy comparison: evaluate every Python run on the images registered by ALL of them.
    all_py = [r for n_, r in runs if n_ != 'COLMAP' and r['centres']]
    shared = set.intersection(*[set(r['centres']) for r in all_py]) if (colmap and all_py) else set()
    rows = []
    for name, r in runs:
        pe = pose_error(r, colmap) if (colmap and name != 'COLMAP') else None
        pc = pose_error(r, colmap, shared) if (colmap and name != 'COLMAP' and len(shared) >= 4) else None
        rows.append({'method': name, 'input_images': r['input_images'],
                     'registered_images': r['registered'],
                     'registration_rate_percent': round(100 * r['registered'] / r['input_images'], 2),
                     '3d_points': r['points'], 'mean_reprojection_error_px': r['error'],
                     'median_reprojection_error_px': r['median_error'], 'runtime_s': r['runtime'],
                     'pose_common_images': pe['n'] if pe else None,
                     'ate_rmse_vs_colmap_sim3': pe['rmse'] if pe else (0.0 if name == 'COLMAP' else None),
                     'ate_median_vs_colmap_sim3': pe['median'] if pe else (0.0 if name == 'COLMAP' else None),
                     'ate_rmse_percent_of_extent': pe['rmse_pct_extent'] if pe else None,
                     'shared_images': len(shared) if pc else None,
                     'ate_rmse_on_shared_images': pc['rmse'] if pc else (0.0 if name == 'COLMAP' and shared else None),
                     'metric_note': 'Mean per-point error (COLMAP)' if name == 'COLMAP'
                     else 'Mean per-observation error (Python)'})

    print('\nSAME-IMAGE SPARSE SFM COMPARISON')
    print(f'{"Method":24s} {"Reg.":>9s} {"%":>6s} {"3D pts":>9s} {"mean px*":>9s} {"med px*":>8s} '
          f'{"time s":>7s} {"ATE n":>6s} {"ATE rmse":>9s} {"%extent":>8s} {"ATE@shared":>11s}')
    for r in rows:
        print(f'{r["method"]:24s} {r["registered_images"]:>4}/{r["input_images"]:<4} '
              f'{r["registration_rate_percent"]:>6.1f} {r["3d_points"]:>9,d} '
              f'{fmt(r["mean_reprojection_error_px"]):>9s} {fmt(r["median_reprojection_error_px"]):>8s} '
              f'{fmt(r["runtime_s"], 0):>7s} {fmt(r["pose_common_images"]):>6s} '
              f'{fmt(r["ate_rmse_vs_colmap_sim3"]):>9s} {fmt(r["ate_rmse_percent_of_extent"], 2):>8s} '
              f'{fmt(r["ate_rmse_on_shared_images"]):>11s}')
    print('\n* Reprojection errors are NOT strictly equivalent between COLMAP and Python; compare cautiously.')
    print('  ATE = camera-centre RMSE vs COLMAP after Sim(3) alignment, on images registered by both')
    print('  (units: COLMAP world units; %extent normalises by the COLMAP trajectory diagonal).')
    print(f'  ATE@shared = same metric restricted to the {len(shared)} images registered by EVERY Python run:')
    print('  the only like-for-like accuracy comparison (ATE over different image sets is not comparable).')
    print('  Runs without model.npz poses (older outputs) show N/A for ATE.')
    args.csv.parent.mkdir(exist_ok=True, parents=True)
    with args.csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'Saved: {args.csv}')


if __name__ == '__main__':
    main()
