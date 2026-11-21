"""Print a fair, auditable SfM comparison from measured outputs.

Run COLMAP on the same input files and export its sparse model to TXT first:
  colmap model_converter --input_path colmap_sparse/0 --output_path colmap_txt --output_type TXT
"""
import argparse
import csv
import json
import math
from pathlib import Path


def read_python_report(path):
    r = json.loads((path / 'report.json').read_text())
    return {'input_images':r['images_total'], 'registered':r['images_registered'],
            'points':r['3d_points'], 'error':r['reprojection_error_mean_px'],
            'image_files':set(r.get('input_image_files',[])),
            'source':str(path / 'report.json')}


def read_colmap_txt(path):
    images_path = path / 'images.txt'
    points_path = path / 'points3D.txt'
    if not images_path.is_file() or not points_path.is_file():
        raise FileNotFoundError('COLMAP export requires images.txt and points3D.txt')
    lines = [x.strip() for x in images_path.read_text().splitlines() if x.strip() and not x.startswith('#')]
    if len(lines) % 2:
        raise ValueError('Invalid COLMAP images.txt, expected two lines per image')
    names = set()
    for i in range(0,len(lines),2):
        names.add(lines[i].split(maxsplit=9)[-1])
    point_errors = []
    for line in points_path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 8:
            point_errors.append(float(parts[7]))
    # COLMAP per-point ERROR is a pointwise mean reprojection error.
    # It is NOT identical to our observation-weighted error metric.
    return {'input_images':None, 'registered':len(names), 'points':len(point_errors),
            'error':sum(point_errors)/len(point_errors) if point_errors else None,
            'image_files':names, 'source':str(path)}


def fmt(v, digits=3):
    return f'{v:.{digits}f}' if isinstance(v, (int,float)) and not isinstance(v,bool) else ('N/A' if v is None else str(v))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original',type=Path,default=Path('outputs/original'))
    p.add_argument('--adaptive',type=Path,default=Path('outputs/adaptive'))
    p.add_argument('--colmap-txt',type=Path,default=None)
    p.add_argument('--csv',type=Path,default=Path('outputs/comparison.csv'))
    args=p.parse_args()
    original=read_python_report(args.original)
    adaptive=read_python_report(args.adaptive)
    if original['input_images'] != adaptive['input_images'] or (
        original['image_files'] and adaptive['image_files'] and
        original['image_files'] != adaptive['image_files']):
        raise ValueError('Original and adaptive must use EXACTLY the same photos')
    rows=[]
    datasets=[('Original Python SfM',original),('Adaptive Python SfM',adaptive)]
    if args.colmap_txt:
        c=read_colmap_txt(args.colmap_txt)
        if original['image_files'] and not c['image_files'].issubset(original['image_files']):
            raise ValueError('COLMAP model includes photos absent from the Python dataset')
        c['input_images']=original['input_images']
        datasets.append(('COLMAP',c))
    for name,r in datasets:
        rows.append({'method':name,'input_images':r['input_images'],
             'registered_images':r['registered'],
             'registration_rate_percent':round(100*r['registered']/r['input_images'],2),
             '3d_points':r['points'], 'mean_reprojection_error_px':r['error'],
             'metric_note': 'Mean per-point error (COLMAP)' if name=='COLMAP' else 'Mean per-observation error (Python)'})
    print('\nSAME-IMAGE SPARSE SFM COMPARISON')
    print(f'{"Method":23s} {"Registered":>11s} {"3D points":>12s} {"Error px*":>11s}')
    for r in rows:
        print(f'{r["method"]:23s} {r["registered_images"]:>3}/{r["input_images"]:<7} {r["3d_points"]:>12,d} {fmt(r["mean_reprojection_error_px"]):>11s}')
    print('\n* Errors are not strictly equivalent between COLMAP and Python; compare cautiously.')
    args.csv.parent.mkdir(exist_ok=True,parents=True)
    with args.csv.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)
    print(f'Saved: {args.csv}')

if __name__=='__main__':main()
