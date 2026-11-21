"""RUN THIS FILE. Example: python main.py --images images --out output"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from sfm_core import (load_images,camera_matrix,reconstruct,
                      reprojection_errors,refine_points_fixed_cameras)
from visualize import save_matches,save_ply,save_reconstruction,save_error_plot


def main():
    parser=argparse.ArgumentParser(description='Simple educational multi-view 3D reconstruction')
    parser.add_argument('--images',default='images',help='Folder of ordered, overlapping photographs')
    parser.add_argument('--out',default='output',help='Where results will be saved')
    parser.add_argument('--focal',type=float,default=None,help='Camera focal length in resized-image pixels')
    parser.add_argument('--max-size',type=int,default=1200,help='Resize longest image side to this many pixels')
    parser.add_argument('--features',type=int,default=3500,help='Maximum SIFT features per photo')
    parser.add_argument('--point-refine',action='store_true',help='Optional point-only optimization with fixed camera poses (NOT full bundle adjustment)')
    parser.add_argument('--refine-points',type=int,default=300,help='Max 3D points for point-only refinement')
    parser.add_argument('--min-parallax',type=float,default=1.0,help='Adaptive-only minimum triangulation angle in degrees')
    parser.add_argument('--mode',choices=['original','adaptive'],default='original')
    parser.add_argument('--window',type=int,default=10,help='Max image-index gap for adaptive candidate matching')
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    start_time=time.perf_counter()
    print('STEP 1: Loading photographs')
    images,paths=load_images(args.images,args.max_size)
    K=camera_matrix(images[0],args.focal)
    print(f'  {len(images)} images; approximate focal length {K[0,0]:.1f} px')
    print('STEP 2-5: SIFT, matching, camera geometry, incremental triangulation')
    model,pairs=reconstruct(images,K,max_features=args.features,
                            adaptive=(args.mode=='adaptive'),candidate_window=args.window,
                            min_parallax_deg=args.min_parallax)
    errors_before=reprojection_errors(model)
    if args.point_refine:
        print('STEP 6: Safe point-only refinement; camera poses stay fixed')
        updated=refine_points_fixed_cameras(model, max_points=args.refine_points)
        print('  Refined points:', updated)
    errors=reprojection_errors(model)
    print('STEP 7: Saving 3D point cloud and visual explanations')
    save_ply(model.points,model.colors,out/'points.ply')
    save_matches(images,model.keypoints,pairs,out)
    save_reconstruction(model,out)
    save_error_plot(errors,out)
    np.savez_compressed(out/'model.npz',points=model.points,colors=model.colors,K=model.K,
                        camera_ids=np.array(model.registered))
    report={'refinement':'point_only_fixed_cameras' if args.point_refine else 'none',
            'refinement_points_limit':args.refine_points if args.point_refine else 0,
            'min_parallax_deg':args.min_parallax if args.mode=='adaptive' else 0.0,
            'focal_px':float(K[0,0]),'max_size':args.max_size,
            'runtime_seconds':round(time.perf_counter()-start_time,3),
            'features_limit':args.features,'mode':args.mode,'input_image_files':[p.name for p in paths],
            'seed_images':model.diagnostics['seed_images'],
            'candidate_window':model.diagnostics['candidate_window'],
            'seed_candidates':model.diagnostics['seed_candidates'],
            'registration_log':model.diagnostics['registration_log'],
            'images_total':len(paths),'images_registered':len(model.registered),
            'registered_image_numbers':[i+1 for i in model.registered],
            'unregistered_image_numbers':model.diagnostics['unregistered_images'],
            '3d_points':len(model.points),'reprojection_error_mean_px':float(np.mean(errors)) if len(errors) else None,
            'reprojection_error_median_px':float(np.median(errors)) if len(errors) else None,
            'reprojection_error_before_ba_mean_px':float(np.mean(errors_before)) if len(errors_before) else None,
            'notes':'Sparse points only; scale arbitrary; camera intrinsics approximate unless --focal supplied.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print(f'\nSUCCESS. See {out}/points.ply, reconstruction.png, feature_matches.jpg, reprojection_error.png')


if __name__=='__main__':
    main()
