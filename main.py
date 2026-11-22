"""RUN THIS FILE. Example: python main.py --images images --out output"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import cv2
from sfm_core import (load_images,camera_matrix,reconstruct,
                      reprojection_errors,refine_points_fixed_cameras)
from adaptive_sfm import Options,reconstruct_adaptive,check_track_consistency
from visualize import (save_matches,save_ply,save_reconstruction,save_error_plot,
                       save_clean_ply,save_cameras_ply)


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
    parser.add_argument('--mode',choices=['original','adaptive_v1','adaptive'],default='original',
                        help='original = baseline (unchanged); adaptive_v1 = first attempt (kept for the ablation story); '
                             'adaptive = geometry-aware pipeline in adaptive_sfm.py')
    parser.add_argument('--window',type=int,default=None,
                        help='Index window for candidate pairs (default: 3 for original/adaptive, 10 for adaptive_v1)')
    parser.add_argument('--ablate',default=None,
                        help="adaptive only: comma list of components to ENABLE from retrieval,seed,order,vote,baseline,localba ('none' = all off)")
    parser.add_argument('--topk',type=int,default=8,help='adaptive: retrieved candidate images per image')
    parser.add_argument('--ba-window',type=int,default=8,help='adaptive: cameras freed in each local BA')
    parser.add_argument('--ba-every',type=int,default=5,help='adaptive: run local BA after every k registrations')
    parser.add_argument('--seed',type=int,default=0,help='OpenCV RNG seed (RANSAC) for reproducible runs')
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    cv2.setRNGSeed(args.seed)
    start_time=time.perf_counter()
    print('STEP 1: Loading photographs')
    images,paths=load_images(args.images,args.max_size)
    K=camera_matrix(images[0],args.focal)
    print(f'  {len(images)} images; approximate focal length {K[0,0]:.1f} px')
    print('STEP 2-5: SIFT, matching, camera geometry, incremental triangulation')
    if args.mode=='adaptive':
        opt=Options.from_ablate(args.ablate)
        opt.window=args.window if args.window is not None else 3
        opt.topk=args.topk;opt.ba_window=args.ba_window;opt.ba_every=args.ba_every
        opt.min_parallax_deg=args.min_parallax
        model,pairs=reconstruct_adaptive(images,K,max_features=args.features,opt=opt)
    else:
        model,pairs=reconstruct(images,K,max_features=args.features,
                                adaptive=(args.mode=='adaptive_v1'),
                                candidate_window=args.window if args.window is not None else 10,
                                min_parallax_deg=args.min_parallax)
    errors_before=reprojection_errors(model)
    if args.point_refine:
        print('STEP 6: Safe point-only refinement; camera poses stay fixed')
        updated=refine_points_fixed_cameras(model, max_points=args.refine_points)
        print('  Refined points:', updated)
    errors=reprojection_errors(model)
    print('STEP 7: Saving 3D point cloud and visual explanations')
    save_ply(model.points,model.colors,out/'points.ply')          # RAW geometry, never filtered
    kept=save_clean_ply(model.points,model.colors,out/'points_clean.ply')   # viewer-friendly copy
    save_cameras_ply(model,out/'cameras.ply')
    save_matches(images,model.keypoints,pairs,out)
    save_reconstruction(model,out)
    save_error_plot(errors,out)
    cam_ids=np.array(model.registered)
    np.savez_compressed(out/'model.npz',points=model.points,colors=model.colors,K=model.K,
                        camera_ids=cam_ids,
                        cam_R=np.array([model.cameras[i][0] for i in model.registered]),
                        cam_t=np.array([model.cameras[i][1] for i in model.registered]))
    report={'refinement':'point_only_fixed_cameras' if args.point_refine else 'none',
            'refinement_points_limit':args.refine_points if args.point_refine else 0,
            'min_parallax_deg':args.min_parallax if args.mode!='original' else 0.0,
            'ablate':args.ablate,'rng_seed':args.seed,
            'components':model.diagnostics.get('components'),
            'track_stats':model.diagnostics.get('track_stats'),
            'track_consistency':check_track_consistency(model),
            'local_ba_log':model.diagnostics.get('local_ba_log'),
            'points_clean_ply_kept':kept,
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
