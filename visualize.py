"""Save lecture-friendly pictures and the reconstructed colored sparse point cloud."""
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_matches(images, keypoints, pairs, folder):
    """Visualize feature correspondence lines for the best neighbour pair."""
    folder = Path(folder)
    if not pairs:
        return
    (i,j), matches = max(pairs.items(), key=lambda item: len(item[1]))
    if not len(matches):
        return
    a,b = images[i].copy(), images[j].copy()
    h=max(a.shape[0],b.shape[0]); wa=a.shape[1]
    canvas=np.zeros((h,wa+b.shape[1],3),dtype=np.uint8)
    canvas[:a.shape[0],:wa]=a
    canvas[:b.shape[0],wa:]=b
    rng=np.random.default_rng(42)
    for k in rng.choice(len(matches),size=min(75,len(matches)),replace=False):
        p,q=matches[k]
        x,y=np.round(keypoints[i][p]).astype(int)
        u,v=np.round(keypoints[j][q]).astype(int)
        color=tuple(int(c) for c in rng.integers(50,240,3))
        cv2.line(canvas,(x,y),(u+wa,v),color,1,cv2.LINE_AA)
        cv2.circle(canvas,(x,y),3,color,-1)
        cv2.circle(canvas,(u+wa,v),3,color,-1)
    cv2.imwrite(str(folder/'feature_matches.jpg'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))


def save_ply(points, colors, filename):
    """Export an RGB point cloud viewable in MeshLab or CloudCompare."""
    with open(filename,'w',encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for pt,c in zip(points,colors):
            f.write(f'{pt[0]:.8f} {pt[1]:.8f} {pt[2]:.8f} {int(c[0])} {int(c[1])} {int(c[2])}\n')


def save_reconstruction(model, folder):
    """Plot robustly clipped 3D points, camera locations, and color information."""
    folder=Path(folder)
    pts=model.points
    if len(pts)<3: return
    # Plot only well-behaved central percentile: extreme triangulations spoil figure scaling.
    mid=np.median(pts,axis=0); radius=np.linalg.norm(pts-mid,axis=1)
    good=np.isfinite(pts).all(axis=1)&(radius<=np.percentile(radius,96))
    shown=pts[good]
    fig=plt.figure(figsize=(11,8))
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(shown[:,0],shown[:,1],shown[:,2],c=model.colors[good]/255.,s=2,alpha=.85)
    camera_centers=[]
    for idx,(R,t) in sorted(model.cameras.items()):
        center=-R.T@t
        camera_centers.append(center)
        ax.text(*center,str(idx+1),fontsize=8)
    centers=np.array(camera_centers)
    ax.scatter(centers[:,0],centers[:,1],centers[:,2],c='red',s=35,marker='^',label='Camera centers')
    ax.set_title('Sparse 3D reconstruction (Structure-from-Motion)')
    ax.set_xlabel('X (arbitrary scale)');ax.set_ylabel('Y');ax.set_zlabel('Z')
    ax.legend(loc='upper right')
    # Equal-ish axes scale based on clipped point cloud, not extreme outliers.
    center=np.median(shown,axis=0)
    spread=max(np.ptp(shown,axis=0).max()/2.,1e-3)
    ax.set_xlim(center[0]-spread,center[0]+spread)
    ax.set_ylim(center[1]-spread,center[1]+spread)
    ax.set_zlim(center[2]-spread,center[2]+spread)
    fig.tight_layout()
    fig.savefig(folder/'reconstruction.png',dpi=160)
    plt.close(fig)


def save_error_plot(errors, folder):
    """Make the reprojection-error chart useful in the professor demo."""
    fig,ax=plt.subplots(figsize=(8,4.5))
    if len(errors): ax.hist(np.clip(errors,0,10),bins=35,color='#4466aa')
    ax.set_xlabel('Reprojection error (pixels, capped at 10 for display)')
    ax.set_ylabel('Number of observations')
    ax.set_title('Reprojection error distribution')
    fig.tight_layout();fig.savefig(Path(folder)/'reprojection_error.png',dpi=150)
    plt.close(fig)
