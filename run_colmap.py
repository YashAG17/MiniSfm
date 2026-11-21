"""Run COLMAP CLI on EXACTLY the images used for both Python methods."""
import argparse
from pathlib import Path
import shutil
import subprocess


def call(args):
    print('RUNNING:', ' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)), check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('outputs/colmap'))
    parser.add_argument('--colmap',default='colmap',help='COLMAP executable path (or COLMAP.bat on Windows)')
    parser.add_argument('--matcher',choices=['sequential','exhaustive'],default='sequential')
    args=parser.parse_args()
    photos=sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'))
    if len(photos)<3: raise RuntimeError('Need at least three image files')
    exe=shutil.which(args.colmap) or (str(Path(args.colmap).resolve()) if Path(args.colmap).exists() else None)
    if not exe: raise RuntimeError('COLMAP executable not found. Pass --colmap with its path.')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    db=out/'database.db'; sparse=out/'sparse';sparse.mkdir(exist_ok=True)
    if db.exists():
        raise FileExistsError(f'{db} already exists; choose a fresh --out to avoid mixing experiment runs')
    images=args.images.resolve()
    call([exe,'feature_extractor','--database_path',db,'--image_path',images,'--ImageReader.single_camera','1'])
    call([exe,args.matcher+'_matcher','--database_path',db])
    call([exe,'mapper','--database_path',db,'--image_path',images,'--output_path',sparse])
    models=sorted([p for p in sparse.iterdir() if p.is_dir() and ((p/'images.bin').is_file() or (p/'images.txt').is_file())])
    if not models: raise RuntimeError('COLMAP generated no sparse model; see its console log')
    print('Available COLMAP sparse models:',[m.name for m in models])
    for model in models:
        txt=out / ('txt_' + model.name);txt.mkdir(exist_ok=True)
        call([exe,'model_converter','--input_path',model,'--output_path',txt,'--output_type','TXT'])
    print('Run compare_three.py with --colmap-txt',out / ('txt_' + models[0].name))

if __name__=='__main__': main()
