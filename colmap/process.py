import os
from pathlib import Path
import subprocess
import numpy as np
from skimage.io import imread

from colmap.database import COLMAPDatabase
from colmap.read_write_model import CAMERA_MODEL_NAMES

# 把图像分为若干份
def split_list(listTemp, n):
    for i in range(0, len(listTemp), n):
        if len(listTemp[i:i + n])>=3:
            yield listTemp[i:i + n]
        else:
            yield listTemp[i-10:i+n]

def process_example_dataset(example_name, same_camera=False, colmap_path='$HOME/code/colmap/build/src/exe/colmap'):
    project_dir=f'data/LINEMOD/{example_name}'
    print("project_dir:", project_dir)
    database_name = f'{project_dir}/database.db'
    image_path_name = f'{project_dir}/images'

    if not os.path.exists(image_path_name):
        os.mkdir( image_path_name )

    # add images
    img_dir = Path(image_path_name)

    img_dir_full = Path(f'{project_dir}/JPEGImages')
    full_image_fns = []
    for pattern in ['*.jpg','*.png','*.PNG','*.JPG']:
        full_image_fns+=[fn for fn in img_dir_full.glob(pattern)]
    full_image_fns = sorted(full_image_fns)
    tmp_list = split_list(full_image_fns, 8)
    for item in tmp_list:
        # if database exists, delete it 
        if os.path.exists(database_name):
            os.remove(database_name)
        
        clear_project( example_name )

        # create database for all images
        db = COLMAPDatabase.connect(database_name)
        db.create_tables()

        # copy image from JPEGImage to imgs  
        for image_path in item:
            os.system("cp {} {}".format(image_path, image_path_name))

        img_fns = []
        for pattern in ['*.jpg','*.png','*.PNG','*.JPG']:
            img_fns+=[fn for fn in img_dir.glob(pattern)]
        img_fns = sorted(img_fns)

        global_cam_id = None
        for k, img_fn in enumerate(img_fns):
            img = imread(img_fn)
            h, w, _ = img.shape
            focal = np.sqrt(h**2+w**2) # guess a focal here
            if same_camera:
                if k==0: global_cam_id = db.add_camera(CAMERA_MODEL_NAMES['SIMPLE_PINHOLE'].model_id,
                                                    float(w), float(h), np.array([focal,w/2, h/2], np.float64), prior_focal_length=True)
                db.add_image(img_fn.name, global_cam_id)
            else:
                cam_id = db.add_camera(CAMERA_MODEL_NAMES['SIMPLE_PINHOLE'].model_id,
                                    float(w), float(h), np.array([focal,w/2,h/2],np.float64),prior_focal_length=True)
                db.add_image(img_fn.name, cam_id)

        db.commit()
        db.close()

        # feature extraction
        cmd=[colmap_path,'feature_extractor',
            '--database_path',f'{project_dir}/database.db',
            '--image_path',f'{project_dir}/images',
            ]
        print(' '.join(cmd))
        subprocess.run(cmd,check=True)

        # feature matching
        cmd=[colmap_path,'exhaustive_matcher',
            '--database_path',f'{project_dir}/database.db']
        print(' '.join(cmd))
        subprocess.run(cmd,check=True)

        # SfM
        Path(f'{project_dir}/sparse').mkdir(exist_ok=True,parents=True)
        cmd=[colmap_path,'mapper',
            '--database_path',f'{project_dir}/database.db',
            '--image_path',f'{project_dir}/images',
            '--output_path',f'{project_dir}/sparse',
         ]

        print(' '.join(cmd))
        subprocess.run(cmd,check=True)

        # dense reconstruction
        Path(f'{project_dir}/dense').mkdir(exist_ok=True,parents=True)
        cmd=[colmap_path,'image_undistorter',
            '--image_path',f'{project_dir}/images',
            '--input_path',f'{project_dir}/sparse/0',
            '--output_path',f'{project_dir}/dense',
          ]
        print(' '.join(cmd))
        subprocess.run(cmd,check=True)

        cmd=[colmap_path,'patch_match_stereo',
            '--workspace_path',f'{project_dir}/dense']
        print(' '.join(cmd))
        subprocess.run(cmd,check=True)

        cmd = [str(colmap_path), 'stereo_fusion',
            '--workspace_path', f'{project_dir}/dense',
            '--workspace_format', 'COLMAP',
            '--input_type', 'geometric',
            '--output_path', f'{project_dir}/points.ply',
          ]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

        for image_path in img_fns:
            os.system("rm {} ".format(image_path))


def clear_project(example_name):
    output_dir = f'data/LINEMOD/{example_name}'
    os.system(f'rm {output_dir}/database.db')
    os.system(f'rm {output_dir}/dense/images -r')
    os.system(f'rm {output_dir}/dense/sparse -r')
    os.system(f'rm {output_dir}/dense/*.sh')
    os.system(f'rm {output_dir}/dense/stereo/depth_maps/*photometric.bin')
    os.system(f'rm {output_dir}/dense/stereo/normal_maps -r')