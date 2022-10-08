import argparse
import numpy as np

from colmap.process import process_example_dataset, clear_project
from dataset.database import parse_database_name
from utils.base_utils import pose_inverse
from utils.gen6d_draw_utils import output_points

parser = argparse.ArgumentParser()
parser.add_argument('--example_name', type=str, default='ape')
parser.add_argument('--same_camera', action='store_true', dest='same_camera', default=False)
parser.add_argument('--colmap_path', type=str, default='colmap')
flags = parser.parse_args()

"""
    custom datasets
    CUDA_VISIBLE_DEVICES=2  nohup python3 run_colmap.py &
"""

LINEMOD_OBJ_NAME = ("ape", "benchvise" , "cam", "can", \
                    "cat", "driller", "duck", "eggbox", \
                    "glue", "holepuncher", "iron", "lamp", \
                    "phone")

def visualize_camera_locations(example_name):
    database = parse_database_name(f'LINEMOD/{example_name}/JPEGImages')
    img_ids = database.get_img_ids()
    cam_pts = []
    for k, img_id in enumerate(img_ids):
        pose = database.get_pose(img_id)
        cam_pt = pose_inverse(pose)[:, 3]
        cam_pts.append(cam_pt)

    output_points(f'data/LINEMOD/{example_name}/cam_pts.txt', np.stack(cam_pts, 0))

for idx, example_name in enumerate(LINEMOD_OBJ_NAME):
    process_example_dataset(example_name,flags.same_camera,flags.colmap_path)
    clear_project(example_name)
