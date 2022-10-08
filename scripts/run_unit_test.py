import argparse
import os
from pathlib import Path
from utils.eval_time import get_time_second
import numpy as np
import torch
from skimage.io import imsave
from tqdm import tqdm
from colorama import Fore, Back, Style
from dataset.database import parse_database_name, get_database_split, ExampleDatabase
from dataset.train_dataset import build_src_imgs_info_select
from network.renderer import name2network
from utils.base_utils import load_cfg, to_cuda, color_map_backward, make_dir
from utils.imgs_info import build_imgs_info, build_render_imgs_info, imgs_info_to_torch, imgs_info_slice
from utils.render_poses import get_render_poses
from utils.view_select import select_working_views_db
from Gen6DNeRF.scripts.run_render import prepare_render_info, save_renderings
from utils.save_rgb_depth import save_render_depth, save_render_rgb, save_gt

# If you want to store gif video of op/home/panpanwang/landing/transpose/3rdparty/feather/feather/quant/model_surgeon/experimentaltimization process, set OVERLAY = True
default_render_cfg = {
    'min_wn': 3, # working view number
    'ref_pad_interval': 16, # input image size should be multiple of 16
    'use_src_imgs': False, # use source images to construct cost volume or not
    'cost_volume_nn_num': 3, # number of source views used in cost volume
    'use_depth': True, # use colmap depth in rendering or not
}

@get_time_second
def prepare_render_info(database):
    """AI is creating summary for prepare_render_info

    Args:
        database ([type]): [description]

    Returns:
        que_pose ([type]): the Ground Ture of target image

    """
    image_ids = database.get_img_ids()
    assert len(image_ids) >= 2
    que_id = image_ids[0]
    que_pose = np.asarray( database.get_pose(que_id) ,np.float32 )
    que_K = np.asarray( database.get_K(que_id) ,np.float32)
    que_shape = np.asarray( database.get_image(que_id).shape[:2] ,np.int64)
    que_depth_ranges = np.asarray( database.get_depth_range(que_id),np.float32  )
    # exclude query id
    que_image = database.get_image(que_id)
    # image_ids.remove( que_id )

    ref_ids = image_ids
    dataset_info = {
        "que_image": que_image,
        "que_pose": que_pose, 
        "que_K": que_K,
        "que_shape": que_shape, 
        "que_depth_ranges": que_depth_ranges,
        "ref_ids": ref_ids
    }
    return dataset_info

@get_time_second
def gen_6dof_pose( cfg_fn, database_name, ray_num ):
    """_summary_

    Args:
        cfg_fn (_type_): linemod dataset config file
        database_name (_type_): linemod dataset path
    """
    print(f"load config file from {cfg_fn}")
    cfg = load_cfg(cfg_fn)
    cfg["ray_batch_num"] = ray_num
    renderer = name2network[cfg['network']](cfg)
    ckpt = torch.load(f'data/model/{cfg["name"]}/model_best.pth')
    renderer.load_state_dict(ckpt['network_state_dict'], strict=False )
    renderer.cuda()
    renderer.eval()
    database = parse_database_name(database_name)
    render_cfg = dict()
    render_cfg = {**default_render_cfg, **render_cfg}

    dataset_info = prepare_render_info(database)
    # print( dataset_info )
    # dataset_info = {
    #     "que_pose": que_pose, 
    #     "que_K": que_K,
    #     "que_shape": que_shape, 
    #     "que_depth_ranges": que_depth_ranges,
    #     "ref_ids": ref_ids
    # }
    h, w = dataset_info.get("que_shape")
    # ref_ids_list = select_working_views_db(database, ref_ids, que_pose, render_cfg['min_wn'])
    output_dir = f'data/render/{database.database_name}/gen_6dof_pose'
    make_dir(output_dir)

    # revise version: use multi-view images as reference  
    que_imgs_info = build_render_imgs_info(dataset_info["que_pose"], \
                                           dataset_info["que_K"], \
                                           dataset_info["que_shape"] , \
                                           dataset_info["que_depth_ranges"] )

    que_imgs_info = to_cuda(imgs_info_to_torch(que_imgs_info))
    data = {'que_imgs_info': que_imgs_info, 'eval': True}
    ref_imgs_info = build_imgs_info(database, dataset_info["ref_ids"] , render_cfg["ref_pad_interval"])
    ref_imgs_info = to_cuda(imgs_info_to_torch(ref_imgs_info))
    data['ref_imgs_info'] = ref_imgs_info
    with torch.no_grad():
        render_info = renderer(data)

    # save the render result
    save_render_rgb(output_dir, 0 , render_info, h, w)
    save_gt(output_dir,  dataset_info["que_image"])
    
def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/gen/neuray_gen_6dof_pose.yaml', help='config path of the renderer')
    parser.add_argument('--database_name', type=str, default='linemod/custom/512', help='<dataset_name>/<scene_name>/<scene_setting>')
    parser.add_argument('--ray_num', type=int, default=8192, help='number of rays in one rendering batch')
    parser.add_argument('--depth', action='store_true', dest='depth', default=False)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_argument()
    assert args.cfg, args.database_name    
    gen_6dof_pose( cfg_fn = args.cfg, \
                   database_name = args.database_name, \
                   ray_num = args.ray_num )






