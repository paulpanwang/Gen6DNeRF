import argparse
from operator import ne
import os
from pathlib import Path
from utils.eval_time import get_time_second
import numpy as np
import torch
import skimage
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
from run_render import prepare_render_info, save_renderings
from utils.save_rgb_depth import save_render_depth, save_render_rgb, save_gt
from utils.inerf_helpers import camera_transf
from network.loss import INeRFLoss
from utils.inerf_utils import rot_psi, rot_theta, rot_phi, trans_t 
from utils.imgs_info import build_inv_render_imgs_info

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import  gc

scaler = GradScaler()

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
def gen_6dof_pose( cfg_fn, database_name, ray_num, args ):
    """_summary_

    Args:
        cfg_fn (_type_): linemod dataset config file
        database_name (_type_): linemod dataset path
    """
    lrate = args.lrate
    noise, sigma, amount = args.noise, args.sigma, args.amount
    print(f"load config file from {cfg_fn}")
    cfg = load_cfg(cfg_fn)
    cfg["ray_batch_num"] = ray_num
    renderer = name2network[cfg['network']](cfg)
    ckpt = torch.load(f'data/model/{cfg["name"]}/model_best.pth')
    renderer.load_state_dict(ckpt['network_state_dict'], strict=False )
    renderer.cuda()
    renderer.eval()

    for name, param in renderer.named_parameters():
        param.requires_grad = False


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
    target_pose, target_image = dataset_info["que_pose"], dataset_info["que_image"]

    phi_ref = np.arctan2(target_pose[1,0], target_pose[0,0])*180/np.pi
    theta_ref = np.arctan2(-target_pose[2, 0], np.sqrt(target_pose[2, 1]**2 + target_pose[2, 2]**2))*180/np.pi
    psi_ref = np.arctan2(target_pose[2, 1], target_pose[2, 2])*180/np.pi
    translation_ref = np.sqrt(target_pose[0,3]**2 + target_pose[1,3]**2 + target_pose[2,3]**2)

    if noise == 'gaussian':
        target_image = skimage.util.random_noise(target_image, mode='gaussian', var=sigma**2)
    elif noise == 's_and_p':
        target_image = skimage.util.random_noise(target_image, mode='s&p', amount=amount)
    elif noise == 'pepper':
        target_image = skimage.util.random_noise(target_image, mode='pepper', amount=amount)
    elif noise == 'salt':
        target_image = skimage.util.random_noise(target_image, mode='salt', amount=amount)
    elif noise == 'poisson':
        target_image = skimage.util.random_noise(target_image, mode='poisson')
    else:
        target_image = target_image

    # prepare image
    delta_phi, delta_theta, delta_psi, delta_t = args.delta_phi, args.delta_theta, args.delta_psi, args.delta_t
    print(  delta_phi, delta_theta, delta_psi, delta_t )
    obs_img_pose = np.concatenate((target_pose, np.array([[0,0,0,1.]])), axis=0)
    start_pose = rot_phi(delta_phi/180.*np.pi) @ rot_theta(delta_theta/180.*np.pi) @ \
                 rot_psi(delta_psi/180.*np.pi) @ trans_t(delta_t) @ obs_img_pose

    start_pose =  torch.Tensor(start_pose).cuda()
    target_image = torch.Tensor(target_image).cuda()

    # 学习的是相对的变换关系
    cam_transf = camera_transf().cuda()
    inerf_loss = INeRFLoss()
    optimizer = torch.optim.Adam(params=cam_transf.parameters(), \
                                 lr=lrate, betas=(0.9, 0.999))

    # llff / nerf /     
    # 3-views 
    # muti-views 6dpose > single view
    # pixelNeRF (datasets ) multi-views  
    ref_imgs_info = build_imgs_info(database, dataset_info["ref_ids"] , render_cfg["ref_pad_interval"])
    ref_imgs_info = to_cuda(imgs_info_to_torch(ref_imgs_info))

    for iter in range(299):
        optimizer.zero_grad()
        pose = cam_transf(start_pose)
        que_imgs_info = build_inv_render_imgs_info( pose[:3,:4].reshape(1,3,4) , \
                                            dataset_info["que_K"], \
                                            dataset_info["que_shape"] , \
                                            dataset_info["que_depth_ranges"] )
                                            
        que_imgs_info = to_cuda(imgs_info_to_torch(que_imgs_info))
        data_info = {'que_imgs_info': que_imgs_info, 'eval': True}
        data_info['ref_imgs_info'] = ref_imgs_info
        data_info['que_shape'] =  dataset_info["que_shape"]

        with torch.cuda.amp.autocast():
            render_rgb, results = renderer.forword_6dpose(data_info, target_image)
        
        loss = inerf_loss(*results)
    
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        new_lrate = lrate * (0.08 ** ((iter + 1) / 100))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if (iter + 1) % 20 == 0 or iter == 0:
            print('iter : ', iter)
            print('Loss: ', loss)
            with torch.no_grad():
                pose_dummy = pose.cpu().detach().numpy()
                # calculate angles and translation of the optimized pose
                phi = np.arctan2(pose_dummy[1, 0], pose_dummy[0, 0]) * 180 / np.pi
                theta = np.arctan2(-pose_dummy[2, 0], np.sqrt(pose_dummy[2, 1] ** 2 + pose_dummy[2, 2] ** 2)) * 180 / np.pi
                psi = np.arctan2(pose_dummy[2, 1], pose_dummy[2, 2]) * 180 / np.pi
                translation = np.sqrt(pose_dummy[0,3]**2 + pose_dummy[1,3]**2 + pose_dummy[2,3]**2)
                #translation = pose_dummy[2, 3]
                # calculate error between optimized and observed pose
                phi_error = abs(phi_ref - phi) if abs(phi_ref - phi)<300 else abs(abs(phi_ref - phi)-360)
                theta_error = abs(theta_ref - theta) if abs(theta_ref - theta)<300 else abs(abs(theta_ref - theta)-360)
                psi_error = abs(psi_ref - psi) if abs(psi_ref - psi)<300 else abs(abs(psi_ref - psi)-360)
                rot_error = phi_error + theta_error + psi_error
                translation_error = abs(translation_ref - translation)
                print('Rotation error: ', rot_error)
                print('Translation error: ', translation_error)
                print('-----------------------------------')
        gc.collect()
        torch.cuda.empty_cache()
    
def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/gen/neuray_gen_6dof_pose.yaml', help='config path of the renderer')
    parser.add_argument('--database_name', type=str, default='linemod/custom/64', help='<dataset_name>/<scene_name>/<scene_setting>')
    parser.add_argument('--ray_num', type=int, default=256, help='number of rays in one rendering batch')
    parser.add_argument('--depth', action='store_true', dest='depth', default=False)
    parser.add_argument("--noise", type=str, default='gauss',
                        help='options: gauss / salt / pepper / sp / poisson')
    parser.add_argument("--sigma", type=float, default=0.01,
                        help='var = sigma^2 of applied noise (variance = std)')
    parser.add_argument("--amount", type=float, default=0.05,
                        help='proportion of image pixels to replace with noise (used in ‘salt’, ‘pepper’, and ‘s&p)')
    
    parser.add_argument("--lrate", type=float, default=0.01,
                        help='Initial learning rate')
    
    # parameters to define initial pose
    parser.add_argument("--delta_psi", type=float, default=3,
                        help='Rotate camera around x axis')
    parser.add_argument("--delta_phi", type=float, default=2,
                        help='Rotate camera around z axis')
    parser.add_argument("--delta_theta", type=float, default=3,
                        help='Rotate camera around y axis')
    parser.add_argument("--delta_t", type=float, default=1.0,
                        help='translation of camera (negative = zoom in)')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_argument()
    assert args.cfg, args.database_name    
    gen_6dof_pose( cfg_fn = args.cfg, \
                   database_name = args.database_name, \
                   ray_num = args.ray_num, args = args )






