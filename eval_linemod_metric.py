import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from skimage.io import imsave
from copy import copy
from pathlib import Path
from gen6d_pose_api import init_render_model
from utils.imgs_info import build_render_imgs_info, imgs_info_to_torch, build_imgs_info
from dataset.gen6d_database import parse_database_name, get_database_split, get_ref_point_cloud, get_diameter, get_object_center
from utils.gen6d_base_utils import transformation_offset_2d,project_points, transformation_crop
from utils.gen6d_database_utils import compute_normalized_view_correlation
from utils.gen6d_draw_utils import draw_bbox, concat_images_list, draw_bbox_3d, pts_range_to_bbox_pts
from utils.gen6d_pose_utils import compute_metrics_each, scale_rotation_difference_from_cameras
from utils.gen6d_database_utils import normalize_reference_views
from utils.base_utils import to_cuda, to_cuda_half
from utils.base_utils import color_map_forward, pad_img_end
from utils.save_rgb_depth import save_render_depth, save_render_rgb
from utils.view_select import select_reference_views, select_working_views
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from utils.inerf_utils import rot_psi, rot_theta, rot_phi, trans_t 
from utils.imgs_info import build_inv_render_imgs_info

import  gc
scaler = GradScaler()

LINEMOD_OBJ_NAME = ("ape", "benchvise" , "cam", "can", \
                    "cat", "driller", "duck", "eggbox", \
                    "glue", "holepuncher", "iron", "lamp", \
                    "phone")

def main(args):
    est_name = "Gen6DNeRF"
    # load the render model
    renderer = init_render_model()
    lrate = args.lrate
    for obj_idx,  object_name in enumerate(LINEMOD_OBJ_NAME):
        object_name = "linemod/" + object_name
        if object_name.startswith('linemod'):
            ref_database_name = que_database_name = object_name
            que_split = 'linemod_test'
        else:
            raise NotImplementedError
        # Path(f'data/eval/poses/{object_name}').mkdir(exist_ok=True,parents=True)
        Path(f'data/eval/vis_final/{est_name}/{object_name}').mkdir(exist_ok=True,parents=True)
       
        # step1: build reference and query dataset
        ref_database = parse_database_name(ref_database_name)
        que_database = parse_database_name(que_database_name)
        _, que_ids = get_database_split(que_database, que_split)

        # step2: select the reference views, with/without zoomin
        zoom_in = True
        ref_view_num, ref_resolution = 6, 32
        ref_info = select_reference_views(ref_database, que_split, ref_view_num=ref_view_num)
        # ref_info = select_reference_views_zoomin(ref_database, que_split,ref_resolution=ref_resolution, ref_view_num=4 )

        ref_images = ref_info.get("imgs", None )
        ref_masks = ref_info.get("masks", None)
        ref_ks = ref_info.get("Ks", None)
        ref_ids = ref_info.get("ref_ids", None)
        ref_info["ref_resolution"] = ref_resolution
        save_reference_images(ref_images, ref_masks, est_name, object_name)

        # step3 : copy to devices
        ref_info["imgs"] = color_map_forward(ref_info["imgs"]).transpose([0, 3, 1, 2])
        ref_info = to_cuda(imgs_info_to_torch(ref_info))
        object_pts = get_ref_point_cloud(ref_database)
        object_center = get_object_center(ref_database)
        object_bbox_3d = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))
        pose_pr_list = []
        object_diameter = get_diameter(que_database)

        que_imgs, que_masks, que_Ks, que_poses, que_Hs, que_depths = \
            normalize_reference_views(que_database, que_ids, ref_resolution, 0.05)

        _, _,_, normalize_poses, _,_ = \
            normalize_reference_views(ref_database, que_ids, ref_info["ref_resolution"], 0.05)
        
        dists_idx = select_working_views(normalize_poses, que_poses, 1, exclude_self=False )

        from network.loss import INeRFLoss
        from utils.inerf_helpers import camera_transf
        from utils.pose_utils import calc_diff

        inerf_loss = INeRFLoss()
        cam_transf = camera_transf().cuda()
        optimizer = torch.optim.Adam(params=cam_transf.parameters(), \
                                 lr=lrate, betas=(0.9, 0.999))

        for idx, que_id in tqdm(enumerate(que_ids)):
            if idx%50 != 0: # choose one out of every 50 pictures
                continue
            # skip query image for speed up the evalution
            que_img =  que_imgs[idx]  if zoom_in else  que_database.get_image(que_id)
            mask = que_masks[idx] if zoom_in else que_database.get_mask(que_id)
            que_mask = np.stack([mask, mask, mask], axis=2)
            K = que_Ks[idx] if zoom_in else que_database.get_K(que_id)
            pose_gt = que_poses[idx] if zoom_in else que_database.get_pose(que_id)
            que_depth_range = que_database.get_depth_range(que_id)
            h, w = que_img.shape[:2]
            
            # print(que_imgs.shape, pose_gt, que_pose)
            # final_img = visualize_final_poses(img, K, object_bbox_3d, None, pose_gt)
            # imsave(f'data/eval/vis_final/{est_name}/{object_name}/{que_id}-bbox3d.jpg', final_img)
            # query pose , K , que_img.shape 

            work_idx = dists_idx[idx]
            obs_view_pose = normalize_poses[work_idx]
            obs_img_pose = np.concatenate((obs_view_pose.reshape(3,4), np.array([[0,0,0,1.]])), axis=0)

            target_image = np.uint8(que_img*(que_mask))
            target_image = torch.Tensor(target_image).cuda()

            best_res = {"best_loss": 100 , "best_pose":None}

            for iter in range(100):
                optimizer.zero_grad()
                pose = cam_transf(torch.Tensor(obs_img_pose).cuda() ) # 4*4
                # 3-views , muti-views 6dpose > single view
                # pixelNeRF (datasets ) multi-views
                que_imgs_info = build_inv_render_imgs_info(pose[:3,:4].reshape(1,3,4), \
                                                    K, \
                                                    que_img.shape[:2] , \
                                                    que_depth_range )
                                                    
                que_imgs_info = to_cuda(imgs_info_to_torch(que_imgs_info))
                que_shape = np.asarray( que_img.shape[:2] ,np.int64)
                data_info = {'que_imgs_info': que_imgs_info, 'eval': False}
                data_info['ref_imgs_info'] = ref_info
                data_info['que_shape'] = que_shape

                with torch.cuda.amp.autocast():
                    render_rgb, results = renderer.forword_6dpose(data_info, target_image)
                
                loss = inerf_loss(*results)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if loss.item()< best_res["best_loss"]:
                    best_res["best_loss"] = loss.item()
                    best_res["best_pose"] = pose.cpu().detach().numpy()

                new_lrate = lrate * (0.8 ** ((iter + 1) / 100))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate

                if (iter + 1) % 20 == 0:
                    print('iter :{}, Loss:{} '.format(iter, loss.item() ) )
                    with torch.no_grad():
                        pose_dummy = pose.cpu().detach().numpy()
                        calc_diff(pose_dummy, pose_gt)

            # save render result
            # output_dir = "data/render/"
            # save_render_rgb(output_dir, idx , renderout, h, w)
            # imsave(f'{output_dir}/gt_{idx}.jpg', np.uint8(que_img*(que_mask)) )
            # evaluation metrics
            pose_pr =best_res["best_pose"][:3,:4]
            results = compute_metrics_each(object_pts, object_diameter, pose_gt , pose_pr , K, scale=1, symmetric=args.symmetric)
            
            msg_pr = f'{object_name:10} {est_name:20} '
            for k, v in results.items(): msg_pr+=f'{k} {v:.4f} '
            with open('data/eval/performance.log','a') as f: f.write(msg_pr+"\n")
            

# save rederence images
def save_reference_images(ref_images, ref_masks, est_name, object_name):
    for idx, (ref_img, msk) in enumerate(zip(ref_images, ref_masks )):
        imsave(f'data/eval/vis_final/{est_name}/{object_name}/ref_img_{idx}.jpg', ref_img)
        imsave(f'data/eval/vis_final/{est_name}/{object_name}/ref_mask_{idx}.jpg', msk)
        ref_mask = np.stack([msk, msk, msk], axis=2)
        ref_masked_img = np.uint8(ref_img*(ref_mask))
        imsave(f'data/eval/vis_final/{est_name}/{object_name}/ref_wo_zoomin_masked_img_{idx}.jpg', ref_masked_img)


def get_gt_info(que_pose, que_K, render_poses, render_Ks, object_center):
    gt_corr = compute_normalized_view_correlation(que_pose[None], render_poses, object_center, False)
    gt_ref_idx = np.argmax(gt_corr[0]) # 找到参考图像的最大值
    gt_scale_r2q, gt_angle_r2q = scale_rotation_difference_from_cameras(
        render_poses[gt_ref_idx][None], que_pose[None], render_Ks[gt_ref_idx][None], que_K[None], object_center)
    gt_scale_r2q, gt_angle_r2q = gt_scale_r2q[0], gt_angle_r2q[0]
    gt_position = project_points(object_center[None], que_pose, que_K)[0][0]
    size = 128
    gt_bbox = np.concatenate([gt_position - size / 2 * gt_scale_r2q, np.full(2, size) * gt_scale_r2q])
    return gt_position, gt_scale_r2q, gt_angle_r2q, gt_ref_idx, gt_bbox, gt_corr[0]

def visualize_final_poses(img, K, object_bbox_3d, pose_pr=None, pose_gt=None):
    bbox_img = img
    if pose_gt is not None:
        bbox_pts_gt, _ = project_points(object_bbox_3d, pose_gt, K)
        bbox_img = draw_bbox_3d(bbox_img, bbox_pts_gt, (255,255,255))
    if pose_pr is not None:
        bbox_pts_pr, _ = project_points(object_bbox_3d, pose_pr, K)
        bbox_img = draw_bbox_3d(bbox_img, bbox_pts_pr, (0, 0, 255))
    return bbox_img


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_only', action='store_true', dest='eval_only', default=False)
    # add-0.1-s(symmetric=True) add-0.1(symmetric=False) 
    parser.add_argument('--symmetric', action='store_true', dest='symmetric', default=False)
    parser.add_argument('--split_type', type=str, default=None)
    parser.add_argument("--lrate", type=float, default=0.02,
                        help='Initial learning rate')

    args = parser.parse_args()
    main(args)


