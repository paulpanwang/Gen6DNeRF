import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from skimage.io import imsave
from copy import copy
from pathlib import Path
from gen6d_pose_api import init_render_model

from utils.imgs_info import build_render_imgs_info, imgs_info_to_torch
from dataset.gen6d_database import parse_database_name, get_database_split, get_ref_point_cloud, get_diameter, get_object_center
from utils.gen6d_base_utils import transformation_offset_2d,project_points, transformation_crop
from utils.gen6d_database_utils import compute_normalized_view_correlation
from utils.gen6d_draw_utils import draw_bbox, concat_images_list, draw_bbox_3d, pts_range_to_bbox_pts
from utils.gen6d_pose_utils import compute_metrics_impl, scale_rotation_difference_from_cameras
from utils.gen6d_database_utils import normalize_reference_views
from utils.base_utils import to_cuda, to_cuda_half
from utils.base_utils import color_map_forward, pad_img_end
from utils.save_rgb_depth import save_render_depth, save_render_rgb
from utils.view_select import select_reference_views, select_reference_views_zoomin

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
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
    # cost volume
    # ref_imgs_info, ref_cv_idx, ref_real_idx = build_src_imgs_info_select(
    # database, ref_ids, ref_ids_all, render_cfg["cost_volume_nn_num"], render_cfg["ref_pad_interval"])
    # src_imgs_info = ref_imgs_info.copy()
    # data['src_imgs_info'] = to_cuda(imgs_info_to_torch(src_imgs_info))
    # ref_imgs_info = imgs_info_slice(ref_imgs_info, ref_real_idx)
    # ref_imgs_info['nn_ids'] = ref_cv_idx

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
        ref_view_num, ref_resolution = 8, 32
        ref_info = select_reference_views(ref_database, que_split, ref_view_num=ref_view_num)
       
       
        ref_images = ref_info.get("imgs", None )
        ref_masks = ref_info.get("masks", None)
        ref_ks = ref_info.get("Ks", None)
        ref_ids = ref_info.get("ref_ids", None)
        save_reference_images(ref_images, ref_masks, est_name, object_name)

        # step3 : copy to devices
        ref_info["imgs"] = color_map_forward(ref_info["imgs"]).transpose([0, 3, 1, 2])
        ref_info = to_cuda(imgs_info_to_torch(ref_info))

        object_pts = get_ref_point_cloud(ref_database)
        object_center = get_object_center(ref_database)
        object_bbox_3d = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))

        pose_pr_list = []

        que_imgs, que_masks, que_Ks, que_poses, que_Hs, que_depths = \
            normalize_reference_views(que_database, que_ids, ref_resolution, 0.05)
        
        for idx, que_id in tqdm(enumerate(que_ids)):
            que_img =  que_imgs[idx]  if zoom_in else  que_database.get_image(que_id)
            mask = que_masks[idx] if zoom_in else que_database.get_mask(que_id)
            que_mask = np.stack([mask, mask, mask], axis=2)
            K = que_Ks[idx] if zoom_in else que_database.get_K(que_id)
            pose_gt = que_poses[idx] if zoom_in else que_database.get_pose(que_id)
            que_depth_range = que_database.get_depth_range(que_id)
            # print(que_imgs.shape, pose_gt, que_pose)
            # final_img = visualize_final_poses(img, K, object_bbox_3d, None, pose_gt)
            # imsave(f'data/eval/vis_final/{est_name}/{object_name}/{que_id}-bbox3d.jpg', final_img)
            # query pose , K , que_img.shape 
            que_imgs_info = build_render_imgs_info(pose_gt, \
                                                    K, \
                                                    que_img.shape[:2] , \
                                                    que_depth_range )

            print(ref_ks[0],  "\n" , K )
            que_imgs_info = to_cuda(imgs_info_to_torch(que_imgs_info))
            que_shape = np.asarray( que_img.shape[:2] ,np.int64)
            h, w = que_img.shape[:2]

            data_info = {'que_imgs_info': que_imgs_info, 'eval': False}
            data_info['ref_imgs_info'] = ref_info
            data_info['que_shape'] = que_shape
            with torch.no_grad():
                renderout = renderer(data_info)
            output_dir = "data/render/"
            save_render_rgb(output_dir, idx , renderout, h, w)
            imsave(f'{output_dir}/gt_{idx}.jpg', np.uint8(que_img*(que_mask)))

        # evaluation metrics
        pose_gt_list = [que_database.get_pose(que_id) for que_id in que_ids]
        que_Ks = [que_database.get_K(que_id) for que_id in que_ids]
        object_diameter = get_diameter(que_database)
        def get_eval_msg(pose_in_list,msg_in,scale=1.0):
            msg_in = copy(msg_in)
            results = compute_metrics_impl(object_pts, object_diameter, pose_gt_list, pose_in_list, que_Ks, scale, symmetric=args.symmetric)
            for k, v in results.items(): msg_in+=f'{k} {v:.4f} '
            return msg_in + '\n'

        msg_pr = f'{object_name:10} {est_name:20} '
        msg_pr = get_eval_msg(pose_pr_list, msg_pr)
        print(msg_pr)
        with open('data/eval/performance.log','a') as f: f.write(msg_pr)

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
    args = parser.parse_args()
    main(args)


