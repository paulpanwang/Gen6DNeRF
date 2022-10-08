import numpy as np

from dataset.database import BaseDatabase
# from utils.base_utils import pose_inverse, project_points

# 找到离相机最近的位置
def compute_nearest_camera_indices(database, que_ids, ref_ids=None):
    if ref_ids is None: ref_ids = que_ids
    ref_poses = [database.get_pose(ref_id) for ref_id in ref_ids]
    ref_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in ref_poses])
    que_poses = [database.get_pose(que_id) for que_id in que_ids]

    que_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in que_poses])

    dists = np.linalg.norm(ref_cam_pts[None, :, :] - que_cam_pts[:, None, :], 2, 2)
    dists_idx = np.argsort(dists, 1)
    return dists_idx

# reference image poses , 查询图像的pose, 工作数量， 是否要排除本身
def select_working_views(ref_poses, que_poses, work_num, exclude_self=False):
    ref_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in ref_poses])
    render_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in que_poses])
    dists = np.linalg.norm(ref_cam_pts[None, :, :] - render_cam_pts[:, None, :], 2, 2) # qn,rfn
    ids = np.argsort(dists)
    if exclude_self:
        ids = ids[:, 1:work_num+1]
    else:
        ids = ids[:, :work_num]
    return ids


def select_working_views_db(database: BaseDatabase, ref_ids, que_poses, work_num, exclude_self=False):
    ref_ids = database.get_img_ids() if ref_ids is None else ref_ids
    ref_poses = [database.get_pose(img_id) for img_id in ref_ids]

    ref_ids = np.asarray(ref_ids)
    ref_poses = np.asarray(ref_poses)
    indices = select_working_views(ref_poses, que_poses, work_num, exclude_self)
    return ref_ids[indices] # qn,wn

import cv2
from dataset.gen6d_database import get_object_vert
from utils.gen6d_base_utils import transformation_offset_2d
from utils.gen6d_base_utils import transformation_compose_2d
from utils.gen6d_base_utils import transformation_rotation_2d 
from utils.gen6d_database_utils import select_reference_img_ids_fps, normalize_reference_views
from dataset.gen6d_database import get_object_vert
from dataset.gen6d_database import  get_database_split, get_object_center

# Do we need zoom in procedure ？ 
def select_reference_views(database, split_type, ref_view_num=3 ):
    """_summary_
    select reference views(without zoom in)

    Args:
        database (_type_): linemod dataset
        split_type (_type_): Defaults to linemod_test
        ref_view_num (int, optional): _description_. Defaults to 8.

    Returns:
        dict : dataset_info
    """
    object_center = get_object_center(database)
    object_vert = get_object_vert(database)
    ref_ids_all, _ = get_database_split(database, split_type)

    # use fps to select reference images for detection and selection
    ref_ids = select_reference_img_ids_fps(database, ref_ids_all, ref_view_num)
    ref_imgs_list, ref_masks_list = list(),list()
    
    ref_poses = np.asarray([database.get_pose(ref_id) for ref_id in ref_ids]) # rfn,3,3
    ref_Ks = np.asarray([database.get_K(ref_id) for ref_id in ref_ids]) # rfn,3,3
    ref_depth_range = np.asarray([database.get_depth_range(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_depths = [database.get_depth(ref_id) for ref_id in ref_ids]
    ref_depths = np.asarray(ref_depths, dtype=np.float32)[:, None, :, :]


    for k in range(len(ref_ids)):
        ref_img = database.get_image(ref_ids[k])
        ref_mask = database.get_mask(ref_ids[k]).astype(np.float32)
        ref_masks_list.append(ref_mask)

        objmask = np.stack([ref_mask, ref_mask, ref_mask], axis=2)
        ref_masked_img = np.uint8(ref_img*(objmask))

        ref_imgs_list.append(ref_masked_img )

    ref_imgs = np.stack(ref_imgs_list, 0)
    ref_masks = np.stack(ref_masks_list, 0 )

    ref_ids = np.asarray([float(ref_id) for ref_id in ref_ids], dtype=np.float32)
    return {'imgs': ref_imgs, 'ref_ids':ref_ids , 'masks': ref_masks, 'depth': ref_depths , 'depth_range': ref_depth_range, \
             'Ks': ref_Ks, 'poses': ref_poses, 'center': object_center}

# Do we need zoom in procedure ？ 
def select_reference_views_zoomin( database, split_type, ref_resolution=128, ref_view_num=3 ):
    """_summary_
    select reference views(with zoom in)

    Args:
        database (_type_): _description_
        split_type (_type_): _description_
        ref_resolution (int, optional): _description_. Defaults to 128.
        ref_view_num (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    object_center = get_object_center(database)
    object_vert = get_object_vert(database)
    ref_ids_all, _ = get_database_split(database, split_type)
    # use fps to select reference images for detection and selection
    ref_ids = select_reference_img_ids_fps(database, ref_ids_all, ref_view_num)

    ref_imgs, ref_masks, ref_Ks, ref_poses, ref_Hs, ref_depth = \
        normalize_reference_views(database, ref_ids, ref_resolution, 0.05)
    
    # in-plane rotation for viewpoint selection
    rfn, h, w, _ = ref_imgs.shape
    ref_imgs_rots = []
    angles = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
    for angle in angles:
        M = transformation_offset_2d(-w/2,-h/2)
        M = transformation_compose_2d(M, transformation_rotation_2d(angle))
        M = transformation_compose_2d(M, transformation_offset_2d(w/2,h/2))
        H_ = np.identity(3).astype(np.float32)
        H_[:2,:3] = M
        ref_imgs_rot = []
        for rfi in range(rfn):
            H_new = H_ @ ref_Hs[rfi]
            ref_imgs_rot.append(cv2.warpPerspective(database.get_image(ref_ids[rfi]), H_new, (w,h), flags=cv2.INTER_LINEAR))
        ref_imgs_rots.append(np.stack(ref_imgs_rot, 0))
    ref_imgs_rots = np.stack(ref_imgs_rots, 0) # an,rfn,h,w,3
    ref_depth_range = np.asarray([database.get_depth_range(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_ids = np.asarray([float(ref_id) for ref_id in ref_ids], dtype=np.float32)
    return {'imgs': ref_imgs,'ref_ids':ref_ids , 'ref_imgs': ref_imgs_rots,'depth': ref_depth ,'masks': ref_masks, \
             'Ks': ref_Ks, 'poses': ref_poses, 'center': object_center, "depth_range":ref_depth_range }