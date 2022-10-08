import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
from skimage.io import imsave
from utils.eval_time import get_time_millisecond
from utils.base_utils import  color_map_backward

@get_time_millisecond
def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_

def save_gt(output_dir, image):
    out_file_name = f'{output_dir}/ground-true.png'
    imsave(out_file_name, image)
    print("out_file_name:", out_file_name)


def save_render_rgb(output_dir, qi,  render_info, h, w):
    def output_image(suffix):
        if f'pixel_colors_{suffix}' in render_info:
            render_image = color_map_backward(render_info[f'pixel_colors_{suffix}'].cpu().numpy().reshape([h, w, 3]))
            out_file_name = f'{output_dir}/{qi}-{suffix}.jpg'
            imsave(out_file_name , render_image)
            print("out_file_name:", out_file_name)
    output_image('nr_fine') 

def save_render_depth(output_dir, qi, render_info, h, w, depth_range):
    suffix = 'fine'
    cmap = cv2.COLORMAP_JET
    print(render_info.keys())
    depth = render_info['depth_mean_fine'].cpu().numpy().reshape([h, w])
    near, far = depth_range
    depth = np.clip(depth, a_min=near, a_max=far)
    depth = (1/depth - 1/near)/(1/far - 1/near)
    depth = color_map_backward(depth)
    depth = (255 * depth).astype(np.uint8)
    depth_ = Image.fromarray(cv2.applyColorMap(depth, cmap))
    out_file_name = f'{output_dir}/{qi}-{suffix}-depth.png'
    imsave(out_file_name, depth_)
    print("out_file_name:", out_file_name)

