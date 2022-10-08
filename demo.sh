
# the original code is very slow ....


# Demo1: Render with llff scenes (set up a custom dataset via colmap ) 
CUDA_VISIBLE_DEVICES=3 python3 run_render.py --cfg configs/gen/neuray_gen_cost_volume.yaml \
                 --database llff_colmap/fern/high  \
                 --pose_type eval

# Demo2: Render with linemod scenes (set up a custom dataset via colmap ) 
python3 run_render.py --database example/custom/480 --cfg configs/gen/neuray_gen_depth.yaml --pose_type circle  
ffmpeg -y -framerate 30 -r 30   \
       -i data/render/example/custom/480/neuray_gen_depth-pretrain-circle/%d-nr_fine.jpg   \
       -vcodec libx264 -pix_fmt yuv420p  linemod.mp4

# Demo3: 
python3 run_linemod_render.py


# Demo4:
 python3 eval_linemod_metric.py














