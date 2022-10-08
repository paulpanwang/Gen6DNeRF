
### the original code is very slow ....

###
```shell

models: 

|-- data
    |--model
        |-- neuray_gen_cost_volume: https://drive.google.com/file/d/16EVg1ql86FlHwP4JaBhR79dT-gBhDUK2/view
        |-- neuray_gen_depth: https://drive.google.com/file/d/1ZIwCnnD8avga8f-p5-Z8m1C_lRgdDaLq/view
    |-- dtu_test
    |-- llff_colmap:https://drive.google.com/file/d/1CPfdekwKM6zt_skb-IYOruN3Dyzwfhdm/view
    |-- nerf_synthetic: https://drive.google.com/file/d/1EEwDBQY2jUNJpptxOZPG9nkHz96HChed/view
    |-- LINEMOD: https://connecthkuhk-my.sharepoint.com/personal/yuanly_connect_hku_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fyuanly%5Fconnect%5Fhku%5Fhk%2FDocuments%2FGen6D%2Flinemod%2Etar%2Egz&parent=%2Fpersonal%2Fyuanly%5Fconnect%5Fhku%5Fhk%2FDocuments%2FGen6D

```
### Demo1: Render with llff scenes (set up a custom dataset via colmap ) 
```shell
CUDA_VISIBLE_DEVICES=3 python3 run_render.py --cfg configs/gen/neuray_gen_cost_volume.yaml \
                 --database llff_colmap/fern/high  \
                 --pose_type eval
```
### Demo2: Render with linemod scenes (set up a custom dataset via colmap ) 
```shell
python3 run_render.py --database example/custom/480 --cfg configs/gen/neuray_gen_depth.yaml --pose_type circle  

ffmpeg -y -framerate 30 -r 30   \
       -i data/render/example/custom/480/neuray_gen_depth-pretrain-circle/%d-nr_fine.jpg   \
       -vcodec libx264 -pix_fmt yuv420p  linemod.mp4

```
#### Demo3: 

'''
python3 run_linemod_render.py 
'''


'''
python3 scripts/run_unit_test.py
'''





#### Demo4: 

'''
python3 eval_linemod_metric.py

'''


#### codes_explanations

```shell
├── README.md
├── asset.py
├── colmap     # colmap scripts
│   ├── build.py
│   ├── build_windows_app.py
│   ├── bundler_to_ply.py
│   ├── clang_format_code.py
│   ├── crawl_camera_specs.py
│   ├── database.py
│   ├── export_inlier_matches.py
│   ├── export_inlier_pairs.py
│   ├── export_to_bundler.py
│   ├── export_to_visualsfm.py
│   ├── flickr_downloader.py
│   ├── merge_ply_files.py
│   ├── nvm_to_ply.py
│   ├── plyfile.py
│   ├── read_write_dense.py
│   ├── read_write_fused_vis.py
│   ├── read_write_model.py
│   ├── test_read_write_dense.py
│   ├── test_read_write_fused_vis.py
│   ├── test_read_write_model.py
│   └── visualize_model.py
├── colmap_build.md
├── configs  # config files
│   ├── dtu_test_scans.txt
│   ├── ft  # finetune config file
│   │   ├── neuray_ft_chair_pretrain.yaml
│   │   ├── neuray_ft_drums_pretrain.yaml
│   │   ├── neuray_ft_ficus_pretrain.yaml
│   │   ├── neuray_ft_hotdog_pretrain.yaml
│   │   ├── neuray_ft_lego_pretrain.yaml
│   │   ├── neuray_ft_linemod_pretrain.yaml
│   │   ├── neuray_ft_materials_pretrain.yaml
│   │   ├── neuray_ft_mic_pretrain.yaml
│   │   └── neuray_ft_ship_pretrain.yaml
│   ├── gen # Generalizable config file
│   │   ├── neuray_gen_6dof_pose.yaml
│   │   ├── neuray_gen_cost_volume.yaml
│   │   └── neuray_gen_depth.yaml
│   ├── inter_trajectory
│   │   └── blended_mvs
│   │       ├── building.txt
│   │       ├── dragon.txt
│   │       ├── iron_dog.txt
│   │       ├── laid_man.txt
│   │       ├── mermaid.txt
│   │       └── santa.txt
│   └── train
│       ├── ft
│       │   ├── example
│       │   │   └── neuray_ft_desktop_pretrain.yaml
│       │   ├── neuray_ft_cv_lego.yaml
│       │   ├── neuray_ft_depth_birds.yaml
│       │   ├── neuray_ft_depth_fern.yaml
│       │   └── neuray_ft_depth_lego.yaml
│       └── gen
│           ├── neuray_gen_cost_volume_train.yaml
│           └── neuray_gen_depth_train.yaml
├── custom_rendering.md
├── data # data and model
│   ├── example
│   ├── linemod
│   ├── llff_colmap
│   ├── model
│   └── render #render result
├── dataset # dataset loader
│   ├── database.py
│   ├── name2dataset.py
│   └── train_dataset.py
├── demo.sh
├── eval.py
├── format_code.sh
├── network  #define model 
│   ├── aggregate_net.py
│   ├── dist_decoder.py
│   ├── ibrnet.py
│   ├── init_net.py
│   ├── loss.py
│   ├── metrics.py
│   ├── mvsnet # for cost volume
│   │   ├── modules.py
│   │   ├── mvsnet.py
│   │   └── mvsnet_pl.ckpt
│   ├── ops.py
│   ├── render_ops.py
│   ├── renderer.py    #define main model 
│   ├── sph_solver.py
│   ├── vis_dino_encoder.py
│   ├── vis_encoder.py
│   └── vision_transformer.py
├── neuray_rendering.md
├── requirements.txt
├── run_colmap.py
├── run_gen6d_pose.py
├── run_inerf.py
├── run_render.py
├── run_training.py
├── run_unit_test.py
├── train
│   ├── lr_common_manager.py
│   ├── train_tools.py
│   ├── train_valid.py
│   └── trainer.py
└── utils
    ├── base_utils.py
    ├── dataset_utils.py
    ├── draw_utils.py
    ├── eval_time.py
    ├── imgs_info.py
    ├── inerf_helpers.py
    ├── inerf_utils.py
    ├── llff_utils.py
    ├── real_estate_utils.py
    ├── render_poses.py
    ├── save_rgb_depth.py
    ├── space_dataset_utils.py
    └── view_select.py # FPS ...

```