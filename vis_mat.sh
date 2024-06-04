

data_path="/pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_modified"
load_ply_path="/pscratch/sd/j/jy-nyu/neurips_expe/last_scripts/mball2/mball2_16g_2/point_cloud/iteration_299985"

python visualization_generator.py \
    -s $data_path \
    --load_ply_path $load_ply_path \
    --reference_idx 220 \
    --distributed_load \
    --model_path /pscratch/sd/j/jy-nyu/release_expes/mat_vis

