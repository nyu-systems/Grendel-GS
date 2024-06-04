

data_path="/pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred"
load_ply_path="/pscratch/sd/j/jy-nyu/neurips_expe/last_scripts/rub/rub_16g_e/point_cloud/iteration_199985"

python visualization_generator.py \
    -s $data_path \
    --load_ply_path $load_ply_path \
    --reference_idx 3 \
    --distributed_load \
    --model_path /pscratch/sd/j/jy-nyu/release_expes/rub_vis

