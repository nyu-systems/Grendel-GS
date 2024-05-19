

# bash /global/homes/j/jy-nyu/gaussian-splatting/building_scripts/bui2k_1g_test.sh


# source ~/zhx.sh

# head_node_ip="nid003080"
# port=27709
# rid=104

expe_name="mat_ds2_sm_ae_bA_4g_test1"

# echo "connecting to head_node_ip: $head_node_ip, port: $port"

# export FI_MR_CACHE_MONITOR=disabled
python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
    -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_A_my_ds2 \
    --model_path /pscratch/sd/j/jy-nyu/mat_expes/$expe_name \
    --iteration 29997 \
    --sample_freq 5 \
    --num_train_cameras 100 \
    --num_test_cameras 100 \
    --distributed_load

python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
    -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_A_my_ds2 \
    --model_path /pscratch/sd/j/jy-nyu/mat_expes/$expe_name \
    --iteration 6997 \
    --sample_freq 5 \
    --num_train_cameras 100 \
    --num_test_cameras 100 \
    --distributed_load








