

# bash /global/homes/j/jy-nyu/gaussian-splatting/building_scripts/bui2k_1g_test.sh


# source ~/zhx.sh

# head_node_ip="nid003080"
# port=27709
# rid=104

expe_name="mat_sm_ae_ball_4g_nodsfy"

python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
    -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_my \
    --model_path /pscratch/sd/j/jy-nyu/mat_expes/$expe_name \
    --iteration 79997 \
    --sample_freq 5 \
    --num_train_cameras 100 \
    --num_test_cameras 100 \
    --distributed_load




