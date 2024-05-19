

# bash /global/homes/j/jy-nyu/gaussian-splatting/building_scripts/bui2k_1g_test.sh


# source ~/zhx.sh

# head_node_ip="nid003080"
# port=27709
# rid=104

expe_name="matr_small_4g_test1"

python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
    -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/street/pose/block_small_my \
    --model_path /pscratch/sd/j/jy-nyu/running_expes/matr_small_4g_test1 \
    --iteration 49997 \
    --sample_freq 5 \
    --num_train_cameras 50 \
    --num_test_cameras 50 \
    --distributed_load


