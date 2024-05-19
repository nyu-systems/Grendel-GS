

# bash /global/homes/j/jy-nyu/gaussian-splatting/building_scripts/bui2k_1g_test.sh


# source ~/zhx.sh

# head_node_ip="nid003080"
# port=27709
# rid=104

expe_name="mat_ds2_sm_ae_bA_4g_test1_re"

# echo "connecting to head_node_ip: $head_node_ip, port: $port"
iterations=(6997 14997 19997 29997 39997 49997)
# iterations=(99997)
for iteration in "${iterations[@]}"
do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_A_my_ds2 \
        --model_path /pscratch/sd/j/jy-nyu/mat_expes/$expe_name \
        --iteration $iteration \
        --sample_freq 5 \
        --num_train_cameras 30 \
        --num_test_cameras 30 \
        --distributed_load
done


