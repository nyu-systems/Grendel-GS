
expe_name="mat_ds2_sm_ae_bA_4g_test1_re_c1"

# echo "connecting to head_node_ip: $head_node_ip, port: $port"
iterations=(15497 15997 16497 16997 17497 17997 18497 18997 19497 19997)
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