


expe_name="mat_ds2_bA_4g_mp_3"

iterations=(200000 100000 50000 7000)

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




