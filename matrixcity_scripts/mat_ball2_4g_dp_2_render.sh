
expe_name="mat_ball2_4g_dp_2"

iterations=(199997 19997 99997)

for iteration in "${iterations[@]}"
do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_my2 \
        --model_path /pscratch/sd/j/jy-nyu/mat_expes/$expe_name \
        --iteration $iteration \
        --sample_freq 5 \
        --num_train_cameras 30 \
        --num_test_cameras 30 \
        --distributed_load
done