


expe_name="definal_bA_4g_7"

iterations=(299997)

for iteration in "${iterations[@]}"
do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_A_my \
        --model_path /pscratch/sd/j/jy-nyu/definal_expes/$expe_name \
        --iteration $iteration \
        --sample_freq 5 \
        --num_train_cameras 30 \
        --num_test_cameras 30 \
        --distributed_load
done




