
EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_1"
CHECKPOINTS=(199993)

for iter in ${CHECKPOINTS[@]}; do
    python render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
        --model_path ${EXPE_NAME} \
        --iteration $iter \
        --sample_freq 5 \
        --num_train_cameras 30 \
        --num_test_cameras 30 \
        --distributed_load \
        --llffhold 83
done