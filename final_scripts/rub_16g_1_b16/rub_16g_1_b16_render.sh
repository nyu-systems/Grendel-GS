
EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_1_b16"
CHECKPOINTS=(199985)

for iter in ${CHECKPOINTS[@]}; do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
        --model_path ${EXPE_NAME} \
        --iteration $iter \
        --distributed_load \
        --skip_train \
        --llffhold 83
done