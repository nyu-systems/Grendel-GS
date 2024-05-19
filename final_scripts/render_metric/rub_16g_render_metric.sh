
echo "render and metric for rub_16g_1"

EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_1"
CHECKPOINTS=(199993)

for iter in ${CHECKPOINTS[@]}; do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
        --model_path ${EXPE_NAME} \
        --iteration $iter \
        --distributed_load \
        --skip_train \
        --llffhold 83
done

python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}

sleep 2





echo "render and metric for rub_16g_2"

EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_2"
CHECKPOINTS=(199993)

for iter in ${CHECKPOINTS[@]}; do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
        --model_path ${EXPE_NAME} \
        --iteration $iter \
        --distributed_load \
        --skip_train \
        --llffhold 83
done

python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}

sleep 2




echo "render and metric for rub_16g_3"

EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_3"
CHECKPOINTS=(199993)

for iter in ${CHECKPOINTS[@]}; do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
        --model_path ${EXPE_NAME} \
        --iteration $iter \
        --distributed_load \
        --skip_train \
        --llffhold 83
done

python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}

sleep 2




echo "render and metric for rub_16g_4"

EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_4"
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

python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}

sleep 2





echo "render and metric for rub_16g_5"

EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_5"
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

python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}

sleep 2




echo "render and metric for rub_16g_6"

EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_6"
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

python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}

sleep 2






echo "render and metric for rub_16g_7"

EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_7"
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

python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}

sleep 2






echo "render and metric for rub_16g_7_c2"

EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_c2"
CHECKPOINTS=(239985)

for iter in ${CHECKPOINTS[@]}; do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
        --model_path ${EXPE_NAME} \
        --iteration $iter \
        --distributed_load \
        --skip_train \
        --llffhold 83
done

python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}