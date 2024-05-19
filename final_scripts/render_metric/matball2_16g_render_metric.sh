
# echo "render and metric for mball2_16g_1"

# EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_1"
# CHECKPOINTS=(299985)

# for iter in ${CHECKPOINTS[@]}; do
#     python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_my2 \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --distributed_load \
#         --skip_train \
#         --sample_freq 10 \
#         --llffhold 10
# done

# python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
#     --mode test \
#     --model_paths ${EXPE_NAME}

# sleep 2





# echo "render and metric for mball2_16g_2"

# EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_2"
# CHECKPOINTS=(299985)

# for iter in ${CHECKPOINTS[@]}; do
#     python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_my2 \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --distributed_load \
#         --sample_freq 10 \
#         --skip_train \
#         --llffhold 83
# done

# python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
#     --mode test \
#     --model_paths ${EXPE_NAME}

# sleep 2




# echo "render and metric for mball2_16g_3"

# EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_3"
# CHECKPOINTS=(299985)

# for iter in ${CHECKPOINTS[@]}; do
#     python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_my2 \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --distributed_load \
#         --sample_freq 10 \
#         --skip_train \
#         --llffhold 83
# done

# python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
#     --mode test \
#     --model_paths ${EXPE_NAME}

# sleep 2




# echo "render and metric for mball2_16g_4"

# EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_4"
# CHECKPOINTS=(299985)

# for iter in ${CHECKPOINTS[@]}; do
#     python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_my2 \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --distributed_load \
#         --sample_freq 10 \
#         --skip_train \
#         --llffhold 83
# done

# python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
#     --mode test \
#     --model_paths ${EXPE_NAME}

# sleep 2


# echo "render and metric for mball2_16g_5"

# EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_5"
# CHECKPOINTS=(299985)

# for iter in ${CHECKPOINTS[@]}; do
#     python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_my2 \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --distributed_load \
#         --sample_freq 10 \
#         --skip_train \
#         --llffhold 83
# done

# python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
#     --mode test \
#     --model_paths ${EXPE_NAME}

# sleep 2




echo "render and metric for mball2_16g_6_re"

EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_6_re"
CHECKPOINTS=(299985)

for iter in ${CHECKPOINTS[@]}; do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_my2 \
        --model_path ${EXPE_NAME} \
        --iteration $iter \
        --distributed_load \
        --sample_freq 10 \
        --skip_train \
        --llffhold 83
done

python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}

sleep 2






echo "render and metric for mball2_16g_7_re"

EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_7_re"
CHECKPOINTS=(299985)

for iter in ${CHECKPOINTS[@]}; do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_my2 \
        --model_path ${EXPE_NAME} \
        --iteration $iter \
        --distributed_load \
        --sample_freq 10 \
        --skip_train \
        --llffhold 83
done

python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}

sleep 2






echo "render and metric for mball2_16g_8"

EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_8"
CHECKPOINTS=(299985)

for iter in ${CHECKPOINTS[@]}; do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_my2 \
        --model_path ${EXPE_NAME} \
        --iteration $iter \
        --distributed_load \
        --sample_freq 10 \
        --skip_train \
        --llffhold 83
done

python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
    --mode test \
    --model_paths ${EXPE_NAME}