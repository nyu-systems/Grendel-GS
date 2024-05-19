

# EXPE_NAME="/pscratch/sd/j/jy-nyu/running_expes/rubble_2k_mp_1"
# CHECKPOINTS=(200000 150000 100000 50000 80000 7000 30000)

# for iter in ${CHECKPOINTS[@]}; do
#     python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-2k \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --sample_freq 5 \
#         --skip_test \
#         --num_train_cameras 50
# done

# EXPE_NAME="/pscratch/sd/j/jy-nyu/running_expes/rubble_2k_mp_2"
# CHECKPOINTS=(150000 100000 50000 80000 7000 30000)

# for iter in ${CHECKPOINTS[@]}; do
#     python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-2k \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --sample_freq 5 \
#         --skip_test \
#         --num_train_cameras 50 \
#         --distributed_load
# done

# EXPE_NAME="/pscratch/sd/j/jy-nyu/running_expes/rubble_2k_mp_3"
# CHECKPOINTS=(50000 100000)

# for iter in ${CHECKPOINTS[@]}; do
#     python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-2k \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --sample_freq 5 \
#         --skip_test \
#         --num_train_cameras 50
# done

# EXPE_NAME="/pscratch/sd/j/jy-nyu/running_expes/rubble_2k_mp_4"
# CHECKPOINTS=(50000 100000)

# for iter in ${CHECKPOINTS[@]}; do
#     python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-2k \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --sample_freq 5 \
#         --skip_test \
#         --num_train_cameras 50
# done

# EXPE_NAME="/pscratch/sd/j/jy-nyu/running_expes/rubble_2k_mp_6"
# CHECKPOINTS=(200000)
# #  150000 50000

# for iter in ${CHECKPOINTS[@]}; do
#     python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-2k \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --sample_freq 5 \
#         --skip_test \
#         --num_train_cameras 50 \
#         --distributed_load        
# done

EXPE_NAME="/pscratch/sd/j/jy-nyu/expe_backup/expe04252400/rubble_2k_mp_6_eval"
CHECKPOINTS=(200000 150000 100000)

for iter in ${CHECKPOINTS[@]}; do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-2k \
        --model_path ${EXPE_NAME} \
        --iteration $iter \
        --sample_freq 5 \
        --num_train_cameras 50 \
        --distributed_load \
        --eval
done