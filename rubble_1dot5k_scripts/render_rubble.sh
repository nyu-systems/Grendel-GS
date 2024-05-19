# python render.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --model_path experiments/rubble_mp4_30k \
#     --iteration 7000 \
#     --sample_freq 5 \
#     --skip_test \
#     --num_train_cameras 50


# python render.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --model_path experiments/rubble_mp4_30k \
#     --iteration 15000 \
#     --sample_freq 5 \
#     --skip_test \
#     --num_train_cameras 50


# python render.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --model_path experiments/rubble_mp4_30k \
#     --iteration 20000 \
#     --sample_freq 5 \
#     --skip_test \
#     --num_train_cameras 50


# python render.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --model_path experiments/rubble_mp4_30k \
#     --iteration 30000 \
#     --sample_freq 5 \
#     --skip_test \
#     --num_train_cameras 50


# python render.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --model_path experiments/rubble_mp4_30k \
#     --iteration 40000 \
#     --sample_freq 5 \
#     --skip_test \
#     --num_train_cameras 50

# python render.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --model_path /pscratch/sd/j/jy-nyu/expe_backup/expe04222335/rubble_mp4_30k \
#     --iteration 50000 \
#     --sample_freq 5 \
#     --skip_test \
#     --num_train_cameras 50

# write above into a single script

# torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py \
#     -s /scratch/hz3496/3dgs_data/tandt_db/tandt/${SCENE} \
#     --iterations 30000 \
#     --log_interval 50 \
#     --log_folder experiments_adam/${SCENE}_bsz_${BSZ}_scalemode_${SCALE_MODE} \
#     --model_path experiments_adam/${SCENE}_bsz_${BSZ}_scalemode_${SCALE_MODE} \
#     --render_distribution_adjust_mode "5" \
#     --memory_distribution_mode "1" \
#     --redistribute_gaussians_mode "1" \
#     --loss_distribution_mode "avoid_pixel_all2all" \
#     --test_iterations 1000 7000 10000 12500 15000 17500 20000 22500 25000 27500 30000 \
#     --dp_size $DP_SIZE \
#     --bsz $BSZ \
#     --lr_scale_mode $SCALE_MODE # can be "linear", "sqrt" or "accumu

# EXPE_NAME="/pscratch/sd/j/jy-nyu/expe_backup/expe04222335/rubble_mp4_30k"
# CHECKPOINTS=(7000 15000 20000 30000 40000 50000)

# for iter in ${CHECKPOINTS[@]}; do
#     python render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --sample_freq 5 \
#         --skip_test \
#         --num_train_cameras 50
# done


# EXPE_NAME="/pscratch/sd/j/jy-nyu/expe_backup/expe04222335/rubble_dp4_30k_sbc"
# CHECKPOINTS=(6997 14997 19997 29997)

# for iter in ${CHECKPOINTS[@]}; do
#     python render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --sample_freq 5 \
#         --skip_test \
#         --num_train_cameras 50
# done

# EXPE_NAME="/pscratch/sd/j/jy-nyu/expe_backup/expe04222335/rubble_mp4_4_90k_sbc"
# CHECKPOINTS=(7000 15000 20000 30000 50000 80000 90000)

# for iter in ${CHECKPOINTS[@]}; do
#     python render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --sample_freq 5 \
#         --skip_test \
#         --num_train_cameras 50
# done

EXPE_NAME="/pscratch/sd/j/jy-nyu/running_expes/rubble_1d5k_mp_2"
# CHECKPOINTS=(7000 15000 20000 30000 50000 80000)
CHECKPOINTS=(100000 50000 80000 7000 30000)

for iter in ${CHECKPOINTS[@]}; do
    python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
        -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-1.5k \
        --model_path ${EXPE_NAME} \
        --iteration $iter \
        --sample_freq 5 \
        --skip_test \
        --num_train_cameras 50
done