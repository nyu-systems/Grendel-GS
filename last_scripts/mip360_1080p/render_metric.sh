
# echo "render and metric for rub_16g_1"

# EXPE_NAME="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_1"
# CHECKPOINTS=(199993)

# for iter in ${CHECKPOINTS[@]}; do
#     python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#         -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#         --model_path ${EXPE_NAME} \
#         --iteration $iter \
#         --distributed_load \
#         --skip_train \
#         --llffhold 83
# done

# python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
#     --mode test \
#     --model_paths ${EXPE_NAME}

# sleep 2

# folder=(4g_4b)
# SCENE=(counter kitchen room stump bicycle garden bonsai flowers treehill)

SCENE=(kitchen)


for scene in ${SCENE[@]}; do
    EXPE_NAME="/pscratch/sd/j/jy-nyu/last_scripts/mip360_1080p/4g_4b/e_${scene}"
    CHECKPOINTS=(29997)

    for iter in ${CHECKPOINTS[@]}; do
        python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
            -s /pscratch/sd/j/jy-nyu/datasets/360_v2/${scene}1080p \
            --model_path ${EXPE_NAME} \
            --iteration $iter \
            --skip_train \
            --llffhold 8
    done

    # python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
    #     --mode test \
    #     --model_paths ${EXPE_NAME}

    sleep 2
done



