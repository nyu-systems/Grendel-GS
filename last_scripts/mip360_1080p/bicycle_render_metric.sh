
source ~/zhx.sh

# ALL_EXPES=(bicycle_1 bicycle_2 bicycle_3 bicycle_4 bicycle_5)
ALL_EXPES=(bicycle_6 bicycle_7 bicycle_8)

for expe in ${ALL_EXPES[@]}; do
    EXPE_NAME="/pscratch/sd/j/jy-nyu/last_scripts/bicycle/${expe}"
    CHECKPOINTS=(29997 49997)

    for iter in ${CHECKPOINTS[@]}; do
        python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
            -s /pscratch/sd/j/jy-nyu/datasets/360_v2/bicycle1080p \
            --model_path ${EXPE_NAME} \
            --iteration $iter \
            --skip_train \
            --llffhold 8
    done

    python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
        --mode test \
        --model_paths ${EXPE_NAME}

    sleep 2
done



