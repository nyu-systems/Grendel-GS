
# echo "render and metric for rub_16g_1"

EXPE_NAMES=(rub_16g_8 rub_16g_9 rub_16g_a rub_16g_b rub_16g_c rub_16g_d rub_16g_e rub_16g_f)
CHECKPOINTS=(199985)

for name in ${EXPE_NAMES[@]}; do
    for iter in ${CHECKPOINTS[@]}; do
        echo "render and metric for ${name}"

        python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
            -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
            --model_path "/pscratch/sd/j/jy-nyu/final_expes/${name}" \
            --iteration $iter \
            --distributed_load \
            --skip_train \
            --llffhold 83
        
        python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
            --mode test \
            --model_paths "/pscratch/sd/j/jy-nyu/final_expes/${name}"

    done
done

sleep 2





