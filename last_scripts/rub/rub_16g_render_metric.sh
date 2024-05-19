
source ~/zhx.sh


# bash /global/homes/j/jy-nyu/gaussian-splatting/last_scripts/rub/rub_16g_render_metric.sh

# EXPE_NAMES=(rub_16g_8 rub_16g_b)
# EXPE_NAMES=(rub_16g_e rub_16g_f)
# EXPE_NAMES=(rub_16g_g)
# EXPE_NAMES=(rub_16g_c)
EXPE_NAMES=(rub_16g_h rub_16g_9)

CHECKPOINTS=(199985)

for name in ${EXPE_NAMES[@]}; do
    for iter in ${CHECKPOINTS[@]}; do
        echo "render and metric for ${name}"

        python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
            -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
            --model_path "/pscratch/sd/j/jy-nyu/last_scripts/rub/${name}" \
            --iteration $iter \
            --distributed_load \
            --skip_train \
            --llffhold 83
        
        python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
            --mode test \
            --model_paths "/pscratch/sd/j/jy-nyu/last_scripts/rub/${name}"

    done
done


