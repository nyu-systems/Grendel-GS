
source ~/zhx.sh


# bash /global/homes/j/jy-nyu/gaussian-splatting/last_scripts/rub3/rub3_16g_render_metric.sh

EXPE_NAMES=(rub_16g_f)

CHECKPOINTS=(199985)

for name in ${EXPE_NAMES[@]}; do
    for iter in ${CHECKPOINTS[@]}; do
        echo "render and metric for ${name}"

        python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
            -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
            --model_path "/pscratch/sd/j/jy-nyu/last_scripts/rub3/${name}" \
            --iteration $iter \
            --distributed_load \
            --skip_train \
            --llffhold 83
        
        python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
            --mode test \
            --model_paths "/pscratch/sd/j/jy-nyu/last_scripts/rub3/${name}"

    done
done


