
source ~/zhx.sh


# bash /global/homes/j/jy-nyu/gaussian-splatting/last_scripts/rub/rub_16g_render_vis.sh

# EXPE_NAMES=(rub_16g_g)

# CHECKPOINTS=(149985)

EXPE_NAMES=(rub_16g_f)
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
            --llffhold 83 \
            --render_one
     
    done
done


