
source ~/zhx.sh


# bash /global/homes/j/jy-nyu/gaussian-splatting/last_scripts/mball2/mball2_16g_render_metric.sh

# EXPE_NAMES=(mball2_16g_2 mball2_16g_3 mball2_16g_4 mball2_16g_5 mball2_16g_6 mball2_16g_7)
EXPE_NAMES=(mball2_16g_8 mball2_16g_9)

CHECKPOINTS=(299985)

for name in ${EXPE_NAMES[@]}; do
    for iter in ${CHECKPOINTS[@]}; do
        echo "render and metric for ${name}"

        python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
            -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_my2 \
            --model_path "/pscratch/sd/j/jy-nyu/last_scripts/mball2/${name}" \
            --iteration $iter \
            --distributed_load \
            --skip_train \
            --sample_freq 10 \
            --llffhold 10
        
        python /global/homes/j/jy-nyu/gaussian-splatting/metrics.py \
            --mode test \
            --model_paths "/pscratch/sd/j/jy-nyu/last_scripts/mball2/${name}"

    done
done


