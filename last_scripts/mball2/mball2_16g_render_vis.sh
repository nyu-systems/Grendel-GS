
source ~/zhx.sh


# bash /global/homes/j/jy-nyu/gaussian-splatting/last_scripts/mball2/mball2_16g_render_vis.sh

EXPE_NAMES=(mball2_16g_2)

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
            --llffhold 10 \
            --render_one \
            --num_test_cameras 230     

    done
done


