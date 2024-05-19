
# bash /global/homes/j/jy-nyu/gaussian-splatting/last_scripts/tandb/4g_4b.sh

source ~/zhx.sh

SCENE=(train truck drjohnson playroom)
DATA_PATH=(/pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/truck /pscratch/sd/j/jy-nyu/datasets/tandt_db/db/drjohnson /pscratch/sd/j/jy-nyu/datasets/tandt_db/db/playroom)
BSZ=(4)

for i in {0..3}; do
    scene=${SCENE[$i]}
    data_path=${DATA_PATH[$i]}
    for bsz in ${BSZ[@]}; do
        torchrun --standalone --nnodes=1 --nproc-per-node=4 /global/homes/j/jy-nyu/gaussian-splatting/train.py \
            -s $data_path \
            --llffhold 8 \
            --iterations 30000 \
            --log_interval 250 \
            --log_folder /pscratch/sd/j/jy-nyu/tmp/4g4b/$scene \
            --model_path /pscratch/sd/j/jy-nyu/tmp/4g4b/$scene \
            --redistribute_gaussians_mode "1" \
            --gaussians_distribution \
            --bsz 4 \
            --densify_until_iter 15000 \
            --densify_grad_threshold 0.0002 \
            --percent_dense 0.01 \
            --opacity_reset_interval 3000 \
            --zhx_python_time \
            --zhx_time \
            --end2end_time \
            --test_iterations 7000 15000 30000 \
            --save_iterations 7000 15000 30000 \
            --checkpoint_iterations 30000 \
            --eval \
            --lr_scale_mode "sqrt" \
            --use_final_system2 \
            --preload_dataset_gpu \
            --no_heuristics_update
    done
done
