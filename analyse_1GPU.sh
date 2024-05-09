SCENE=$1
START_ITER=$(echo "$2" | grep -E '^[0-9]+$' || echo "0")
CONT_ITERS=$(echo "$3" | grep -E '^[0-9]+$' || echo "0")
SCALE_MODE=$4
ITERATIONS=$((START_ITER+CONT_ITERS))
echo "Scene: $SCENE"
echo "Start iteration: $START_ITER"
echo "Continue iterations: $CONT_ITERS"
echo "Total iterations: $ITERATIONS"

python analyse.py \
    -s /scratch/hz3496/3dgs_data/tandt_db/tandt/${SCENE} \
    --iterations $ITERATIONS \
    --model_path experiments_analyse/${SCENE}_analyse_${SCALE_MODE}_start${START_ITER}_cont${CONT_ITERS} \
    --lr_scale_mode $SCALE_MODE \
    --batch_grad_stats \
    --fixed_random_sequence \
    --disable_densification \
    --start_checkpoint experiments_analyse/${SCENE}_bsz_1_scale_sqrt/chkpnt${START_ITER}.pth

