BSZ=$1
SCENE=$2
SCALE_MODE=$3
echo "Batch size: $BSZ"
echo "Scene: $SCENE"
echo "Scale mode: $SCALE_MODE"

python train.py \
    -s /scratch/hz3496/3dgs_data/tandt_db/tandt/${SCENE} \
    --iterations 30000 \
    --model_path experiments_analyse/${SCENE}_bsz_${BSZ}_scale_${SCALE_MODE} \
    --bsz $BSZ \
    --lr_scale_mode $SCALE_MODE \
    --test_iterations 1000 7000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 100 1000 7000 10000 15000 20000 25000 30000\

    # --start_checkpoint experiments_test/chkpnt100.pth
    # --fixed_random_sequence \
