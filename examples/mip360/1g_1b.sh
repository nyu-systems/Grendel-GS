
if [ $# -ne 2 ]; then
    echo "Please specify exactly two arguments: the folder to save the experiments' log and checkpoints, and the folder of the dataset."
    exit 1
fi

expe_folder=$1
echo "The experiments will be saved in $expe_folder"
dataset_folder=$2
echo "The dataset is in $dataset_folder"

# scenes to be trained
SCENE=(counter bicycle stump garden room bonsai kitchen)
# batch size
BSZ=1

# Monitoring Settings
monitor_opts="--enable_timer \
--end2end_time \
--check_gpu_memory \
--check_cpu_memory"

for scene in ${SCENE[@]}; do
    expe_name="e_${scene}"

    # the following is to match the experiments setting in original gaussian splatting repository
    if [ "$scene" = "bicycle" ] || [ "$scene" = "stump" ] || [ "$scene" = "garden" ]; then
        image_folder="images_4"
    else
        image_folder="images_2"
    fi

    torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py \
        -s ${dataset_folder}/${scene} \
        --images ${image_folder} \
        --llffhold 8 \
        --iterations 30000 \
        --log_interval 250 \
        --model_path ${expe_folder}/1g_1b/${expe_name} \
        --bsz $BSZ \
        $monitor_opts \
        --test_iterations 7000 15000 30000 \
        --save_iterations 7000 30000 \
        --eval
done
