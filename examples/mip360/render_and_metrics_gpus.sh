
if [ $# -ne 2 ]; then
    echo "Please specify exactly two arguments: the folder to save the experiments' log and checkpoints, and the folder of the dataset."
    exit 1
fi

expe_folder=$1
echo "The experiments are saved in $expe_folder"
dataset_folder=$2
echo "The dataset is in $dataset_folder"

# SCENE=(counter bicycle stump garden room bonsai kitchen)
# EXPE_sets=("1g_1b" "4g_1b" "4g_4b")
# BSZ_for_expesets=(1 1 4)

SCENE=(garden)
EXPE_sets=("4g_4b")
# BSZ_for_expesets=(1 1 4)
# NPROC_for_expesets=(1 4 4)

for i in ${!EXPE_sets[@]}; do
    expe_set=${EXPE_sets[$i]}
    # bsz=${BSZ_for_expesets[$i]}
    # nproc=${NPROC_for_expesets[$i]}
    bsz=4
    nproc=4

    for scene in ${SCENE[@]}; do

        if [ "$scene" = "bicycle" ] || [ "$scene" = "stump" ] || [ "$scene" = "garden" ]; then
            image_folder="images_4"
        elif [ "$scene" = "train" ]; then
            image_folder="images"
        else
            image_folder="images_2"
        fi

        EXPE_NAME="${expe_folder}/${expe_set}/e_${scene}"

        # you should change the checkpoints as it may depends on the batch size
        CHECKPOINTS=(29993)

        for iter in ${CHECKPOINTS[@]}; do
            torchrun --standalone --nnodes=1 --nproc-per-node=${nproc} render.py \
                -s ${dataset_folder}/${scene} \
                --images ${image_folder} \
                --model_path ${EXPE_NAME} \
                --iteration $iter \
                --llffhold 8 \
                --skip_train \
                --bsz ${bsz}
        done

        python metrics.py \
            --mode test \
            --model_paths ${EXPE_NAME}
        
        rm -rf ${EXPE_NAME}/test
    done

done



