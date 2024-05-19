

# bash /global/homes/j/jy-nyu/gaussian-splatting/building_scripts/bui2k_1g_test.sh


# source ~/zhx.sh

# head_node_ip="nid003080"
# port=27709
# rid=104

expe_name="uni_4g_1"

python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
    -s /pscratch/sd/j/jy-nyu/datasets/university4/university4 \
    --model_path /pscratch/sd/j/jy-nyu/uni_expes/$expe_name \
    --iteration 79997 \
    --sample_freq 5 \
    --num_train_cameras 50 \
    --num_test_cameras 50 \
    --distributed_load








