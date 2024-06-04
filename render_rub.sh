

data_path="/pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred"

python render.py \
    -s $data_path \
    --model_path /pscratch/sd/j/jy-nyu/release_expes/rub_4g_1 \
    --iteration 199997 \
    --distributed_load \
    --skip_train \
    --llffhold 83


