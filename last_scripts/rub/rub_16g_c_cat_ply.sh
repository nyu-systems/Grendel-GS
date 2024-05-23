

python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
    -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
    --model_path /pscratch/sd/j/jy-nyu/last_scripts/rub/rub_16g_c \
    --iteration 199985 \
    --distributed_load \
    --skip_train \
    --skip_test \
    --llffhold 83 \
    --save_catted_gaussian_model

# rub_16g_7_c2