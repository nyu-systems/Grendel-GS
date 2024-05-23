

# python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --model_path /pscratch/sd/j/jy-nyu/last_scripts/rub/rub_16g_c \
#     --iteration 199985 \
#     --distributed_load \
#     --skip_train \
#     --skip_test \
#     --llffhold 83 \
#     --save_one_radii_distribution



# python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/360_v2/bicycle1080p \
#     --model_path /pscratch/sd/j/jy-nyu/last_scripts/mip360_1080p/4g_4b/e_bicycle \
#     --iteration 49997 \
#     --skip_train \
#     --skip_test \
#     --llffhold 8 \
#     --save_one_radii_distribution



python /global/homes/j/jy-nyu/gaussian-splatting/render.py \
    -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
    --model_path /pscratch/sd/j/jy-nyu/last_scripts/tandb/4g4b/train \
    --iteration 29997 \
    --skip_train \
    --skip_test \
    --llffhold 8 \
    --save_one_radii_distribution
