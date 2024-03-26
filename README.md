# This Branch supports both DP and MP for 3DGS

## Cloning the Repository on the dp_system branch

```shell
# SSH
git clone git@github.com:TarzanZhao/gaussian-splatting.git -b dp_system --recursive
```

#### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```
If you want to use other name for this conda environment, you should change the `name:` field in the environment.yml

Then, we need to compile two dependent cuda repo `diff-gaussian-rasterization` and `simple-knn`. `diff-gaussian-rasterization` contains render cuda kernels, which will be continuously modified by us. Therefore, let us first install it in development mode. 
```shell
pip install -e submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

### Training

```shell

# 1 GPU
python train.py \
    -s /scratch/hz3496/3dgs_data/tandt_db/tandt/train \
    --iterations 7000 \
    --log_interval 50 \
    --log_folder experiments/debug_dp_1gpu \
    --model_path experiments/debug_dp_1gpu \
    --bsz 1 \
    --test_iterations 1000 7000 \
    --benchmark_stats

# 4 GPU Data Parallel using bsz=4, each GPU compute one data point.
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /scratch/hz3496/3dgs_data/tandt_db/tandt/train \
    --iterations 7000 \
    --log_interval 50 \
    --log_folder experiments/test_dp \
    --model_path experiments/test_dp \
    --render_distribution_adjust_mode "2" \
    --memory_distribution_mode "2" \
    --redistribute_gaussians_mode "1" \
    --loss_distribution_mode "general" \
    --test_iterations 1000 7000 \
    --dp_size 4 \
    --bsz 4 \
    --benchmark_stats

# 4 GPU Model Parallel using bsz=1, 4 GPUs compute single data point together.
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /scratch/hz3496/3dgs_data/tandt_db/tandt/train \
    --iterations 7000 \
    --log_interval 50 \
    --log_folder experiments/test_mp \
    --model_path experiments/test_mp \
    --render_distribution_adjust_mode "2" \
    --memory_distribution_mode "1" \
    --redistribute_gaussians_mode "1" \
    --loss_distribution_mode "general" \
    --test_iterations 1000 7000 \
    --dp_size 1 \
    --bsz 1 \
    --benchmark_stats

# 4 GPU Data Parall and Model Parallel used together, using bsz=2, 2 GPU compute one data point and the other 2 GPU compute the other data point.
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /scratch/hz3496/3dgs_data/tandt_db/tandt/train \
    --iterations 7000 \
    --log_interval 50 \
    --log_folder experiments/test_dp_and_mp \
    --model_path experiments/test_dp_and_mp \
    --render_distribution_adjust_mode "2" \
    --memory_distribution_mode "2" \
    --redistribute_gaussians_mode "1" \
    --loss_distribution_mode "general" \
    --test_iterations 1000 7000 \
    --dp_size 2 \
    --bsz 2 \
    --benchmark_stats
```
`--dp_size` should be the same as `--bsz`. 
In hz3496's account in greene, the dataset here is `/scratch/hz3496/3dgs_data/tandt_db/tandt/train`

### Rendering


### Evaluating metrics
