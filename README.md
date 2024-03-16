## Cloning the Repository

```shell
# SSH
git clone git@github.com:TarzanZhao/gaussian-splatting.git --recursive
```

#### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```
If you want to use other name for this conda environment, you should change the `name:` field in the environment.yml

### Training

```shell

# 1 GPU
python train.py \
    -s dataset path \
    --iterations 7000 \
    --log_interval 50 \
    --log_folder experiments/test_1gpu \
    --model_path experiments/test_1gpu \
    --benchmark_stats

# 4 GPU
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s dataset path \
    --iterations 7000 \
    --log_interval 50 \
    --log_folder experiments/test_4gpu \
    --model_path experiments/test_4gpu \
    --render_distribution_mode "2" \
    --redistribute_gaussians_mode "1" \
    --loss_distribution_mode "general" \
    --benchmark_stats
```
In hz3496's account in greene, it is `/scratch/hz3496/3dgs_data/tandt_db/tandt/train`

### Rendering


### Evaluating metrics
