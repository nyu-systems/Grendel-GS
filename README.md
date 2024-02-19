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

### Training

```shell

torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /scratch/hz3496/3dgs_data/tandt_db/tandt/train \
    --iterations 300 \
    --log_interval 250 \
    --log_folder experiments/test \
    --model_path experiments/test \
    --adjust_div_stra \
    --adjust_mode "1" \
    --lazy_load_image \
    --memory_distribution \
    --image_distribution \
    --benchmark_stats

```

### Rendering


### Evaluating metrics
