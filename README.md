# On Scaling Up 3D Gaussian Splatting Training
Hexu Zhao, Haoyang Weng*, Daohan Lu*, Ang Li, Jinyang Li, Aurojit Panda, Saining Xie (* indicates equal contribution)<br>
| [Webpage](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | [Full Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) | [Video](https://youtu.be/T_kXY43VZnk) | [FUNGRAPH project page](https://fungraph.inria.fr) |<br>
| [Pre-trained Models (14 GB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip) | [Evaluation Images (7 GB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip) |<br>
![Teaser image](assets/teaser.png)

This repository contains the official implementation associated with the paper "On Scaling Up 3D Gaussian Splatting Training", which is developed base on the official implementation of "3D Gaussian Splatting for Real-Time Radiance Field Rendering". We also release the two pretrained models and evaluation images reported in our paper for the two large-scale Rubble and MatrixCity Block_All scenes. 

Abstract: *3D Gaussian Splatting (3DGS) is gaining popularity for 3D reconstruction because of its superior visual quality and rendering speed. However, training is currently done on a single GPU, and thus cannot handle high-resolution and large-scale 3D reconstruction tasks due to the GPU's memory capacity limit. We build a distributed system, called Grendel, to partition 3DGS' parameters and parallelize its computation across multiple GPUs. As each Gaussian affects a small and changing subset of rendered pixels, Grendel relies on sparse all-to-all communication to transfer each required Gaussian to a pixel partition and performs dynamic load balancing. Unlike existing 3DGS systems that train using one camera view image at a time, Grendel supports batched training using multiple views. We explore different learning rate scaling strategies and identify the simple sqrt(batch size) scaling rule to be highly effective. Evaluation using large-scale high-resolution scenes show that Grendel can improve rendering quality by scaling up 3DGS quantity using multiple GPUs. On the "Rubble" dataset, we achieve a test PSNR of 27.28 by distributing 40.4 million Gaussians across 16 GPUs. By comparison, one achieves a PSNR of 26.28 when using 11.2 million Gaussians in order to fit in a single GPU's memory.*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:TarzanZhao/gaussian-splatting.git --recursive
```

## Overview

The codebase contains a distributed PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs. The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models.  I delete other functions of original 3DGS repository to make this repo focused. 


We only tested this repo on linux with nvidia GPU; Instructions for setting up and running are found in the section below. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)
- Please see FAQ for smaller VRAM configurations

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used Visual Studio 2019 for Windows)
- CUDA SDK 11 for PyTorch extensions, install *after* Visual Studio (we used 11.8, **known issues with 11.6**)
- C++ Compiler and CUDA SDK must be compatible

#### Setup

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```(I am not able to set up this yet)
If you want to use other name for this conda environment, you should change the `name:` field in the environment.yml

Dependencies:
- plyfile
- pytorch(version>=2.0.1???)
- torchvision
- tqdm



Then, we need to compile two dependent cuda repo `diff-gaussian-rasterization` and `simple-knn`. `diff-gaussian-rasterization` contains render cuda kernels.

```shell
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```



### Running

To run the optimizer, simply use

For single-GPU non-distributed training,
```shell
python train.py -s <path to COLMAP dataset>
python train.py -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train --model_path ./expe/train_1gpu
```

For 4 GPU distributed training and batch size of 4,
```shell
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --bsz 4 -s <path to COLMAP dataset>
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --bsz 4 -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train --model_path ./expe/train_4gpu
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP data set.
  #### --model_path / -m 
  Path where the trained model and loggings should be stored (```/tmp/gaussian_splatting``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --bsz
  The batch size(the number of camera views) in single step training. ```1``` by default.
  #### --lr_scale_mode
  The mode of scaling learning rate given larger batch size. ```sqrt``` by default.
  #### --preload_dataset_to_gpu
  Save all groundtruth images from the dataset in GPU, rather than load each image on-the-fly at each training step. 
  If dataset is large, preload_dataset_to_gpu will lead to OOM; when the dataset is small, preload_dataset_to_gpu could 
  speed up the training a little bit by avoiding some cpu-gpu communication. 
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.



</details>
<br>

### Rendering Trained point cloud

#### NOTE


### Rendering


### Evaluating metrics
