<div align="center">

Grendal-GS
===========================
_<h4>Gaussian Splatting At Scale by Distributed Training System</h4>_

### [Paper](http://arxiv.org/abs/2212.09748) | [Project Page](https://www.wpeebles.com/DiT)

<div align="left">

<div align="center">
    <img src="assets/teaser.png" width="900">
</div>

<details>
  <summary> <strong> Click Here to Download Pre-trained Models behind the above visualizations </strong> </summary>
  
  - [Pre-trained Rubble Model (On the left)](https://www.wpeebles.com/DiT), [Corresponding Evaluation Images](https://www.wpeebles.com/DiT)
  - [Pre-trained MatrixCity Model (On the right)](https://www.wpeebles.com/DiT), [Corresponding Evaluation Images](https://www.wpeebles.com/DiT)

</details>

<!-- 
### [Rubble Model ](https://www.wpeebles.com/DiT) | [Rubble Evaluation Images](https://www.wpeebles.com/DiT) | [MatrixCity Model](https://www.wpeebles.com/DiT) |  [MatrixCity Evaluation Images](https://www.wpeebles.com/DiT) 

### [Pre-trained Rubble Model(the left one)](https://www.wpeebles.com/DiT) | [Rubble Evaluation Images(the left one)](https://www.wpeebles.com/DiT) | [Pre-trained MatrixCity Model(the right one)](https://www.wpeebles.com/DiT) |  [MatrixCity Evaluation Images](https://www.wpeebles.com/DiT) 
-->



# Overview

We design and implement **Grendal-GS**, which serves as a distributed implementation of 3D Gaussian Splatting training. We aim to help 3DGS achieve its *scaling laws* the with support from distributed systems, just as the achievements of current LLMs rely on distributed training. 

By using Grendal, your 3DGS training could leverage multiple GPUs' capability to achieve significantly ***faster training***, supports a substantially ***more Gaussians*** in GPU memory, and ultimately allows for the reconstruction of ***larger-area***, ***higher-resolution*** scenes to better PSNR. Grendal-GS retains the original algorithm, making it a ***direct and safe replacement*** for original 3DGS implementation in any Gaussian Splatting workflow or application.

<!-- 
*Many more new features are developing now, following us!*
-->

Grendal-GS is continuously adding new features. Follow us for updates! Interested in contributing? [Email us!](mailto:hz3496@nyu.edu)

**Table of contents**
-----
- [How to use Grendal-GS](#how-to-use-grendal-gs)
    - [Setup](#setup)
    - [Training](#training)
    - [Render Pretrained-Model](#rendering)
    - [Calculate Metrics](#evaluating-metrics)
    - [Migrating from original 3DGS codebase](#migrating-from-original-3dgs-codebase)
- [Benefits and Examples](#benefits-and-examples)
- [Paper](#paper-and-citation)
- [Reference](#reference)
------

# How to use Grendal-GS

This repo and its dependency, render cuda code([diff-gaussian-rasterization](https://github.com/TarzanZhao/diff-gaussian-rasterization)), are both forks from the [original 3DGS implementation](https://github.com/graphdeco-inria/gaussian-splatting). Therefore, the usage is generally very similar to the original 3DGS. 

The two main differences are:

1. We support training on multiple GPUs, using the `torchrun` command-line utility provided by PyTorch to launch jobs.
2. We support batch sizes greater than 1, with the `--bsz` argument flag used to specify the batch size.


<!-- 
Our repository contains a distributed PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs. The optimizer uses PyTorch and CUDA extensions to produce trained models. 
-->

## Setup

### Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
git clone git@github.com:TarzanZhao/Dist-GS.git --recursive
```

### Pytorch Environment

Ensure you have Conda, GPU with compatible driver and cuda environment installed on your machine, as prerequisites. Then please install `PyTorch`, `Torchvision`, `Plyfile`, `tqdm` which are essential packages. Make sure PyTorch version >= 1.10 to have torchrun for distributed training. Finally, compile and install two dependent cuda repo `diff-gaussian-rasterization` and `simple-knn` containing our customized cuda kernels for rendering and etc.

We provide a yml file for easy environment setup. However, you should choose the versions to match your local running environment. 
```
conda env create --file environment.yml
conda activate gaussian_splatting
```

NOTES: We kept additional dependencies minimal compared to the original 3DGS. For environment setup issues, maybe you could refer to the [original 3DGS repo issue section](https://github.com/graphdeco-inria/gaussian-splatting/issues) first.

### Dataset

We use colmap format to load dataset. Therefore, please download and unzip colmap datasets before trainning, for example [Mip360 dataset](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) and 4 scenes from [Tanks&Temple and DeepBlending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). 



<!-- Its core idea is to leverage more GPU's capability during training and to increase the batch size to utilize these GPU better. Therefore, we could accommodate much more gaussian primitives in large-scale and high resolution scenes, and speed up at the same time. It contains a distributed PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs. The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models.  -->





<!--     
_Authors: [**Hexu Zhao¹**](https://tarzanzhao.github.io), [**Haoyang Weng¹\***](https://egalahad.github.io), [**Daohan Lu¹\***](https://daohanlu.github.io), [**Ang Li²**](https://www.angliphd.com), [**Jinyang Li¹**](https://www.news.cs.nyu.edu/~jinyang/), [**Aurojit Panda¹**](https://cs.nyu.edu/~apanda/), [**Saining Xie¹**](https://www.sainingxie.com)_  (\* *Indicates equal contribution*)

_Affiliations: [**¹New York University**](https://cs.nyu.edu/home/index.html), [**²Pacific Northwest National Laboratory**](https://www.pnnl.gov)_
- **[Arxiv Paper](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fp6/03-05-2024/README.md)**
- **[Pre-trained Models (14 GB)](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fp6/03-05-2024/README.md)**
- **[Evaluation Images (7 GB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip)**


(TODO: check all the links here before releasing)

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
(TODO:change it to our bib)
</section> 
-->

<!--     
---

# Repository Overview
Our repository is a research-focused framework for distributed Gaussian Splatting training, as detailed in the paper. Its core idea is to leverage more GPU by distributed computation during training and to increase the batch size to utilize these GPU better. Therefore, we could accommodate much more gaussian primitives in large-scale and high resolution scenes, and speed up at the same time. It contains a distributed PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs. The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

Advantages of using our distributed implementation for gaussians splatting:
1. Train at higher resolution.
2. Train larger scenes.
3. Train the same scene dozens of times faster.
4. Increased PSNR with the same training time.
-->



## Training

For single-GPU non-distributed training with batch size of 1:
```shell
python train.py -s <path to COLMAP dataset>
```

For 4 GPU distributed training and batch size of 4:
```shell
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --bsz 4 -s <path to COLMAP dataset>
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

## Rendering

```shell
python render.py -s <path to COLMAP dataset> --model_path <path to folder of saving model> 
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --distributed_load
  If point cloud models are saved distributedly during training, we should set this flag to load all of them.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 

  **The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.** 

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --llffhold
  The training/test split ratio in the whole dataset for evaluation. llffhold=8 means 1/8 is used as test set and others are used as train set.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.

</details>

## Evaluating metrics

```shell
python metrics.py --model_path <path to folder of saving model> 
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for metrics.py</span></summary>

  #### --model_paths / -m 
  Space-separated list of model paths for which metrics should be computed.
</details>
<br>

## Migrating from original 3DGS codebase

If you are currently using the original 3DGS codebase for training in your application, you can effortlessly switch to our codebase because we haven't made any algorithmic changes. This will allow you to train faster and successfully train larger, higher-precision scenes without running out of memory (OOM) within a reasonable time frame. 

It is worth noting that we only support the training functionality; this repository does not include the interactive viewer, network viewer, or colmap features from the original 3DGS. We are actively developing to support more features. Please let us know your needs or directly contribute to our project. Thank you!

---



# Benefits and Examples

## Significantly Faster Training Without Compromising Reconstruction Quality On Mip360 Dataset

### Training Time

| 30k Train Time(min)   |   stump |   bicycle |   kitchen |   room |   counter |   garden |   bonsai |
|:----------------------|--------:|----------:|----------:|-------:|----------:|---------:|---------:|
| 1 GPU + Batch Size=1  |   24.03 |     30.18 |     25.58 |  22.45 |     21.6  |    30.15 |    19.18 |
| 4 GPU + Batch Size=1  |    9.07 |     11.67 |      9.53 |   8.93 |      8.82 |    10.85 |     8.03 |
| 4 GPU + Batch Size=4  |    5.22 |      6.47 |      6.98 |   6.18 |      5.98 |     6.48 |     5.28 |

### Test PSNR

| 30k Test PSNR        |   stump |   bicycle |   kitchen |   room |   counter |   garden |   bonsai |
|:---------------------|--------:|----------:|----------:|-------:|----------:|---------:|---------:|
| 1 GPU + Batch Size=1 |   26.61 |     25.21 |     31.4  |  31.4  |     28.93 |    27.27 |    32.01 |
| 4 GPU + Batch Size=1 |   26.65 |     25.19 |     31.41 |  31.38 |     28.98 |    27.28 |    31.92 |
| 4 GPU + Batch Size=4 |   26.59 |     25.17 |     31.37 |  31.32 |     28.98 |    27.2  |    31.94 |
---

### Reproduction Instructions

1. Download and unzip the [Mip360 dataset](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip).
2. Activate the appropriate conda/python environment.
3. To execute all experiments and generate this table, run the following command:
   ```bash
   bash examples/mip360/eval_all_mip360.sh <path_to_save_experiment_results> <path_to_mip360_dataset>
   ```

## Significantly Speed up and Reduce per-GPU memory usage on Mip360 at *4K Resolution*

| Configuration                  | 50k Training Time   |   Memory Per GPU |   PSNR |
|:-------------------------------|:--------------------|-----------------:|-------:|
| bicycle + 1 GPU + Batch Size=1 | 2h 38min            |            37.18 |  23.78 |
| bicycle + 4 GPU + Batch Size=1 | 0h 50min            |            10.39 |  23.79 |
| garden + 1 GPU + Batch Size=1  | 2h 49min            |            29.87 |  26.06 |
| garden + 4 GPU + Batch Size=1  | 0h 50min            |             7.88 |  26.06 |

Unlike the typical approach of downsampling the Mip360 dataset by a factor of four before training, our system can train directly at full resolution. The bicycle and garden images have resolutions of 4946x3286 and 5187x3361, respectively. Our distributed system demonstrates that we can significantly accelerate and reduce memory usage per GPU by several folds without sacrificing quality.

### Reproduction Instructions

Set up the dataset and Python environment as outlined previously, then execute the following:
```bash
   bash examples/mip360_4k/eval_mip360_4k.sh <path_to_save_experiment_results> <path_to_mip360_dataset>
   ```

## Train in 45 Seconds on Tanks&Temple at *1K Resolution*

| Configuration                | 7k Training Time   |   7k test PSNR | 30k Training Time   |   30k test PSNR |
|:-----------------------------|:-------------------|---------------:|:--------------------|----------------:|
| train + 4 GPU + Batch Size=8 | 44s                |          19.37 | 3min 30s            |           21.87 |
| truck + 4 GPU + Batch Size=8 | 45s                |          23.79 | 3min 39s            |           25.35 |

Tanks&Temples dataset includes train and truck scenes with resolutions of 980x545 and 979x546, respectively. Utilizing 4 GPUs, we've managed to train on these small scenes to a reasonable quality in just 45 seconds(7k iterations). In the original Gaussian splatting papers, achieving a test PSNR of 18.892 and 23.506 at 7K resolution was considered good on train and truck, respectively. Our results are comparable to these benchmarks.

### Reproduction Instructions

Set up the [Tanks&Temple and DeepBlending Dataset](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) and Python environment as outlined previously, then execute the following:
```bash
   bash examples/train_truck_1k/eval_train_truck_1k.sh <path_to_save_experiment_results> <path_to_tandb_dataset>
   ```

(TODO: check these scripts have no side-effects)

## Experimental Setup for all experiments statistics above

- **Hardware**: 4x 40GB NVIDIA A100 GPUs
- **Interconnect**: Fully-connected Bidirectional 25GB/s NVLINK

---



# New features [Please check regularly!]

- We will release our optimized cuda kernels within gaussian splatting soon for further speed up. 
- We will support gsplat later as another choice of our cuda kernel backend. 

# Paper and Citation

Our system design, analysis of large-batch training dynamics, and insights from scaling up are all documented in the paper below: 

> [**On Scaling Up 3D Gaussian Splatting Training**](https://www.wpeebles.com/DiT)<br>
> [**Hexu Zhao¹**](https://tarzanzhao.github.io), [**Haoyang Weng¹\***](https://egalahad.github.io), [**Daohan Lu¹\***](https://daohanlu.github.io), [**Ang Li²**](https://www.angliphd.com), [**Jinyang Li¹**](https://www.news.cs.nyu.edu/~jinyang/), [**Aurojit Panda¹**](https://cs.nyu.edu/~apanda/), [**Saining Xie¹**](https://www.sainingxie.com)  (\* *co-second authors*)
> <br>¹New York University, ²Pacific Northwest National Laboratory <br>

TODO: change to our own bibtex
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

# Reference

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
