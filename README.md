# DiffusionSfM

This repository contains the official implementation for **DiffusionSfM: Predicting Structure and Motion**
**via Ray Origin and Endpoint Diffusion**. The paper has been accepted to [CVPR 2025](https://cvpr.thecvf.com/Conferences/2025).

[Project Page](https://qitaozhao.github.io/DiffusionSfM) | arXiv (Coming Soon)

### News

- 2025.05.04: Initial code release.

## Install

1. Clone DiffusionSfM:

```bash
git clone https://github.com/QitaoZhao/DiffusionSfM.git
cd DiffusionSfM
```

2. Create the environment and install packages:

```bash
conda create -n diffusionsfm python=3.9
conda activate diffusionsfm

# enable nvcc
conda install -c conda-forge cudatoolkit-dev

### torch
# CUDA 11.7
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -r requirements.txt

### pytorch3D
# CUDA 11.7
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.7/download/linux-64/pytorch3d-0.7.7-py39_cu117_pyt201.tar.bz2

# xformers
conda install xformers -c xformers
```

Tested on:

- Springdale Linux 8.6 with torch 2.0.1 & CUDA 11.7 on A6000 GPUs.

> **Note:** If you encounter the error

> ImportError: .../libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent

> when importing PyTorch, refer to this [related issue](https://github.com/coleygroup/shepherd-score/issues/1) or try installing Intel MKL explicitly with:

```
conda install mkl==2024.0  
```

## Training

Set up wandb:

```bash
wandb login
```

See [docs/train.md](https://github.com/QitaoZhao/DiffusionSfM/blob/main/docs/train.md) for more detailed instructions on training.

## Evaluation

See [docs/eval.md](https://github.com/QitaoZhao/DiffusionSfM/blob/main/docs/eval.md) for instructions on how to run evaluation code.

## Cite DiffusionSfM

If you find this code helpful, please cite:

```
@inproceedings{zhao2025diffusionsfm,
  title={DiffusionSfM: Predicting Structure and Motion via Ray Origin and Endpoint Diffusion}, 
  author={Qitao Zhao and Amy Lin and Jeff Tan and Jason Y. Zhang and Deva Ramanan and Shubham Tulsiani},
  booktitle={CVPR},
  year={2025}
}
```