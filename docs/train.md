## Training Directions

### Prepare CO3D Dataset

Please refer to the instructions from [RayDiffusion](https://github.com/jasonyzhang/RayDiffusion/blob/main/docs/train.md#training-directions) to set up the CO3D dataset.

### Setting up `accelerate`

Use `accelerate config` to set up `accelerate`. We recommend using multiple GPUs without any mixed precision (we handle AMP ourselves).

### Training models

Our model is trained in two stages. In the first stage, we train a *sparse model* that predicts ray origins and endpoints at a low resolution (16×16). In the second stage, we initialize the dense model using the DiT weights from the sparse model and append a DPT decoder to produce high-resolution outputs (256×256 ray origins and endpoints).

To train the sparse model, run:

```
accelerate launch --multi_gpu --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 train.py \
    training.batch_size=8 \
    training.max_iterations=400000 \
    model.num_images=8 \
    dataset.name=co3d \
    debug.project_name=diffusionsfm_co3d \
    debug.run_name=co3d_diffusionsfm_sparse
```

To train the dense model (initialized from the sparse model weights), run:

```
accelerate launch --multi_gpu --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 train.py \
    training.batch_size=4 \
    training.max_iterations=800000 \
    model.num_images=8 \
    dataset.name=co3d \
    debug.project_name=diffusionsfm_co3d \
    debug.run_name=co3d_diffusionsfm_dense \
    training.dpt_head=True \
    training.full_num_patches_x=256 \
    training.full_num_patches_y=256 \
    training.gradient_clipping=True \
    training.reinit=True \
    training.freeze_encoder=True \
    model.freeze_transformer=True \
    training.pretrain_path=</path/to/your/checkpoint>.pth
```

Some notes:

- `batch_size` refers to the batch size per GPU. The total batch size will be `batch_size * num_gpu`.
- Depending on your setup, you can adjust the number of GPUs and batch size. You may also need to adjust the number of training iterations accordingly.
- You can resume training from a checkpoint by specifying `train.resume=True hydra.run.dir=/path/to/your/output_dir`
- If you are getting NaNs, try turning off mixed precision. This will increase the amount of memory used.

For debugging, we recommend using a single-GPU job with a single category:

```
accelerate launch train.py training.batch_size=4 dataset.category=apple debug.wandb=False hydra.run.dir=output_debug
```