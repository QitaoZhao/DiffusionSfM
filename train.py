"""
Configurations can be overwritten by adding: key=value
Use debug.wandb=False to disable logging to wandb.
"""

import datetime
from datetime import timedelta
import os
import random
import socket
import time
from glob import glob

import hydra
import ipdb  # noqa: F401
import numpy as np
import omegaconf
import torch
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from pytorch3d.renderer import PerspectiveCameras

from diffusionsfm.dataset.co3d_v2 import Co3dDataset, unnormalize_image_for_vis
# from diffusionsfm.dataset.multiloader import get_multiloader, MultiDataset
from diffusionsfm.eval.eval_category import evaluate
from diffusionsfm.model.diffuser import RayDiffuser
from diffusionsfm.model.diffuser_dpt import RayDiffuserDPT
from diffusionsfm.model.scheduler import NoiseScheduler
from diffusionsfm.utils.rays import cameras_to_rays, normalize_cameras_batch, compute_ndc_coordinates
from diffusionsfm.utils.visualization import (
    create_training_visualizations,
    view_color_coded_images_from_tensor,
)

os.umask(000)  # Default to 777 permissions


class Trainer(object):
    def __init__(self, cfg):
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.cfg = cfg
        self.debug = cfg.debug
        self.resume = cfg.training.resume
        self.pretrain_path = cfg.training.pretrain_path

        self.batch_size = cfg.training.batch_size
        self.max_iterations = cfg.training.max_iterations
        self.mixed_precision = cfg.training.mixed_precision
        self.interval_visualize = cfg.training.interval_visualize
        self.interval_save_checkpoint = cfg.training.interval_save_checkpoint
        self.interval_delete_checkpoint = cfg.training.interval_delete_checkpoint
        self.interval_evaluate = cfg.training.interval_evaluate
        self.delete_all = cfg.training.delete_all_checkpoints_after_training
        self.freeze_encoder = cfg.training.freeze_encoder
        self.translation_scale = cfg.training.translation_scale
        self.regression = cfg.training.regression
        self.prob_unconditional = cfg.training.prob_unconditional
        self.load_extra_cameras = cfg.training.load_extra_cameras
        self.calculate_intrinsics = cfg.training.calculate_intrinsics
        self.distort = cfg.training.distort
        self.diffuse_origins_and_endpoints = cfg.training.diffuse_origins_and_endpoints
        self.diffuse_depths = cfg.training.diffuse_depths
        self.depth_resolution = cfg.training.depth_resolution
        self.dpt_head = cfg.training.dpt_head
        self.full_num_patches_x = cfg.training.full_num_patches_x
        self.full_num_patches_y = cfg.training.full_num_patches_y
        self.dpt_encoder_features = cfg.training.dpt_encoder_features
        self.nearest_neighbor = cfg.training.nearest_neighbor
        self.no_bg_targets = cfg.training.no_bg_targets
        self.unit_normalize_scene = cfg.training.unit_normalize_scene
        self.sd_scale = cfg.training.sd_scale
        self.bfloat = cfg.training.bfloat
        self.first_cam_mediod = cfg.training.first_cam_mediod
        self.normalize_first_camera = cfg.training.normalize_first_camera
        self.gradient_clipping = cfg.training.gradient_clipping
        self.l1_loss = cfg.training.l1_loss
        self.reinit = cfg.training.reinit

        if self.first_cam_mediod:
            assert self.normalize_first_camera

        self.pred_x0 = cfg.model.pred_x0
        self.num_patches_x = cfg.model.num_patches_x
        self.num_patches_y = cfg.model.num_patches_y
        self.depth = cfg.model.depth
        self.num_images = cfg.model.num_images
        self.num_visualize = min(self.batch_size, 2)
        self.random_num_images = cfg.model.random_num_images
        self.feature_extractor = cfg.model.feature_extractor
        self.append_ndc = cfg.model.append_ndc
        self.use_homogeneous = cfg.model.use_homogeneous
        self.freeze_transformer = cfg.model.freeze_transformer
        self.cond_depth_mask = cfg.model.cond_depth_mask

        self.dataset_name = cfg.dataset.name
        self.shape = cfg.dataset.shape
        self.apply_augmentation = cfg.dataset.apply_augmentation
        self.mask_holes = cfg.dataset.mask_holes
        self.image_size = cfg.dataset.image_size

        if not self.regression and (self.diffuse_origins_and_endpoints or self.diffuse_depths):
            assert self.mask_holes or self.cond_depth_mask

        if self.regression:
            assert self.pred_x0

        self.start_time = None
        self.iteration = 0
        self.epoch = 0
        self.wandb_id = None

        self.hostname = socket.gethostname()

        if self.dpt_head:
            find_unused_parameters = True
        else:
            find_unused_parameters = False

        ddp_scaler = DistributedDataParallelKwargs(
            find_unused_parameters=find_unused_parameters
        )
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        self.accelerator = Accelerator(
            even_batches=False,
            device_placement=False,
            kwargs_handlers=[ddp_scaler, init_kwargs],
        )
        self.device = self.accelerator.device

        scheduler = NoiseScheduler(
            type=cfg.noise_scheduler.type,
            max_timesteps=cfg.noise_scheduler.max_timesteps,
            beta_start=cfg.noise_scheduler.beta_start,
            beta_end=cfg.noise_scheduler.beta_end,
        )
        if self.dpt_head:
            self.model = RayDiffuserDPT(
                depth=self.depth,
                width=self.num_patches_x,
                P=1,
                max_num_images=self.num_images,
                noise_scheduler=scheduler,
                freeze_encoder=self.freeze_encoder,
                feature_extractor=self.feature_extractor,
                append_ndc=self.append_ndc,
                use_unconditional=self.prob_unconditional > 0,
                diffuse_depths=self.diffuse_depths,
                depth_resolution=self.depth_resolution,
                encoder_features=self.dpt_encoder_features,
                use_homogeneous=self.use_homogeneous,
                freeze_transformer=self.freeze_transformer,
                cond_depth_mask=self.cond_depth_mask,
            ).to(self.device)
        else:
            self.model = RayDiffuser(
                depth=self.depth,
                width=self.num_patches_x,
                P=1,
                max_num_images=self.num_images,
                noise_scheduler=scheduler,
                freeze_encoder=self.freeze_encoder,
                feature_extractor=self.feature_extractor,
                append_ndc=self.append_ndc,
                use_unconditional=self.prob_unconditional > 0,
                diffuse_depths=self.diffuse_depths,
                depth_resolution=self.depth_resolution,
                use_homogeneous=self.use_homogeneous,
                cond_depth_mask=self.cond_depth_mask,
            ).to(self.device)

        if self.dpt_head:
            depth_size = self.full_num_patches_x
        elif self.depth_resolution > 1:
            depth_size = self.num_patches_x * self.depth_resolution
        else:
            depth_size = self.num_patches_x
        self.depth_size = depth_size

        if self.dataset_name == "multi":
            self.dataset, self.train_dataloader, self.test_dataset = get_multiloader(
                num_images=self.num_images,
                apply_augmentation=self.apply_augmentation,
                load_extra_cameras=self.load_extra_cameras,
                distort_image=self.distort,
                center_crop=self.diffuse_origins_and_endpoints or self.diffuse_depths,
                crop_images=not (self.diffuse_origins_and_endpoints or self.diffuse_depths),
                load_depths=self.diffuse_origins_and_endpoints or self.diffuse_depths,
                depth_size=depth_size,
                mask_holes=self.mask_holes,
                img_size=self.image_size,
                batch_size=self.batch_size,
                num_workers=cfg.training.num_workers,
                dust3r_pairs=True,
            )
        elif self.dataset_name == "co3d":
            self.dataset = Co3dDataset(
                category=self.shape,
                split="train",
                num_images=self.num_images,
                apply_augmentation=self.apply_augmentation,
                load_extra_cameras=self.load_extra_cameras,
                distort_image=self.distort,
                center_crop=self.diffuse_origins_and_endpoints or self.diffuse_depths,
                crop_images=not (self.diffuse_origins_and_endpoints or self.diffuse_depths),
                load_depths=self.diffuse_origins_and_endpoints or self.diffuse_depths,
                depth_size=depth_size,
                mask_holes=self.mask_holes,
                img_size=self.image_size,
            )
            self.train_dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            self.test_dataset = Co3dDataset(
                category=self.shape,
                split="test",
                num_images=self.num_images,
                apply_augmentation=False,
                load_extra_cameras=self.load_extra_cameras,
                distort_image=self.distort,
                center_crop=self.diffuse_origins_and_endpoints or self.diffuse_depths,
                crop_images=not (self.diffuse_origins_and_endpoints or self.diffuse_depths),
                load_depths=self.diffuse_origins_and_endpoints or self.diffuse_depths,
                depth_size=depth_size,
                mask_holes=self.mask_holes,
                img_size=self.image_size,
            )
        else:
            raise NotImplementedError(f"Dataset '{self.dataset_name}' is not supported.")
        self.lr = 1e-4

        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if self.accelerator.is_main_process:
            name = os.path.basename(self.output_dir)
            name += f"_{self.debug.run_name}"

            print("Output dir:", self.output_dir)
            with open(os.path.join(self.output_dir, name), "w"):
                # Create empty tag with name
                pass
            self.name = name

            conf_dict = omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
            conf_dict["output_dir"] = self.output_dir
            conf_dict["hostname"] = self.hostname

        if self.dpt_head:
            self.init_optimizer_with_separate_lrs()
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.gradscaler = torch.cuda.amp.GradScaler(growth_interval=100000, enabled=self.mixed_precision)

        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

        if self.resume:
            checkpoint_files = sorted(glob(os.path.join(self.checkpoint_dir, "*.pth")))
            last_checkpoint = checkpoint_files[-1]
            print("Resuming from checkpoint:", last_checkpoint)
            self.load_model(last_checkpoint, load_metadata=True)
        elif self.pretrain_path != "":
            print("Loading pretrained model:", self.pretrain_path)
            self.load_model(self.pretrain_path, load_metadata=False)

        if self.accelerator.is_main_process:
            mode = "online" if cfg.debug.wandb else "disabled"
            if self.wandb_id is None:
                self.wandb_id = wandb.util.generate_id()
            self.wandb_run = wandb.init(
                mode=mode,
                name=name,
                project=cfg.debug.project_name,
                config=conf_dict,
                resume=self.resume,
                id=self.wandb_id,
            )
            wandb.define_metric("iteration")
            noise_schedule = self.get_module().noise_scheduler.plot_schedule(
                return_image=True
            )
            wandb.log(
                {"Schedule": wandb.Image(noise_schedule, caption="Noise Schedule")}
            )

    def get_module(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model

        return model
    
    def init_optimizer_with_separate_lrs(self):
        print("Use different LRs for the DINOv2 encoder and DiT!")

        feature_extractor_params = [
            p for n, p in self.model.feature_extractor.named_parameters()
        ]
        feature_extractor_param_names = [
            "feature_extractor." + n for n, _ in self.model.feature_extractor.named_parameters()
        ]
        ray_predictor_params = [
            p for n, p in self.model.ray_predictor.named_parameters()
        ]
        ray_predictor_param_names = [
            "ray_predictor." + n for n, p in self.model.ray_predictor.named_parameters()
        ]
        other_params = [
            p for n, p in self.model.named_parameters()
            if n not in feature_extractor_param_names + ray_predictor_param_names
        ]

        self.optimizer = torch.optim.Adam([
            {'params': feature_extractor_params, 'lr': self.lr * 0.1},  # Lower LR for feature extractor
            {'params': ray_predictor_params, 'lr': self.lr * 0.1},      # Lower LR for DIT (ray_predictor)
            {'params': other_params, 'lr': self.lr}                     # Normal LR for other parts of the model
        ])

    def train(self):
        while self.iteration < self.max_iterations:
            for batch in self.train_dataloader:
                t0 = time.time()
                self.optimizer.zero_grad()

                float_type = torch.bfloat16 if self.bfloat else torch.float16
                with torch.cuda.amp.autocast(
                    enabled=self.mixed_precision, dtype=float_type
                ):
                    images = batch["image"].to(self.device)
                    focal_lengths = batch["focal_length"].to(self.device)
                    crop_params = batch["crop_parameters"].to(self.device)
                    principal_points = batch["principal_point"].to(self.device)
                    R = batch["R"].to(self.device)
                    T = batch["T"].to(self.device)
                    if "distortion_coefficients" in batch:
                        distortion_coefficients = batch["distortion_coefficients"]
                    else:
                        distortion_coefficients = [None for _ in range(R.shape[0])]

                    depths = batch["depth"].to(self.device)
                    if self.no_bg_targets:
                        masks = batch["depth_masks"].to(self.device).bool()

                    cameras_og = [
                        PerspectiveCameras(
                            focal_length=focal_lengths[b],
                            principal_point=principal_points[b],
                            R=R[b],
                            T=T[b],
                            device=self.device,
                        )
                        for b in range(self.batch_size)
                    ]

                    cameras, _ = normalize_cameras_batch(
                        cameras=cameras_og,
                        scale=self.translation_scale,
                        normalize_first_camera=self.normalize_first_camera,
                        depths=(
                            None
                            if not (self.diffuse_origins_and_endpoints or self.diffuse_depths)
                            else depths
                        ),
                        first_cam_mediod=self.first_cam_mediod,
                        crop_parameters=crop_params,
                        num_patches_x=self.depth_size,
                        num_patches_y=self.depth_size,
                        distortion_coeffs=distortion_coefficients,
                    )

                    # Now that cameras are normalized, fix shapes of camera parameters
                    if self.load_extra_cameras or self.random_num_images:
                        if self.random_num_images:
                            num_images = torch.randint(2, self.num_images + 1, (1,))
                        else:
                            num_images = self.num_images

                        # The correct number of images is already loaded.
                        # Only need to modify these camera parameters shapes.
                        focal_lengths = focal_lengths[:, :num_images]
                        crop_params = crop_params[:, :num_images]
                        R = R[:, :num_images]
                        T = T[:, :num_images]
                        images = images[:, :num_images]
                        depths = depths[:, :num_images]
                        masks = masks[:, :num_images]

                        cameras = [
                            PerspectiveCameras(
                                focal_length=cameras[b].focal_length[:num_images],
                                principal_point=cameras[b].principal_point[:num_images],
                                R=cameras[b].R[:num_images],
                                T=cameras[b].T[:num_images],
                                device=self.device,
                            )
                            for b in range(self.batch_size)
                        ]

                    if self.regression:
                        low = self.get_module().noise_scheduler.max_timesteps - 1
                    else:
                        low = 0

                    t = torch.randint(
                        low=low,
                        high=self.get_module().noise_scheduler.max_timesteps,
                        size=(self.batch_size,),
                        device=self.device,
                    )

                    if self.prob_unconditional > 0:
                        unconditional_mask = (
                            (torch.rand(self.batch_size) < self.prob_unconditional)
                            .float()
                            .to(self.device)
                        )
                    else:
                        unconditional_mask = None

                    if self.distort:
                        raise NotImplementedError()
                    else:
                        gt_rays = []
                        rays_dirs = []
                        rays = []
                        for i, (camera, crop_param, depth) in enumerate(
                            zip(cameras, crop_params, depths)
                        ):
                            if self.diffuse_origins_and_endpoints:
                                mode = "segment"
                            else:
                                mode = "plucker"

                            r = cameras_to_rays(
                                cameras=camera,
                                num_patches_x=self.full_num_patches_x,
                                num_patches_y=self.full_num_patches_y,
                                crop_parameters=crop_param,
                                depths=depth,
                                mode=mode,
                                depth_resolution=self.depth_resolution,
                                nearest_neighbor=self.nearest_neighbor,
                                distortion_coefficients=distortion_coefficients[i],
                            )
                            rays_dirs.append(r.get_directions())
                            gt_rays.append(r)

                            if self.diffuse_origins_and_endpoints:
                                assert r.mode == "segment"
                            elif self.diffuse_depths:
                                assert r.mode == "plucker"

                            if self.unit_normalize_scene:
                                if self.diffuse_origins_and_endpoints:
                                    assert r.mode == "segment"
                                    # Let's say SD should be 0.5
                                    scale = r.get_segments().std() * self.sd_scale

                                    if scale.isnan().any():
                                        assert False

                                    camera.T /= scale
                                    r.rays /= scale
                                    depths[i] /= scale
                                else:
                                    assert r.mode == "plucker"
                                    scale = r.depths.std() * self.sd_scale

                                    if scale.isnan().any():
                                        assert False

                                    camera.T /= scale
                                    r.depths /= scale
                                    depths[i] /= scale

                            rays.append(
                                r.to_spatial(
                                    include_ndc_coordinates=self.append_ndc,
                                    include_depths=self.diffuse_depths,
                                    use_homogeneous=self.use_homogeneous,
                                )
                            )

                    rays_tensor = torch.stack(rays, dim=0)

                    if self.append_ndc:
                        ndc_coordinates = rays_tensor[..., -2:, :, :]
                        rays_tensor = rays_tensor[..., :-2, :, :]

                        if self.dpt_head:
                            xy_grid = compute_ndc_coordinates(
                                crop_params,
                                num_patches_x=self.depth_size // 16,
                                num_patches_y=self.depth_size // 16,
                                distortion_coeffs=distortion_coefficients,
                            )[..., :2]
                            ndc_coordinates = xy_grid.permute(0, 1, 4, 2, 3).contiguous()

                    else:
                        ndc_coordinates = None

                    if self.cond_depth_mask:
                        condition_mask = masks
                    else:
                        condition_mask = None

                    if rays_tensor.isnan().any():
                        import pickle

                        with open("bad.json", "wb") as f:
                            pickle.dump(batch, f)
                        ipdb.set_trace()

                    eps_pred, eps = self.model(
                        images=images,
                        rays=rays_tensor,
                        t=t,
                        ndc_coordinates=ndc_coordinates,
                        unconditional_mask=unconditional_mask,
                        depth_mask=condition_mask,
                    )
                    if self.pred_x0:
                        target = rays_tensor
                    else:
                        target = eps

                    if self.no_bg_targets:
                        C = eps_pred.shape[2]
                        loss_masks = masks.unsqueeze(2).repeat(1, 1, C, 1, 1)
                        eps_pred = loss_masks * eps_pred
                        target = loss_masks * target

                    loss = 0

                    if self.l1_loss:
                        loss_reconstruction = torch.mean(torch.abs(eps_pred - target))
                    else:
                        loss_reconstruction = torch.mean((eps_pred - target) ** 2)

                    loss += loss_reconstruction

                if self.mixed_precision:
                    self.gradscaler.scale(loss).backward()

                    scaled_norm = 0
                    for p in self.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            scaled_norm += param_norm.item() ** 2
                    scaled_norm = scaled_norm ** 0.5

                    if self.gradient_clipping and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.get_module().parameters(), 1
                        )

                    clipped_norm = 0
                    for p in self.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            clipped_norm += param_norm.item() ** 2
                    clipped_norm = clipped_norm ** 0.5

                    self.gradscaler.unscale_(self.optimizer)
                    unscaled_norm = 0
                    for p in self.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            unscaled_norm += param_norm.item() ** 2
                    unscaled_norm = unscaled_norm ** 0.5

                    self.gradscaler.step(self.optimizer)
                    self.gradscaler.update()
                else:
                    self.accelerator.backward(loss)

                    if self.gradient_clipping and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.get_module().parameters(), 10
                        )
                    self.optimizer.step()

                if self.accelerator.is_main_process:
                    if self.iteration % 10 == 0:
                        self.log_info(
                            loss_reconstruction,
                            t0,
                            self.lr,
                            scaled_norm,
                            unscaled_norm,
                            clipped_norm,
                        )

                    if self.iteration % self.interval_visualize == 0:
                        self.visualize(
                            images=unnormalize_image_for_vis(images.clone()),
                            cameras_gt=cameras,
                            depths=depths,
                            crop_parameters=crop_params,
                            distortion_coefficients=distortion_coefficients,
                            depth_mask=masks,
                        )

                    if self.iteration % self.interval_save_checkpoint == 0 and self.iteration != 0:
                        self.save_model()

                    if self.iteration % self.interval_delete_checkpoint == 0:
                        self.clear_old_checkpoints(self.checkpoint_dir)

                    if (
                        self.iteration % self.interval_evaluate == 0
                        and self.iteration > 0
                    ):
                        self.evaluate_train_acc()

                    if self.iteration >= self.max_iterations + 1:
                        if self.delete_all:
                            self.clear_old_checkpoints(
                                self.checkpoint_dir, clear_all_old=True
                            )
                        return

                self.iteration += 1

                if self.reinit and self.iteration >= 50000:
                    state_dict = self.get_module().state_dict()
                    self.model = RayDiffuserDPT(
                        depth=self.depth,
                        width=self.num_patches_x,
                        P=1,
                        max_num_images=self.num_images,
                        noise_scheduler=self.get_module().noise_scheduler,
                        freeze_encoder=False,
                        feature_extractor=self.feature_extractor,
                        append_ndc=self.append_ndc,
                        use_unconditional=self.prob_unconditional > 0,
                        diffuse_depths=self.diffuse_depths,
                        depth_resolution=self.depth_resolution,
                        encoder_features=self.dpt_encoder_features,
                        use_homogeneous=self.use_homogeneous,
                        freeze_transformer=False,
                        cond_depth_mask=self.cond_depth_mask,
                    ).to(self.device)

                    self.init_optimizer_with_separate_lrs()
                    self.gradscaler = torch.cuda.amp.GradScaler(growth_interval=100000, enabled=self.mixed_precision)

                    self.model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer
                    )

                    msg = self.get_module().load_state_dict(
                        state_dict,
                        strict=True,
                    )
                    print(msg)

                    self.reinit = False

            self.epoch += 1

    def load_model(self, path, load_metadata=True):
        save_dict = torch.load(path, map_location=self.device)
        del save_dict["state_dict"]["ray_predictor.x_pos_enc.image_pos_table"]

        if not self.resume:
            if len(save_dict["state_dict"]["scratch.input_conv.weight"].shape) == 2 and self.dpt_head:
                print("Initialize conv layer weights from the linear layer!")
                C = save_dict["state_dict"]["scratch.input_conv.weight"].shape[1]
                input_conv_weight = save_dict["state_dict"]["scratch.input_conv.weight"].view(384, C, 1, 1).repeat(1, 1, 16, 16) / 256.
                input_conv_bias = save_dict["state_dict"]["scratch.input_conv.bias"]

                self.get_module().scratch.input_conv.weight.data = input_conv_weight   
                self.get_module().scratch.input_conv.bias.data = input_conv_bias

                del save_dict["state_dict"]["scratch.input_conv.weight"]
                del save_dict["state_dict"]["scratch.input_conv.bias"]

        missing, unexpected = self.get_module().load_state_dict(
            save_dict["state_dict"],
            strict=False,
        )
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        if load_metadata:
            self.iteration = save_dict["iteration"]
            self.epoch = save_dict["epoch"]
            time_elapsed = save_dict["elapsed"]
            self.start_time = time.time() - time_elapsed
            if "wandb_id" in save_dict:
                self.wandb_id = save_dict["wandb_id"]
            self.optimizer.load_state_dict(save_dict["optimizer"])
            self.gradscaler.load_state_dict(save_dict["gradscaler"])

    def save_model(self):
        path = os.path.join(self.checkpoint_dir, f"ckpt_{self.iteration:08d}.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        elapsed = time.time() - self.start_time if self.start_time is not None else 0
        save_dict = {
            "epoch": self.epoch,
            "elapsed": elapsed,
            "gradscaler": self.gradscaler.state_dict(),
            "iteration": self.iteration,
            "state_dict": self.get_module().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "wandb_id": self.wandb_id,
        }
        torch.save(save_dict, path)

    def clear_old_checkpoints(self, checkpoint_dir, clear_all_old=False):
        print("Clearing old checkpoints")
        checkpoint_files = sorted(glob(os.path.join(checkpoint_dir, "ckpt_*.pth")))
        if clear_all_old:
            for checkpoint_file in checkpoint_files[:-1]:
                os.remove(checkpoint_file)
        else:
            for checkpoint_file in checkpoint_files:
                checkpoint = os.path.basename(checkpoint_file)
                checkpoint_iteration = int("".join(filter(str.isdigit, checkpoint)))
                if checkpoint_iteration % self.interval_delete_checkpoint != 0:
                    os.remove(checkpoint_file)

    def log_info(
        self,
        loss,
        t0,
        lr,
        scaled_norm,
        unscaled_norm,
        clipped_norm,
    ):
        if self.start_time is None:
            self.start_time = time.time()
        time_elapsed = round(time.time() - self.start_time)
        time_remaining = round(
            (time.time() - self.start_time)
            / (self.iteration + 1)
            * (self.max_iterations - self.iteration)
        )
        disp = [
            f"Iter: {self.iteration}/{self.max_iterations}",
            f"Epoch: {self.epoch}",
            f"Loss: {loss.item():.4f}",
            f"LR: {lr:.7f}",
            f"Grad Norm: {scaled_norm:.4f}/{unscaled_norm:.4f}/{clipped_norm:.4f}",
            f"Elap: {str(datetime.timedelta(seconds=time_elapsed))}",
            f"Rem: {str(datetime.timedelta(seconds=time_remaining))}",
            self.hostname,
            self.name,
        ]
        print(", ".join(disp), flush=True)
        wandb_log = {
            "loss": loss.item(),
            "iter_time": time.time() - t0,
            "lr": lr,
            "iteration": self.iteration,
            "hours_remaining": time_remaining / 3600,
            "gradient norm": scaled_norm,
            "unscaled norm": unscaled_norm,
            "clipped norm": clipped_norm,
        }
        wandb.log(wandb_log)

    def visualize(
        self,
        images,
        cameras_gt,
        crop_parameters=None,
        depths=None,
        distortion_coefficients=None,
        depth_mask=None,
        high_loss=False,
    ):
        self.get_module().eval()

        for camera in cameras_gt:
            # AMP may not cast back to float
            camera.R = camera.R.float()
            camera.T = camera.T.float()

        loss_tag = "" if not high_loss else " HIGH LOSS"

        for i in range(self.num_visualize):
            imgs = view_color_coded_images_from_tensor(images[i].cpu(), depth=False)
            im = wandb.Image(imgs, caption=f"iteration {self.iteration} example {i}")
            wandb.log({f"Vis images {i}{loss_tag}": im})

            if self.cond_depth_mask:
                imgs = view_color_coded_images_from_tensor(
                    depth_mask[i].cpu(), depth=True
                )
                im = wandb.Image(
                    imgs, caption=f"iteration {self.iteration} example {i}"
                )
                wandb.log({f"Vis masks {i}{loss_tag}": im})

        vis_depths, _, _ = create_training_visualizations(
            model=self.get_module(),
            images=images[: self.num_visualize],
            device=self.device,
            cameras_gt=cameras_gt,
            pred_x0=self.pred_x0,
            num_images=images.shape[1],
            crop_parameters=crop_parameters[: self.num_visualize],
            visualize_pred=self.regression,
            return_first=self.regression,
            calculate_intrinsics=self.calculate_intrinsics,
            mode="segment" if self.diffuse_origins_and_endpoints else "plucker",
            depths=depths[: self.num_visualize],
            diffuse_depths=self.diffuse_depths,
            full_num_patches_x=self.full_num_patches_x,
            full_num_patches_y=self.full_num_patches_y,
            use_homogeneous=self.use_homogeneous,
            distortion_coefficients=distortion_coefficients,
        )

        for i, vis_image in enumerate(vis_depths):
            im = wandb.Image(
                vis_image, caption=f"iteration {self.iteration} example {i}"
            )

            for i, vis_image in enumerate(vis_depths):
                im = wandb.Image(
                    vis_image, caption=f"iteration {self.iteration} example {i}"
                )
                wandb.log({f"Vis origins and endpoints {i}{loss_tag}": im})

        self.get_module().train()

    def evaluate_train_acc(self, num_evaluate=10):
        print("Evaluating train accuracy")
        model = self.get_module()
        model.eval()
        additional_timesteps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        num_images = self.num_images

        for split in ["train", "test"]:
            if split == "train":
                if self.dataset_name != "co3d":
                    to_evaluate = self.dataset.datasets
                    names = self.dataset.names
                else:
                    to_evaluate = [self.dataset]
                    names = ["co3d"]
            elif split == "test":
                if self.dataset_name != "co3d":
                    to_evaluate = self.test_dataset.datasets
                    names = self.test_dataset.names
                else:
                    to_evaluate = [self.test_dataset]
                    names = ["co3d"]

            for name, dataset in zip(names, to_evaluate):                    
                results = evaluate(
                    cfg=self.cfg,
                    model=model,
                    dataset=dataset,
                    num_images=num_images,
                    device=self.device,
                    additional_timesteps=additional_timesteps,
                    num_evaluate=num_evaluate,
                    use_pbar=True,
                    mode="segment" if self.diffuse_origins_and_endpoints else "plucker",
                    metrics=False,
                )

                R_err = []
                CC_err = []
                for key in results.keys():
                    R_err.append([v["R_error"] for v in results[key]])
                    CC_err.append([v["CC_error"] for v in results[key]])

                R_err = np.array(R_err)
                CC_err = np.array(CC_err)

                R_acc_15 = np.mean(R_err < 15, (0, 2)).max()
                CC_acc = np.mean(CC_err < 0.1, (0, 2)).max()

                wandb.log(
                    {
                        f"R_acc_15_{name}_{split}": R_acc_15,
                        "iteration": self.iteration,
                    }
                )
                wandb.log(
                    {
                        f"CC_acc_0.1_{name}_{split}": CC_acc,
                        "iteration": self.iteration,
                    }
                )
        model.train()


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
def main(cfg):
    print(cfg)
    torch.autograd.set_detect_anomaly(cfg.debug.anomaly_detection)
    torch.set_float32_matmul_precision(cfg.training.matmul_precision)
    trainer = Trainer(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
