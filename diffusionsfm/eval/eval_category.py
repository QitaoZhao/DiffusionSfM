import os
import json
import torch
import torchvision
import numpy as np
from tqdm.auto import tqdm

from diffusionsfm.dataset.co3d_v2 import (
    Co3dDataset,
    full_scene_scale,
)
from pytorch3d.renderer import PerspectiveCameras
from diffusionsfm.utils.visualization import filter_and_align_point_clouds
from diffusionsfm.inference.load_model import load_model
from diffusionsfm.inference.predict import predict_cameras
from diffusionsfm.utils.geometry import (
    compute_angular_error_batch,
    get_error,
    n_to_np_rotations,
)
from diffusionsfm.utils.slurm import init_slurm_signals_if_slurm
from diffusionsfm.utils.rays import cameras_to_rays
from diffusionsfm.utils.rays import normalize_cameras_batch


@torch.no_grad()
def evaluate(
    cfg,
    model,
    dataset,
    num_images,
    device,
    use_pbar=True,
    calculate_intrinsics=True,
    additional_timesteps=(),
    num_evaluate=None,
    max_num_images=None,
    mode=None,
    metrics=True,
    load_depth=True,
):
    if cfg.training.get("dpt_head", False):
        H_in = W_in = 224
        H_out = W_out = cfg.training.full_num_patches_y
    else:
        H_in = H_out = cfg.model.num_patches_x
        W_in = W_out = cfg.model.num_patches_y

    results = {}
    instances = np.arange(0, len(dataset)) if num_evaluate is None else np.linspace(0, len(dataset) - 1, num_evaluate, endpoint=True, dtype=int)
    instances = tqdm(instances) if use_pbar else instances

    for counter, idx in enumerate(instances):
        batch = dataset[idx]
        instance = batch["model_id"]
        images = batch["image"].to(device)
        focal_length = batch["focal_length"].to(device)[:num_images]
        R = batch["R"].to(device)[:num_images]
        T = batch["T"].to(device)[:num_images]
        crop_parameters = batch["crop_parameters"].to(device)[:num_images]

        if load_depth:
            depths = batch["depth"].to(device)[:num_images]
            depth_masks = batch["depth_masks"].to(device)[:num_images]
            try:
                object_masks = batch["object_masks"].to(device)[:num_images]
            except KeyError:
                object_masks = depth_masks.clone()

            # Normalize cameras and scale depths for output resolution
            cameras_gt = PerspectiveCameras(
                R=R, T=T, focal_length=focal_length, device=device
            )
            cameras_gt, _, _ = normalize_cameras_batch(
                [cameras_gt],
                first_cam_mediod=cfg.training.first_cam_mediod,
                normalize_first_camera=cfg.training.normalize_first_camera,
                depths=depths.unsqueeze(0),
                crop_parameters=crop_parameters.unsqueeze(0),
                num_patches_x=H_in,
                num_patches_y=W_in,
                return_scales=True,
            )
            cameras_gt = cameras_gt[0]

            gt_rays = cameras_to_rays(
                cameras=cameras_gt,
                num_patches_x=H_in,
                num_patches_y=W_in,
                crop_parameters=crop_parameters,
                depths=depths,
                mode=mode,
            )
            gt_points = gt_rays.get_segments().view(num_images, -1, 3)

            resize = torchvision.transforms.Resize(
                224,
                antialias=False,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT,
            )
        else:
            cameras_gt = PerspectiveCameras(
                R=R, T=T, focal_length=focal_length, device=device
            )

        pred_cameras, additional_cams = predict_cameras(
            model,
            images,
            device,
            crop_parameters=crop_parameters,
            num_patches_x=H_out,
            num_patches_y=W_out,
            max_num_images=max_num_images,
            additional_timesteps=additional_timesteps,
            calculate_intrinsics=calculate_intrinsics,
            mode=mode,
            return_rays=True,
            use_homogeneous=cfg.model.get("use_homogeneous", False),
        )
        cameras_to_evaluate = additional_cams + [pred_cameras]

        all_cams_batch = dataset.get_data(
            sequence_name=instance, ids=np.arange(0, batch["n"]), no_images=True
        )
        gt_scene_scale = full_scene_scale(all_cams_batch)
        R_gt = R
        T_gt = T

        errors = []
        for _, (camera, pred_rays) in enumerate(cameras_to_evaluate):
            R_pred = camera.R
            T_pred = camera.T
            f_pred = camera.focal_length

            R_pred_rel = n_to_np_rotations(num_images, R_pred).cpu().numpy()
            R_gt_rel = n_to_np_rotations(num_images, batch["R"]).cpu().numpy()
            R_error = compute_angular_error_batch(R_pred_rel, R_gt_rel)

            CC_error, _ = get_error(True, R_pred, T_pred, R_gt, T_gt, gt_scene_scale)

            if load_depth and metrics:
                # Evaluate outputs at the same resolution as DUSt3R
                pred_points = pred_rays.get_segments().view(num_images, H_out, H_out, 3)
                pred_points = pred_points.permute(0, 3, 1, 2)
                pred_points = resize(pred_points).permute(0, 2, 3, 1).view(num_images, H_in*W_in, 3)

                (
                    _,
                    _,
                    _,
                    _,
                    metric_values,
                ) = filter_and_align_point_clouds(
                    num_images,
                    gt_points,
                    pred_points,
                    depth_masks,
                    depth_masks,
                    images,
                    metrics=metrics,
                    num_patches_x=H_in,
                )

                (
                    _,
                    _,
                    _,
                    _,
                    object_metric_values,
                ) = filter_and_align_point_clouds(
                    num_images,
                    gt_points,
                    pred_points,
                    depth_masks * object_masks,
                    depth_masks * object_masks,
                    images,
                    metrics=metrics,
                    num_patches_x=H_in,
                )

            result = {
                "R_pred": R_pred.detach().cpu().numpy().tolist(),
                "T_pred": T_pred.detach().cpu().numpy().tolist(),
                "f_pred": f_pred.detach().cpu().numpy().tolist(),
                "R_gt": R_gt.detach().cpu().numpy().tolist(),
                "T_gt": T_gt.detach().cpu().numpy().tolist(),
                "f_gt": focal_length.detach().cpu().numpy().tolist(),
                "scene_scale": gt_scene_scale,
                "R_error": R_error.tolist(),
                "CC_error": CC_error,
            }

            if load_depth and metrics:
                result["CD"] = metric_values[1]
                result["CD_Object"] = object_metric_values[1]
            else:
                result["CD"] = 0
                result["CD_Object"] = 0

            errors.append(result)

        results[instance] = errors
        
        if counter == len(dataset) - 1:
            break
    return results


def save_results(
    output_dir,
    checkpoint=800_000,
    category="hydrant",
    num_images=None,
    calculate_additional_timesteps=True,
    calculate_intrinsics=True,
    split="test",
    force=False,
    sample_num=1,
    max_num_images=None,
    dataset="co3d",
):
    init_slurm_signals_if_slurm()
    os.umask(000)  # Default to 777 permissions
    eval_path = os.path.join(
        output_dir,
        f"eval_{dataset}",
        f"{category}_{num_images}_{sample_num}_ckpt{checkpoint}.json",
    )

    if os.path.exists(eval_path) and not force:
        print(f"File {eval_path} already exists. Skipping.")
        return

    if num_images is not None and num_images > 8:
        custom_keys = {"model.num_images": num_images}
        ignore_keys = ["pos_table"]
    else:
        custom_keys = None
        ignore_keys = []

    device = torch.device("cuda")
    model, cfg = load_model(
        output_dir,
        checkpoint=checkpoint,
        device=device,
        custom_keys=custom_keys,
        ignore_keys=ignore_keys,
    )
    if num_images is None:
        num_images = cfg.dataset.num_images

    if cfg.training.dpt_head:
        # Evaluate outputs at the same resolution as DUSt3R
        depth_size = 224
    else:
        depth_size = cfg.model.num_patches_x

    dataset = Co3dDataset(
        category=category,
        split=split,
        num_images=num_images,
        apply_augmentation=False,
        sample_num=None if split == "train" else sample_num,
        use_global_intrinsics=cfg.dataset.use_global_intrinsics,
        load_depths=True,
        center_crop=True,
        depth_size=depth_size,
        mask_holes=not cfg.training.regression,
        img_size=256 if cfg.model.unet_diffuser else 224,
    )
    print(f"Category {category} {len(dataset)}")

    if calculate_additional_timesteps:
        additional_timesteps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    else:
        additional_timesteps = []

    results = evaluate(
        cfg=cfg,
        model=model,
        dataset=dataset,
        num_images=num_images,
        device=device,
        calculate_intrinsics=calculate_intrinsics,
        additional_timesteps=additional_timesteps,
        max_num_images=max_num_images,
        mode="segment",
    )

    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(results, f)