from diffusionsfm.inference.ddim import inference_ddim
from diffusionsfm.utils.rays import (
    Rays,
    rays_to_cameras,
    rays_to_cameras_homography,
)


def predict_cameras(
    model,
    images,
    device,
    crop_parameters=None,
    stop_iteration=None,
    num_patches_x=16,
    num_patches_y=16,
    additional_timesteps=(),
    calculate_intrinsics=False,
    max_num_images=8,
    mode=None,
    return_rays=False,
    use_homogeneous=False,
    seed=0,
):
    """
    Args:
        images (torch.Tensor): (N, C, H, W)
        crop_parameters (torch.Tensor): (N, 4) or None
    """
    if calculate_intrinsics:
        ray_to_cam = rays_to_cameras_homography
    else:
        ray_to_cam = rays_to_cameras

    get_spatial_rays = Rays.from_spatial

    rays_final, rays_intermediate, pred_intermediate, _ = inference_ddim(
        model,
        images.unsqueeze(0),
        device,
        crop_parameters=crop_parameters.unsqueeze(0),
        pbar=False,
        stop_iteration=stop_iteration,
        eta=[1, 0],
        num_inference_steps=100,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        visualize=True,
        max_num_images=max_num_images,
    )

    spatial_rays = get_spatial_rays(
        rays_final[0],
        mode=mode,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        use_homogeneous=use_homogeneous,
    )

    pred_cam = ray_to_cam(
        spatial_rays,
        crop_parameters,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        depth_resolution=model.depth_resolution,
        average_centers=True,
        directions_from_averaged_center=True,
    )

    additional_predictions = []
    for t in additional_timesteps:
        ray = pred_intermediate[t]

        ray = get_spatial_rays(
            ray[0],
            mode=mode,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            use_homogeneous=use_homogeneous,
        )

        cam = ray_to_cam(
            ray,
            crop_parameters,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            average_centers=True,
            directions_from_averaged_center=True,
        )
        if return_rays:
            cam = (cam, ray)
        additional_predictions.append(cam)

    if return_rays:
        return (pred_cam, spatial_rays), additional_predictions
    return pred_cam, additional_predictions, spatial_rays