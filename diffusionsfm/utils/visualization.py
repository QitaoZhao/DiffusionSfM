from http.client import MOVED_PERMANENTLY
import io

import ipdb  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import torch
import torchvision
from pytorch3d.loss import chamfer_distance
from scipy.spatial.transform import Rotation

from diffusionsfm.inference.ddim import inference_ddim
from diffusionsfm.utils.rays import (
    Rays,
    cameras_to_rays,
    rays_to_cameras,
    rays_to_cameras_homography,
)
from diffusionsfm.utils.geometry import (
    compute_optimal_alignment,
)

cmap = plt.get_cmap("hsv")


def create_training_visualizations(
    model,
    images,
    device,
    cameras_gt,
    num_images,
    crop_parameters,
    pred_x0=False,
    no_crop_param_device="cpu",
    visualize_pred=False,
    return_first=False,
    calculate_intrinsics=False,
    mode=None,
    depths=None,
    scale_min=-1,
    scale_max=1,
    diffuse_depths=False,
    vis_mode=None,
    average_centers=True,
    full_num_patches_x=16,
    full_num_patches_y=16,
    use_homogeneous=False,
    distortion_coefficients=None,
):

    if model.depth_resolution == 1:
        W_in = W_out = full_num_patches_x
        H_in = H_out = full_num_patches_y
    else:
        W_in = H_in = model.width
        W_out = model.width * model.depth_resolution
        H_out = model.width * model.depth_resolution

    rays_final, rays_intermediate, pred_intermediate, _ = inference_ddim(
        model,
        images,
        device,
        crop_parameters=crop_parameters,
        eta=[1, 0],
        num_patches_x=W_in,
        num_patches_y=H_in,
        visualize=True,
    )

    if vis_mode is None:
        vis_mode = mode

    T = model.noise_scheduler.max_timesteps
    if T == 1000:
        ts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    else:
        ts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

    # Get predicted cameras from rays
    pred_cameras_batched = []
    vis_images = []
    pred_rays = []
    for index in range(len(images)):
        pred_cameras = []
        per_sample_images = []
        for ii in range(num_images):
            rays_gt = cameras_to_rays(
                cameras_gt[index],
                crop_parameters[index],
                no_crop_param_device=no_crop_param_device,
                num_patches_x=W_in,
                num_patches_y=H_in,
                depths=None if depths is None else depths[index],
                mode=mode,
                depth_resolution=model.depth_resolution,
                distortion_coefficients=(
                    None
                    if distortion_coefficients is None
                    else distortion_coefficients[index]
                ),
            )
            image_vis = (images[index, ii].cpu().permute(1, 2, 0).numpy() + 1) / 2

            if diffuse_depths:
                fig, axs = plt.subplots(3, 13, figsize=(15, 4.5), dpi=100)
            else:
                fig, axs = plt.subplots(3, 9, figsize=(12, 4.5), dpi=100)

            for i, t in enumerate(ts):
                r, c = i // 4, i % 4
                if visualize_pred:
                    curr = pred_intermediate[t][index]
                else:
                    curr = rays_intermediate[t][index]
                rays = Rays.from_spatial(
                    curr,
                    mode=mode,
                    num_patches_x=H_in,
                    num_patches_y=W_in,
                    use_homogeneous=use_homogeneous,
                )

                if vis_mode == "segment":
                    vis = (
                        torch.clip(
                            rays.get_segments()[ii], min=scale_min, max=scale_max
                        )
                        - scale_min
                    ) / (scale_max - scale_min)

                else:
                    vis = (
                        torch.nn.functional.normalize(rays.get_moments()[ii], dim=-1)
                        + 1
                    ) / 2

                axs[r, c].imshow(vis.reshape(W_out, H_out, 3).cpu())
                axs[r, c].set_title(f"T={T - t}")

            i += 1
            r, c = i // 4, i % 4

            if vis_mode == "segment":
                vis = (
                    torch.clip(rays_gt.get_segments()[ii], min=scale_min, max=scale_max)
                    - scale_min
                ) / (scale_max - scale_min)
            else:
                vis = (
                    torch.nn.functional.normalize(rays_gt.get_moments()[ii], dim=-1) + 1
                ) / 2

            axs[r, c].imshow(vis.reshape(W_out, H_out, 3).cpu())

            type_str = "Endpoints" if vis_mode == "segment" else "Moments"
            axs[r, c].set_title(f"GT {type_str}")

            for i, t in enumerate(ts):
                r, c = i // 4, i % 4 + 4
                if visualize_pred:
                    curr = pred_intermediate[t][index]
                else:
                    curr = rays_intermediate[t][index]
                rays = Rays.from_spatial(
                    curr,
                    mode,
                    num_patches_x=H_in,
                    num_patches_y=W_in,
                    use_homogeneous=use_homogeneous,
                )

                if vis_mode == "segment":
                    vis = (
                        torch.clip(
                            rays.get_origins(high_res=True)[ii],
                            min=scale_min,
                            max=scale_max,
                        )
                        - scale_min
                    ) / (scale_max - scale_min)
                else:
                    vis = (
                        torch.nn.functional.normalize(rays.get_directions()[ii], dim=-1)
                        + 1
                    ) / 2

                axs[r, c].imshow(vis.reshape(W_out, H_out, 3).cpu())
                axs[r, c].set_title(f"T={T - t}")

            i += 1
            r, c = i // 4, i % 4 + 4

            if vis_mode == "segment":
                vis = (
                    torch.clip(
                        rays_gt.get_origins(high_res=True)[ii],
                        min=scale_min,
                        max=scale_max,
                    )
                    - scale_min
                ) / (scale_max - scale_min)
            else:
                vis = (
                    torch.nn.functional.normalize(rays_gt.get_directions()[ii], dim=-1)
                    + 1
                ) / 2
            axs[r, c].imshow(vis.reshape(W_out, H_out, 3).cpu())
            type_str = "Origins" if vis_mode == "segment" else "Directions"
            axs[r, c].set_title(f"GT {type_str}")

            if diffuse_depths:
                for i, t in enumerate(ts):
                    r, c = i // 4, i % 4 + 8
                    if visualize_pred:
                        curr = pred_intermediate[t][index]
                    else:
                        curr = rays_intermediate[t][index]
                    rays = Rays.from_spatial(
                        curr,
                        mode,
                        num_patches_x=H_in,
                        num_patches_y=W_in,
                        use_homogeneous=use_homogeneous,
                    )

                    vis = rays.depths[ii]
                    if len(rays.depths[ii].shape) < 2:
                        vis = rays.depths[ii].reshape(H_out, W_out)

                    axs[r, c].imshow(vis.cpu())
                    axs[r, c].set_title(f"T={T - t}")

                i += 1
                r, c = i // 4, i % 4 + 8

                vis = depths[index][ii]
                if len(rays.depths[ii].shape) < 2:
                    vis = depths[index][ii].reshape(256, 256)

                axs[r, c].imshow(vis.cpu())
                axs[r, c].set_title(f"GT Depths")

            axs[2, -1].imshow(image_vis)
            axs[2, -1].set_title("Input Image")
            for s in ["bottom", "top", "left", "right"]:
                axs[2, -1].spines[s].set_color(cmap(ii / (num_images)))
                axs[2, -1].spines[s].set_linewidth(5)

            for ax in axs.flatten():
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            img = plot_to_image(fig)
            plt.close()
            per_sample_images.append(img)

            if return_first:
                rays_camera = pred_intermediate[0][index]
            elif pred_x0:
                rays_camera = pred_intermediate[-1][index]
            else:
                rays_camera = rays_final[index]
            rays = Rays.from_spatial(
                rays_camera,
                mode=mode,
                num_patches_x=H_in,
                num_patches_y=W_in,
                use_homogeneous=use_homogeneous,
            )
            if calculate_intrinsics:
                pred_camera = rays_to_cameras_homography(
                    rays=rays[ii, None],
                    crop_parameters=crop_parameters[index],
                    num_patches_x=W_in,
                    num_patches_y=H_in,
                    average_centers=average_centers,
                    depth_resolution=model.depth_resolution,
                )
            else:
                pred_camera = rays_to_cameras(
                    rays=rays[ii, None],
                    crop_parameters=crop_parameters[index],
                    no_crop_param_device=no_crop_param_device,
                    num_patches_x=W_in,
                    num_patches_y=H_in,
                    depth_resolution=model.depth_resolution,
                    average_centers=average_centers,
                )
            pred_cameras.append(pred_camera[0])
            pred_rays.append(rays)

        pred_cameras_batched.append(pred_cameras)
        vis_images.append(np.vstack(per_sample_images))

    return vis_images, pred_cameras_batched, pred_rays


def plot_to_image(figure, dpi=100):
    """Converts matplotlib fig to a png for logging with tf.summary.image."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="raw", dpi=dpi)
    plt.close(figure)
    buffer.seek(0)
    image = np.reshape(
        np.frombuffer(buffer.getvalue(), dtype=np.uint8),
        newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1),
    )
    return image[..., :3]


def view_color_coded_images_from_tensor(images, depth=False):
    num_frames = images.shape[0]
    cmap = plt.get_cmap("hsv")
    num_rows = 3
    num_cols = 3
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows * num_cols):
        if i < num_frames:
            if images[i].shape[0] == 3:
                image = images[i].permute(1, 2, 0)
            else:
                image = images[i].unsqueeze(-1)

            if not depth:
                image = image * 0.5 + 0.5
            else:
                image = image.repeat((1, 1, 3)) / torch.max(image)

            axs[i].imshow(image)
            for s in ["bottom", "top", "left", "right"]:
                axs[i].spines[s].set_color(cmap(i / (num_frames)))
                axs[i].spines[s].set_linewidth(5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i].axis("off")
    plt.tight_layout()
    return fig


def color_and_filter_points(points, images, mask, num_show, resolution):
    # Resize images
    resize = torchvision.transforms.Resize(resolution)
    images = resize(images) * 0.5 + 0.5

    # Reshape points and calculate mask
    points = points.reshape(num_show * resolution * resolution, 3)
    mask = mask.reshape(num_show * resolution * resolution)
    depth_mask = torch.argwhere(mask > 0.5)[:, 0]
    points = points[depth_mask]

    # Mask and reshape colors
    colors = images.permute(0, 2, 3, 1).reshape(num_show * resolution * resolution, 3)
    colors = colors[depth_mask]

    return points, colors


def filter_and_align_point_clouds(
    num_frames,
    gt_points,
    pred_points,
    gt_masks,
    pred_masks,
    images,
    metrics=False,
    num_patches_x=16,
):

    # Filter and color points
    gt_points, gt_colors = color_and_filter_points(
        gt_points, images, gt_masks, num_show=num_frames, resolution=num_patches_x
    )
    pred_points, pred_colors = color_and_filter_points(
        pred_points, images, pred_masks, num_show=num_frames, resolution=num_patches_x
    )

    pred_points, _, _, _ = compute_optimal_alignment(
        gt_points.float(), pred_points.float()
    )

    # Scale PCL so that furthest point from centroid is distance 1
    centroid = torch.mean(gt_points, dim=0)
    dists = torch.norm(gt_points - centroid.unsqueeze(0), dim=-1)
    scale = torch.mean(dists)
    gt_points_scaled = (gt_points - centroid) / scale
    pred_points_scaled = (pred_points - centroid) / scale

    if metrics:

        cd, _ = chamfer_distance(
            pred_points_scaled.unsqueeze(0), gt_points_scaled.unsqueeze(0)
        )
        cd = cd.item()
        mse = torch.mean(
            torch.norm(pred_points_scaled - gt_points_scaled, dim=-1), dim=-1
        ).item()
    else:
        mse, cd = None, None

    return (
        gt_points,
        pred_points,
        gt_colors,
        pred_colors,
        [mse, cd, None],
    )


def add_scene_cam(scene, c2w, edge_color, image=None, focal=None, imsize=None, screen_width=0.03):
    OPENGL = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    if image is not None:
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255*image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if focal is None:
        focal = min(H, W) * 1.1  # default value
    elif isinstance(focal, np.ndarray):
        focal = focal[0]

    # create fake camera
    height = focal * screen_width / H
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W/H
    transform = c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(4)).as_matrix()
    vertices = cam.vertices
    vertices_offset = 0.9 * cam.vertices
    vertices = np.r_[vertices, vertices_offset, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b2, a2))
        faces.append((a2, c, c2))
        faces.append((c2, b2, b))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    for i,face in enumerate(cam.faces):
        if 0 in face:
            continue

        if i == 1 or i == 5:
            a, b, c = face
            faces.append((a, b, c))

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    
    scene.add_geometry(cam)


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d+1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim-2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1]+1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res