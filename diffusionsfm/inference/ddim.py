import torch
import random
import numpy as np
from tqdm.auto import tqdm

from diffusionsfm.utils.rays import compute_ndc_coordinates


def inference_ddim(
    model,
    images,
    device,
    crop_parameters=None,
    eta=0,
    num_inference_steps=100,
    pbar=True,
    stop_iteration=None,
    num_patches_x=16,
    num_patches_y=16,
    visualize=False,
    max_num_images=8,
    seed=0,
):
    """
    Implements DDIM-style inference.

    To get multiple samples, batch the images multiple times.

    Args:
        model: Ray Diffuser.
        images (torch.Tensor): (B, N, C, H, W).
        patch_rays_gt (torch.Tensor): If provided, the patch rays which are ground
            truth (B, N, P, 6).
        eta (float, optional): Stochasticity coefficient. 0 is completely deterministic,
            1 is equivalent to DDPM. (Default: 0)
        num_inference_steps (int, optional): Number of inference steps. (Default: 100)
        pbar (bool, optional): Whether to show progress bar. (Default: True)
    """
    timesteps = model.noise_scheduler.compute_inference_timesteps(num_inference_steps)
    batch_size = images.shape[0]
    num_images = images.shape[1]

    if isinstance(eta, list):
        eta_0, eta_1 = float(eta[0]), float(eta[1])
    else:
        eta_0, eta_1 = 0, 0

    # Fixing seed
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    with torch.no_grad():
        x_tau = torch.randn(
            batch_size,
            num_images,
            model.ray_out if hasattr(model, "ray_out") else model.ray_dim,
            num_patches_x,
            num_patches_y,
            device=device,
        )

        if visualize:
            x_taus = [x_tau]
            all_pred = []
            noise_samples = []

        image_features = model.feature_extractor(images, autoresize=True)

        if model.append_ndc:
            ndc_coordinates = compute_ndc_coordinates(
                crop_parameters=crop_parameters,
                no_crop_param_device="cpu",
                num_patches_x=model.width,
                num_patches_y=model.width,
                distortion_coeffs=None,
            )[..., :2].to(device)
            ndc_coordinates = ndc_coordinates.permute(0, 1, 4, 2, 3)
        else:
            ndc_coordinates = None

        if stop_iteration is None:
            loop = range(len(timesteps))
        else:
            loop = range(len(timesteps) - stop_iteration + 1)
        loop = tqdm(loop) if pbar else loop

        for t in loop:
            tau = timesteps[t]

            if tau > 0 and eta_1 > 0:
                z = torch.randn(
                    batch_size,
                    num_images,
                    model.ray_out if hasattr(model, "ray_out") else model.ray_dim,
                    num_patches_x,
                    num_patches_y,
                    device=device,
                )
            else:
                z = 0

            alpha = model.noise_scheduler.alphas_cumprod[tau]
            if tau > 0:
                tau_prev = timesteps[t + 1]
                alpha_prev = model.noise_scheduler.alphas_cumprod[tau_prev]
            else:
                alpha_prev = torch.tensor(1.0, device=device).float()

            sigma_t = (
                torch.sqrt((1 - alpha_prev) / (1 - alpha))
                * torch.sqrt(1 - alpha / alpha_prev)
            )

            if num_images > max_num_images:
                eps_pred = torch.zeros_like(x_tau)
                noise_sample = torch.zeros_like(x_tau)

                # Randomly split image indices (excluding index 0), then prepend 0 to each split
                indices_split = torch.split(
                    torch.randperm(num_images - 1) + 1, max_num_images - 1
                )

                for indices in indices_split:
                    indices = torch.cat((torch.tensor([0]), indices))  # Ensure index 0 is always included

                    eps_pred_ind, noise_sample_ind = model(
                        features=image_features[:, indices],
                        rays_noisy=x_tau[:, indices],
                        t=int(tau),
                        ndc_coordinates=ndc_coordinates[:, indices],
                        indices=indices,
                    )

                    eps_pred[:, indices] += eps_pred_ind

                    if noise_sample_ind is not None:
                        noise_sample[:, indices] += noise_sample_ind

                # Average over splits for the shared reference index (0)
                eps_pred[:, 0] /= len(indices_split)
                noise_sample[:, 0] /= len(indices_split)
            else:
                eps_pred, noise_sample = model(
                    features=image_features,
                    rays_noisy=x_tau,
                    t=int(tau),
                    ndc_coordinates=ndc_coordinates,
                )
                
            if model.use_homogeneous:
                p1 = eps_pred[:, :, :4]
                p2 = eps_pred[:, :, 4:]

                c1 = torch.linalg.norm(p1, dim=2, keepdim=True)
                c2 = torch.linalg.norm(p2, dim=2, keepdim=True)
                eps_pred[:, :, :4] = p1 / c1
                eps_pred[:, :, 4:] = p2 / c2

            if visualize:
                all_pred.append(eps_pred.clone())
                noise_samples.append(noise_sample)
                
            # TODO: Can simplify this a lot
            x0_pred = eps_pred.clone()
            eps_pred = (x_tau - torch.sqrt(alpha) * eps_pred) / torch.sqrt(
                1 - alpha
            )

            dir_x_tau = torch.sqrt(1 - alpha_prev - eta_0*sigma_t**2) * eps_pred
            noise = eta_1 * sigma_t * z

            new_x_tau = torch.sqrt(alpha_prev) * x0_pred + dir_x_tau + noise
            x_tau = new_x_tau

            if visualize:
                x_taus.append(x_tau.detach().clone())
    if visualize:
        return x_tau, x_taus, all_pred, noise_samples
    return x_tau
