import ipdb  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

from diffusionsfm.model.dit import DiT
from diffusionsfm.model.feature_extractors import PretrainedVAE, SpatialDino
from diffusionsfm.model.scheduler import NoiseScheduler

from huggingface_hub import PyTorchModelHubMixin


class RayDiffuser(nn.Module, PyTorchModelHubMixin,
                  repo_url="https://github.com/QitaoZhao/DiffusionSfM",
                  paper_url="https://huggingface.co/papers/2505.05473",
                  pipeline_tag="image-to-3d",
                  license="mit"):
    def __init__(
        self,
        model_type="dit",
        depth=8,
        width=16,
        hidden_size=1152,
        P=1,
        max_num_images=1,
        noise_scheduler=None,
        freeze_encoder=True,
        feature_extractor="dino",
        append_ndc=True,
        use_unconditional=False,
        diffuse_depths=False,
        depth_resolution=1,
        use_homogeneous=False,
        cond_depth_mask=False,
    ):
        super().__init__()
        if noise_scheduler is None:
            self.noise_scheduler = NoiseScheduler()
        else:
            self.noise_scheduler = noise_scheduler

        self.diffuse_depths = diffuse_depths
        self.depth_resolution = depth_resolution
        self.use_homogeneous = use_homogeneous

        self.ray_dim = 3
        if self.use_homogeneous:
            self.ray_dim += 1

        self.ray_dim += self.ray_dim * self.depth_resolution**2

        if self.diffuse_depths:
            self.ray_dim += 1

        self.append_ndc = append_ndc
        self.width = width

        self.max_num_images = max_num_images
        self.model_type = model_type
        self.use_unconditional = use_unconditional
        self.cond_depth_mask = cond_depth_mask

        if feature_extractor == "dino":
            self.feature_extractor = SpatialDino(
                freeze_weights=freeze_encoder, num_patches_x=width, num_patches_y=width
            )
            self.feature_dim = self.feature_extractor.feature_dim
        elif feature_extractor == "vae":
            self.feature_extractor = PretrainedVAE(
                freeze_weights=freeze_encoder, num_patches_x=width, num_patches_y=width
            )
            self.feature_dim = self.feature_extractor.feature_dim
        else:
            raise Exception(f"Unknown feature extractor {feature_extractor}")

        if self.use_unconditional:
            self.register_parameter(
                "null_token", nn.Parameter(torch.randn(self.feature_dim, 1, 1))
            )

        self.input_dim = self.feature_dim * 2

        if self.append_ndc:
            self.input_dim += 2

        if model_type == "dit":
            self.ray_predictor = DiT(
                in_channels=self.input_dim,
                out_channels=self.ray_dim,
                width=width,
                depth=depth,
                hidden_size=hidden_size,
                max_num_images=max_num_images,
                P=P,
            )

        self.scratch = nn.Module()
        self.scratch.input_conv = nn.Linear(self.ray_dim + int(self.cond_depth_mask), self.feature_dim)

    def forward_noise(
        self, x, t, epsilon=None, zero_out_mask=None
    ):
        """
        Applies forward diffusion (adds noise) to the input.

        If a mask is provided, the noise is only applied to the masked inputs.
        """
        t = t.reshape(-1, 1, 1, 1, 1)

        if epsilon is None:
            epsilon = torch.randn_like(x)
        else:
            epsilon = epsilon.reshape(x.shape)

        alpha_bar = self.noise_scheduler.alphas_cumprod[t]
        x_noise = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * epsilon

        if zero_out_mask is not None and self.cond_depth_mask:
            x_noise = x_noise * zero_out_mask

        return x_noise, epsilon

    def forward(
        self,
        features=None,
        images=None,
        rays=None,
        rays_noisy=None,
        t=None,
        ndc_coordinates=None,
        unconditional_mask=None,
        return_dpt_activations=False,
        depth_mask=None,
    ):
        """
        Args:
            images: (B, N, 3, H, W).
            t: (B,).
            rays: (B, N, 6, H, W).
            rays_noisy: (B, N, 6, H, W).
            ndc_coordinates: (B, N, 2, H, W).
            unconditional_mask: (B, N) or (B,). Should be 1 for unconditional samples
                and 0 else.
        """

        if features is None:
            # VAE expects 256x256 images while DINO expects 224x224 images.
            # Both feature extractors support autoresize=True, but ideally we should
            # set this to be false and handle in the dataloader.
            features = self.feature_extractor(images, autoresize=True)

        B = features.shape[0]

        if (
            unconditional_mask is not None
            and self.use_unconditional
        ):
            null_token = self.null_token.reshape(1, 1, self.feature_dim, 1, 1)
            unconditional_mask = unconditional_mask.reshape(B, -1, 1, 1, 1)
            features = (
                features * (1 - unconditional_mask) + null_token * unconditional_mask
            )

        if isinstance(t, int) or isinstance(t, np.int64):
            t = torch.ones(1, dtype=int).to(features.device) * t
        else:
            t = t.reshape(B)

        if rays_noisy is None:
            if self.cond_depth_mask:
                rays_noisy, epsilon = self.forward_noise(rays, t, zero_out_mask=depth_mask.unsqueeze(2))
            else:
                rays_noisy, epsilon = self.forward_noise(rays, t)
        else:
            epsilon = None

        if self.cond_depth_mask:
            if depth_mask is None:
                depth_mask = torch.ones_like(rays_noisy[:, :, 0])
            ray_repr = torch.cat([rays_noisy, depth_mask.unsqueeze(2)], dim=2)
        else:
            ray_repr = rays_noisy

        ray_repr = ray_repr.permute(0, 1, 3, 4, 2)
        ray_repr = self.scratch.input_conv(ray_repr).permute(0, 1, 4, 2, 3).contiguous()

        scene_features = torch.cat([features, ray_repr], dim=2)

        if self.append_ndc:
            scene_features = torch.cat([scene_features, ndc_coordinates], dim=2)

        epsilon_pred = self.ray_predictor(
            scene_features,
            t,
            return_dpt_activations=return_dpt_activations,
        )

        if return_dpt_activations:
            return epsilon_pred, rays_noisy, epsilon

        return epsilon_pred, epsilon
