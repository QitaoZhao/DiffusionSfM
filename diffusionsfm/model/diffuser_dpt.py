import ipdb  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

from diffusionsfm.model.dit import DiT
from diffusionsfm.model.feature_extractors import PretrainedVAE, SpatialDino
from diffusionsfm.model.blocks import _make_fusion_block, _make_scratch
from diffusionsfm.model.scheduler import NoiseScheduler

from huggingface_hub import PyTorchModelHubMixin


# functional implementation
def nearest_neighbor_upsample(x: torch.Tensor, scale_factor: int):
    """Upsample {x} (NCHW) by scale factor {scale_factor} using nearest neighbor interpolation."""
    s = scale_factor
    return (
        x.reshape(*x.shape, 1, 1)
        .expand(*x.shape, s, s)
        .transpose(-2, -3)
        .reshape(*x.shape[:2], *(s * hw for hw in x.shape[2:]))
    )


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class RayDiffuserDPT(nn.Module, PyTorchModelHubMixin,
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
        encoder_features=False,
        use_homogeneous=False,
        freeze_transformer=False,
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
        self.encoder_features = encoder_features

        if feature_extractor == "dino":
            self.feature_extractor = SpatialDino(
                freeze_weights=freeze_encoder,
                num_patches_x=width,
                num_patches_y=width,
                activation_hooks=self.encoder_features,
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

            if freeze_transformer:
                for param in self.ray_predictor.parameters():
                    param.requires_grad = False

        # Fusion blocks
        self.f = 256

        if self.encoder_features:
            feature_lens = [
                self.feature_extractor.feature_dim,
                self.feature_extractor.feature_dim,
                self.ray_predictor.hidden_size,
                self.ray_predictor.hidden_size,
            ]
        else:
            feature_lens = [self.ray_predictor.hidden_size] * 4

        self.scratch = _make_scratch(feature_lens, 256, groups=1, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(
            self.f, use_bn=False, use_ln=False, dpt_time=True, resolution=128
        )
        self.scratch.refinenet2 = _make_fusion_block(
            self.f, use_bn=False, use_ln=False, dpt_time=True, resolution=64
        )
        self.scratch.refinenet3 = _make_fusion_block(
            self.f, use_bn=False, use_ln=False, dpt_time=True, resolution=32
        )
        self.scratch.refinenet4 = _make_fusion_block(
            self.f, use_bn=False, use_ln=False, dpt_time=True, resolution=16
        )

        self.scratch.input_conv = nn.Conv2d(
            self.ray_dim + int(self.cond_depth_mask), 
            self.feature_dim, 
            kernel_size=16, 
            stride=16, 
            padding=0
        )

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(self.f, self.f // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.f // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, self.ray_dim, kernel_size=1, stride=1, padding=0),
            nn.Identity(),
        )

        if self.encoder_features:
            self.project_opers = nn.ModuleList([
                ProjectReadout(in_features=self.feature_extractor.feature_dim),
                ProjectReadout(in_features=self.feature_extractor.feature_dim),
            ])

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
            x_noise = zero_out_mask * x_noise

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
        encoder_patches=16,
        depth_mask=None,
        multiview_unconditional=False,
        indices=None,
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

        if unconditional_mask is not None and self.use_unconditional:
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
                rays_noisy, epsilon = self.forward_noise(
                    rays, t, zero_out_mask=depth_mask.unsqueeze(2)
                )
            else:
                rays_noisy, epsilon = self.forward_noise(
                    rays, t
                )
        else:
            epsilon = None

        # DOWNSAMPLE RAYS
        B, N, C, H, W = rays_noisy.shape

        if self.cond_depth_mask:
            if depth_mask is None:
                depth_mask = torch.ones_like(rays_noisy[:, :, 0])
            ray_repr = torch.cat([rays_noisy, depth_mask.unsqueeze(2)], dim=2)
        else:
            ray_repr = rays_noisy

        ray_repr = self.scratch.input_conv(ray_repr.reshape(B * N, -1, H, W))
        _, CP, HP, WP = ray_repr.shape
        ray_repr = ray_repr.reshape(B, N, CP, HP, WP)
        scene_features = torch.cat([features, ray_repr], dim=2)

        if self.append_ndc:
            scene_features = torch.cat([scene_features, ndc_coordinates], dim=2)

        # DIT FORWARD PASS
        activations = self.ray_predictor(
            scene_features,
            t,
            return_dpt_activations=True,
            multiview_unconditional=multiview_unconditional,
        )

        # PROJECT ENCODER ACTIVATIONS & RESHAPE
        if self.encoder_features:
            for i in range(2):
                name = f"encoder{i+1}"

                if indices is not None:
                    act = self.feature_extractor.activations[name][indices]
                else:
                    act = self.feature_extractor.activations[name]

                act = self.project_opers[i](act).permute(0, 2, 1)
                act = act.reshape(
                    (
                        B * N,
                        self.feature_extractor.feature_dim,
                        encoder_patches,
                        encoder_patches,
                    )
                )
                activations[i] = act

        # UPSAMPLE ACTIVATIONS
        for i, act in enumerate(activations):
            k = 3 - i
            activations[i] = nearest_neighbor_upsample(act, 2**k)

        # FUSION BLOCKS
        layer_1_rn = self.scratch.layer1_rn(activations[0])
        layer_2_rn = self.scratch.layer2_rn(activations[1])
        layer_3_rn = self.scratch.layer3_rn(activations[2])
        layer_4_rn = self.scratch.layer4_rn(activations[3])

        # RESHAPE TIMESTEPS
        if t.shape[0] == B:
            t = t.unsqueeze(-1).repeat((1, N)).reshape(B * N)
        elif t.shape[0] == 1 and B > 1:
            t = t.repeat((B * N))
        else:
            assert False

        path_4 = self.scratch.refinenet4(layer_4_rn, t=t)
        path_3 = self.scratch.refinenet3(path_4, activation=layer_3_rn, t=t)
        path_2 = self.scratch.refinenet2(path_3, activation=layer_2_rn, t=t)
        path_1 = self.scratch.refinenet1(path_2, activation=layer_1_rn, t=t)

        epsilon_pred = self.scratch.output_conv(path_1)
        epsilon_pred = epsilon_pred.reshape((B, N, C, H, W))

        return epsilon_pred, epsilon
