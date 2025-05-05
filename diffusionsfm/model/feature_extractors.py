import importlib
import os
import socket
import sys

import ipdb  # noqa: F401
import torch
import torch.nn as nn
from omegaconf import OmegaConf

HOSTNAME = socket.gethostname()

if "trinity" in HOSTNAME:
    # Might be outdated
    config_path = "/home/amylin2/latent-diffusion/configs/autoencoder/autoencoder_kl_16x16x16.yaml"
    weights_path = "/home/amylin2/latent-diffusion/model.ckpt"
elif "grogu" in HOSTNAME:
    # Might be outdated
    config_path = "/home/jasonzh2/code/latent-diffusion/configs/autoencoder/autoencoder_kl_16x16x16.yaml"
    weights_path = "/home/jasonzh2/code/latent-diffusion/model.ckpt"
elif "ender" in HOSTNAME:
    config_path = "/home/jason/ray_diffusion/external/latent-diffusion/configs/autoencoder/autoencoder_kl_16x16x16.yaml"
    weights_path = "/home/jason/ray_diffusion/external/latent-diffusion/model.ckpt"
else:
    config_path = None
    weights_path = None


if weights_path is not None:
    LDM_PATH = os.path.dirname(weights_path)
    if LDM_PATH not in sys.path:
        sys.path.append(LDM_PATH)


def resize(image, size=None, scale_factor=None):
    return nn.functional.interpolate(
        image,
        size=size,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
    )


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class PretrainedVAE(nn.Module):
    def __init__(self, freeze_weights=True, num_patches_x=16, num_patches_y=16):
        super().__init__()
        config = OmegaConf.load(config_path)
        self.model = instantiate_from_config(config.model)
        self.model.init_from_ckpt(weights_path)
        self.model.eval()
        self.feature_dim = 16
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y

        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, autoresize=False):
        """
        Spatial dimensions of output will be H // 16, W // 16. If autoresize is True,
        then the input will be resized such that the output feature map is the correct
        dimensions.

        Args:
            x (torch.Tensor): Images (B, C, H, W). Should be normalized to be [-1, 1].
            autoresize (bool): Whether to resize the input to match the num_patch
                dimensions.

        Returns:
            torch.Tensor: Latent sample (B, 16, h, w)
        """

        *B, c, h, w = x.shape
        x = x.reshape(-1, c, h, w)
        if autoresize:
            new_w = self.num_patches_x * 16
            new_h = self.num_patches_y * 16
            x = resize(x, size=(new_h, new_w))

        decoded, latent = self.model(x)
        # A little ambiguous bc it's all 16, but it is (c, h, w)
        latent_sample = latent.sample().reshape(
            *B, self.feature_dim, self.num_patches_y, self.num_patches_x
        )
        return latent_sample


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


class SpatialDino(nn.Module):
    def __init__(
        self,
        freeze_weights=True,
        model_type="dinov2_vits14",
        num_patches_x=16,
        num_patches_y=16,
        activation_hooks=False,
    ):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_type)
        self.feature_dim = self.model.embed_dim
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

        self.activation_hooks = activation_hooks

        if self.activation_hooks:
            self.model.blocks[5].register_forward_hook(get_activation("encoder1"))
            self.model.blocks[11].register_forward_hook(get_activation("encoder2"))
            self.activations = activations

    def forward(self, x, autoresize=False):
        """
        Spatial dimensions of output will be H // 14, W // 14. If autoresize is True,
        then the output will be resized to the correct dimensions.

        Args:
            x (torch.Tensor): Images (B, C, H, W). Should be ImageNet normalized.
            autoresize (bool): Whether to resize the input to match the num_patch
                dimensions.

        Returns:
            feature_map (torch.tensor): (B, C, h, w)
        """
        *B, c, h, w = x.shape

        x = x.reshape(-1, c, h, w)
        # if autoresize:
        #     new_w = self.num_patches_x * 14
        #     new_h = self.num_patches_y * 14
        #     x = resize(x, size=(new_h, new_w))

        # Output will be (B, H * W, C)
        features = self.model.forward_features(x)["x_norm_patchtokens"]
        features = features.permute(0, 2, 1)
        features = features.reshape(  # (B, C, H, W)
            -1, self.feature_dim, h // 14, w // 14
        )
        if autoresize:
            features = resize(features, size=(self.num_patches_y, self.num_patches_x))

        features = features.reshape(
            *B, self.feature_dim, self.num_patches_y, self.num_patches_x
        )
        return features
