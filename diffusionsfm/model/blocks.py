import torch
import torch.nn as nn
from diffusionsfm.model.dit import TimestepEmbedder
import ipdb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(
        -1
    )


def _make_fusion_block(features, use_bn, use_ln, dpt_time, resolution):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        dpt_time=dpt_time,
        ln=use_ln,
        resolution=resolution
    )


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, ln, dpt_time=False, resolution=16):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.ln = ln

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        if self.ln == True:
            self.bn1 = nn.LayerNorm((features, resolution, resolution))
            self.bn2 = nn.LayerNorm((features, resolution, resolution))

        self.activation = activation

        if dpt_time:
            self.t_embedder = TimestepEmbedder(hidden_size=features)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(features, 3 * features, bias=True)
            )

    def forward(self, x, t=None):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        if t is not None:
            # Embed timestamp & calculate shift parameters
            t = self.t_embedder(t)  # (B*N)
            shift, scale, gate = self.adaLN_modulation(t).chunk(3, dim=1)  # (B * N, T)

            # Shift & scale x
            x = modulate(x, shift, scale)  # (B * N, T, H, W)

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn or self.ln:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn or self.ln:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        if t is not None:
            out = gate.unsqueeze(-1).unsqueeze(-1) * out

        return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        ln=False,
        expand=False,
        align_corners=True,
        dpt_time=False,
        resolution=16,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        nn.init.kaiming_uniform_(self.out_conv.weight)

        # The second block sees time
        self.resConfUnit1 = ResidualConvUnit_custom(
            features, activation, bn=bn, ln=ln, dpt_time=False, resolution=resolution
        )
        self.resConfUnit2 = ResidualConvUnit_custom(
            features, activation, bn=bn, ln=ln, dpt_time=dpt_time, resolution=resolution
        )

    def forward(self, input, activation=None, t=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = input

        if activation is not None:
            res = self.resConfUnit1(activation)

            output += res

        output = self.resConfUnit2(output, t)

        output = torch.nn.functional.interpolate(
            output.float(),
            scale_factor=2,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        output = self.out_conv(output)

        return output
