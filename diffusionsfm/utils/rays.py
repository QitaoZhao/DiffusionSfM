from tkinter import FALSE
import cv2
import ipdb  # noqa: F401
import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras, RayBundle
from pytorch3d.transforms import Rotate, Translate


from diffusionsfm.utils.normalize import (
    compute_optical_axis_intersection,
    intersect_skew_line_groups,
    first_camera_transform,
    intersect_skew_lines_high_dim,
)
from diffusionsfm.utils.distortion import apply_distortion_tensor


class Rays(object):
    def __init__(
        self,
        rays=None,
        origins=None,
        directions=None,
        moments=None,
        segments=None,
        depths=None,
        moments_rescale=1.0,
        ndc_coordinates=None,
        crop_parameters=None,
        num_patches_x=16,
        num_patches_y=16,
        distortion_coeffs=None,
        camera_coordinate_rays=None,
        mode=None,
        unprojected=None,
        depth_resolution=1,
        row_form=False,
    ):
        """
        Ray class to keep track of current ray representation.

        Args:
            rays: (..., 6).
            origins: (..., 3).
            directions: (..., 3).
            moments: (..., 3).
            mode: One of "ray", "plucker" or "segment".
            moments_rescale: Rescale the moment component of the rays by a scalar.
            ndc_coordinates: (..., 2): NDC coordinates of each ray.
        """
        self.depth_resolution = depth_resolution
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y

        if rays is not None:
            self.rays = rays
            assert mode is not None
            self._mode = mode
        elif segments is not None:
            if not row_form:
                segments = Rays.patches_to_rows(
                    segments,
                    num_patches_x=num_patches_x,
                    num_patches_y=num_patches_y,
                    depth_resolution=depth_resolution,
                )
            self.rays = torch.cat((origins, segments), dim=-1)
            self._mode = "segment"
        elif origins is not None and directions is not None:
            self.rays = torch.cat((origins, directions), dim=-1)
            self._mode = "ray"
        elif directions is not None and moments is not None:
            self.rays = torch.cat((directions, moments), dim=-1)
            self._mode = "plucker"
        else:
            raise Exception("Invalid combination of arguments")

        if depths is not None:
            self._mode = mode
            self.depths = depths
        else:
            self.depths = None
            assert mode is not None

        if unprojected is not None:
            self.unprojected = unprojected
        else:
            self.unprojected = None

        if moments_rescale != 1.0:
            self.rescale_moments(moments_rescale)

        if ndc_coordinates is not None:
            self.ndc_coordinates = ndc_coordinates
        elif crop_parameters is not None:
            # (..., H, W, 2)
            xy_grid = compute_ndc_coordinates(
                crop_parameters,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
                distortion_coeffs=distortion_coeffs,
            )[..., :2]
            xy_grid = xy_grid.reshape(*xy_grid.shape[:-3], -1, 2)
            self.ndc_coordinates = xy_grid
        else:
            self.ndc_coordinates = None

        if camera_coordinate_rays is not None:
            self.camera_ray_coordinates = True
            self.camera_coordinate_ray_directions = camera_coordinate_rays
        else:
            self.camera_ray_coordinates = False

    def __getitem__(self, index):
        cam_coord_rays = None
        if self.camera_ray_coordinates:
            cam_coord_rays = self.camera_coordinate_ray_directions[index]

        return Rays(
            rays=self.rays[index],
            mode=self._mode,
            camera_coordinate_rays=cam_coord_rays,
            ndc_coordinates=(
                self.ndc_coordinates[index]
                if self.ndc_coordinates is not None
                else None
            ),
            num_patches_x=self.num_patches_x,
            num_patches_y=self.num_patches_y,
            depths=(
                self.depths[index]
                if self.ndc_coordinates is not None and self.depths is not None
                else None
            ),
            unprojected=(
                self.unprojected[index] if self.ndc_coordinates is not None else None
            ),
            depth_resolution=self.depth_resolution,
        )

    def __len__(self):
        return self.rays.shape[0]

    def to_spatial(
        self, include_ndc_coordinates=False, include_depths=False, use_homogeneous=False
    ):
        """
        Converts rays to spatial representation: (..., H * W, 6) --> (..., 6, H, W)

        If use_homogeneous is True, then each 3D component will be 4D and normalized.

        Returns:
            torch.Tensor: (..., 6, H, W)
        """
        if self._mode == "ray":
            rays = self.to_plucker().rays
        else:
            rays = self.rays

        *batch_dims, P, D = rays.shape
        H = W = int(np.sqrt(P))
        assert H * W == P

        if use_homogeneous:
            rays_reshaped = rays.reshape(*batch_dims, P, D // 3, 3)
            ones = torch.ones_like(rays_reshaped[..., :1])
            rays_reshaped = torch.cat((rays_reshaped, ones), dim=-1)
            rays = torch.nn.functional.normalize(rays_reshaped, dim=-1)
            D = (4 * D) // 3
            rays = rays.reshape(*batch_dims, P, D)

        rays = torch.transpose(rays, -1, -2)  # (..., 6, H * W)
        rays = rays.reshape(*batch_dims, D, H, W)

        if include_depths:
            depths = self.depths.unsqueeze(1)
            rays = torch.cat((rays, depths), dim=-3)

        if include_ndc_coordinates:
            ndc_coords = self.ndc_coordinates.transpose(-1, -2)  # (..., 2, H * W)
            ndc_coords = ndc_coords.reshape(*batch_dims, 2, H, W)
            rays = torch.cat((rays, ndc_coords), dim=-3)

        return rays

    def to_spatial_with_camera_coordinate_rays(
        self,
        I_camera,
        crop_params,
        moments_rescale=1.0,
        include_ndc_coordinates=False,
        use_homogeneous=False,
    ):
        """
        Converts rays to spatial representation: (..., H * W, 6) --> (..., 6, H, W)

        Returns:
            torch.Tensor: (..., 6, H, W)
        """

        rays = self.to_spatial(
            include_ndc_coordinates=include_ndc_coordinates,
            use_homogeneous=use_homogeneous,
        )
        N, _, H, W = rays.shape

        camera_coord_rays = (
            cameras_to_rays(
                cameras=I_camera,
                num_patches_x=H,
                num_patches_y=W,
                crop_parameters=crop_params,
            )
            .rescale_moments(1 / moments_rescale)
            .get_directions()
        )

        self.camera_coordinate_ray_directions = camera_coord_rays

        # camera_coord_rays = torch.stack(camera_coord_rays)
        camera_coord_rays = torch.transpose(camera_coord_rays, -1, -2)
        camera_coord_rays = camera_coord_rays.reshape(N, 3, H, W)

        rays = torch.cat((camera_coord_rays, rays), dim=-3)

        return rays

    def rescale_moments(self, scale):
        """
        Rescale the moment component of the rays by a scalar. Might be desirable since
        moments may come from a very narrow distribution.

        Note that this modifies in place!
        """
        assert False, "Deprecated"
        if self._mode == "plucker":
            self.rays[..., 3:] *= scale
            return self
        else:
            return self.to_plucker().rescale_moments(scale)

    def to_spatial_with_camera_coordinate_rays_object(
        self,
        I_camera,
        crop_params,
        moments_rescale=1.0,
        include_ndc_coordinates=False,
        use_homogeneous=False,
    ):
        """
        Converts rays to spatial representation: (..., H * W, 6) --> (..., 6, H, W)

        Returns:
            torch.Tensor: (..., 6, H, W)
        """

        rays = self.to_spatial(include_ndc_coordinates, use_homogeneous=use_homogeneous)
        N, _, H, W = rays.shape

        camera_coord_rays = (
            cameras_to_rays(
                cameras=I_camera,
                num_patches_x=H,
                num_patches_y=W,
                crop_parameters=crop_params,
            )
            .rescale_moments(1 / moments_rescale)
            .get_directions()
        )

        self.camera_coordinate_ray_directions = camera_coord_rays

        camera_coord_rays = torch.transpose(camera_coord_rays, -1, -2)
        camera_coord_rays = camera_coord_rays.reshape(N, 3, H, W)

    @classmethod
    def patches_to_rows(cls, x, num_patches_x=16, num_patches_y=16, depth_resolution=1):
        B, P, C = x.shape
        assert P == (depth_resolution**2 * num_patches_x * num_patches_y)

        x = x.reshape(
            B,
            depth_resolution * num_patches_x,
            depth_resolution * num_patches_y,
            C,
        )

        new = x.unfold(1, depth_resolution, depth_resolution).unfold(
            2, depth_resolution, depth_resolution
        )
        new = new.permute((0, 1, 2, 4, 5, 3))
        new = new.reshape(
            (B, num_patches_x * num_patches_y, depth_resolution * depth_resolution * C)
        )
        return new

    @classmethod
    def rows_to_patches(cls, x, num_patches_x=16, num_patches_y=16, depth_resolution=1):
        B, P, CP = x.shape
        assert P == (num_patches_x * num_patches_y)
        C = CP // (depth_resolution**2)
        HP, WP = num_patches_x * depth_resolution, num_patches_y * depth_resolution

        x = x.reshape(
            B, num_patches_x, num_patches_y, depth_resolution, depth_resolution, C
        )
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, HP * WP, C)
        return x

    @classmethod
    def upsample_origins(
        cls, x, num_patches_x=16, num_patches_y=16, depth_resolution=1
    ):
        B, P, C = x.shape
        origins = x.permute((0, 2, 1))
        origins = origins.reshape((B, C, num_patches_x, num_patches_y))
        origins = torch.nn.functional.interpolate(
            origins, scale_factor=(depth_resolution, depth_resolution)
        )
        origins = origins.permute((0, 2, 3, 1)).reshape(
            (B, P * depth_resolution * depth_resolution, C)
        )
        return origins

    @classmethod
    def from_spatial_with_camera_coordinate_rays(
        cls, rays, mode, moments_rescale=1.0, use_homogeneous=False
    ):
        """
        Converts rays from spatial representation: (..., 6, H, W) --> (..., H * W, 6)

        Args:
            rays: (..., 6, H, W)

        Returns:
            Rays: (..., H * W, 6)
        """
        *batch_dims, D, H, W = rays.shape
        rays = rays.reshape(*batch_dims, D, H * W)
        rays = torch.transpose(rays, -1, -2)

        camera_coordinate_ray_directions = rays[..., :3]
        rays = rays[..., 3:]

        return cls(
            rays=rays,
            mode=mode,
            moments_rescale=moments_rescale,
            camera_coordinate_rays=camera_coordinate_ray_directions,
        )

    @classmethod
    def from_spatial(
        cls,
        rays,
        mode,
        moments_rescale=1.0,
        ndc_coordinates=None,
        num_patches_x=16,
        num_patches_y=16,
        use_homogeneous=False,
    ):
        """
        Converts rays from spatial representation: (..., 6, H, W) --> (..., H * W, 6)

        Args:
            rays: (..., 6, H, W)

        Returns:
            Rays: (..., H * W, 6)
        """
        *batch_dims, D, H, W = rays.shape
        rays = rays.reshape(*batch_dims, D, H * W)
        rays = torch.transpose(rays, -1, -2)

        if use_homogeneous:
            D -= 2

        if D == 7:
            if use_homogeneous:
                r1 = rays[..., :3] / (rays[..., 3:4] + 1e-6)
                r2 = rays[..., 4:7] / (rays[..., 7:8] + 1e-6)
                rays = torch.cat((r1, r2), dim=-1)
                depths = rays[8]
            else:
                old_rays = rays
                rays = rays[..., :6]
                depths = old_rays[..., 6]
            return cls(
                rays=rays,
                mode=mode,
                moments_rescale=moments_rescale,
                ndc_coordinates=ndc_coordinates,
                depths=depths.reshape(*batch_dims, H, W),
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
            )
        elif D > 7:

            D += 2
            if use_homogeneous:
                rays_reshaped = rays.reshape((*batch_dims, H * W, D // 4, 4))
                rays_not_homo = rays_reshaped / rays_reshaped[..., :, 3].unsqueeze(-1)
                rays = rays_not_homo[..., :, :3].reshape(
                    (*batch_dims, H * W, (D // 4) * 3)
                )
                D = (D // 4) * 3

            ray = cls(
                origins=rays[:, :, :3],
                segments=rays[:, :, 3:],
                mode="segment",
                moments_rescale=moments_rescale,
                ndc_coordinates=ndc_coordinates,
                # depths=rays[:, :, -1].reshape(*batch_dims, H, W),
                row_form=True,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
                depth_resolution=int(((D - 3) // 3) ** 0.5),
            )

            if mode == "ray":
                return ray.to_point_direction()
            elif mode == "plucker":
                return ray.to_plucker()
            elif mode == "segment":
                return ray
            else:
                assert False
        else:
            if use_homogeneous:
                r1 = rays[..., :3] / (rays[..., 3:4] + 1e-6)
                r2 = rays[..., 4:7] / (rays[..., 7:8] + 1e-6)
                rays = torch.cat((r1, r2), dim=-1)
            return cls(
                rays=rays,
                mode=mode,
                moments_rescale=moments_rescale,
                ndc_coordinates=ndc_coordinates,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
            )

    def to_point_direction(self, normalize_moment=True):
        """
        Convert to point direction representation <O, D>.

        Returns:
            rays: (..., 6).
        """
        if self._mode == "plucker":
            direction = torch.nn.functional.normalize(self.rays[..., :3], dim=-1)
            moment = self.rays[..., 3:]
            if normalize_moment:
                c = torch.linalg.norm(direction, dim=-1, keepdim=True)
                moment = moment / c
            points = torch.cross(direction, moment, dim=-1)
            return Rays(
                rays=torch.cat((points, direction), dim=-1),
                mode="ray",
                ndc_coordinates=self.ndc_coordinates,
                num_patches_x=self.num_patches_x,
                num_patches_y=self.num_patches_y,
                depths=self.depths,
                unprojected=self.unprojected,
                depth_resolution=self.depth_resolution,
            )
        elif self._mode == "segment":
            origins = self.get_origins(high_res=True)

            direction = self.get_segments() - origins
            direction = torch.nn.functional.normalize(direction, dim=-1)

            return Rays(
                rays=torch.cat((origins, direction), dim=-1),
                mode="ray",
                ndc_coordinates=self.ndc_coordinates,
                num_patches_x=self.num_patches_x,
                num_patches_y=self.num_patches_y,
                depths=self.depths,
                unprojected=self.unprojected,
                depth_resolution=self.depth_resolution,
            )
        else:
            return self

    def to_plucker(self):
        """
        Convert to plucker representation <D, OxD>.
        """
        if self._mode == "plucker":
            return self
        elif self._mode == "ray":
            ray = self.rays.clone()
            ray_origins = ray[..., :3]
            ray_directions = ray[..., 3:]

            # Normalize ray directions to unit vectors
            ray_directions = ray_directions / torch.linalg.vector_norm(
                ray_directions, dim=-1, keepdim=True
            )
            plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
            new_ray = torch.cat([ray_directions, plucker_normal], dim=-1)
            return Rays(
                rays=new_ray,
                mode="plucker",
                ndc_coordinates=self.ndc_coordinates,
                num_patches_x=self.num_patches_x,
                num_patches_y=self.num_patches_y,
                depths=self.depths,
                unprojected=self.unprojected,
                depth_resolution=self.depth_resolution,
            )
        elif self._mode == "segment":
            return self.to_point_direction().to_plucker()

    def get_directions(self, normalize=True):
        if self._mode == "plucker":
            directions = self.rays[..., :3]
        elif self._mode == "segment":
            directions = self.to_point_direction().get_directions()
        else:
            directions = self.rays[..., 3:]
        if normalize:
            directions = torch.nn.functional.normalize(directions, dim=-1)
        return directions

    def get_camera_coordinate_rays(self, normalize=True):
        directions = self.camera_coordinate_ray_directions
        if normalize:
            directions = torch.nn.functional.normalize(directions, dim=-1)
        return directions

    def get_origins(self, high_res=False):
        if self._mode == "plucker":
            origins = self.to_point_direction().get_origins(high_res=high_res)
        elif self._mode == "ray":
            origins = self.rays[..., :3]
        elif self._mode == "segment":
            origins = Rays.upsample_origins(
                self.rays[..., :3],
                num_patches_x=self.num_patches_x,
                num_patches_y=self.num_patches_y,
                depth_resolution=self.depth_resolution,
            )
        else:
            assert False

        return origins

    def get_moments(self):
        if self._mode == "plucker":
            moments = self.rays[..., 3:]
        elif self._mode in ["ray", "segment"]:
            moments = self.to_plucker().get_moments()

        return moments

    def get_segments(self):
        assert self._mode == "segment"

        if self.unprojected is not None:
            return self.unprojected
        else:
            return Rays.rows_to_patches(
                self.rays[..., 3:],
                num_patches_x=self.num_patches_x,
                num_patches_y=self.num_patches_y,
                depth_resolution=self.depth_resolution,
            )

    def get_ndc_coordinates(self):
        return self.ndc_coordinates

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode

    @property
    def device(self):
        return self.rays.device

    def __repr__(self, *args, **kwargs):
        ray_str = self.rays.__repr__(*args, **kwargs)[6:]  # remove "tensor"

        if self._mode == "plucker":
            return "PluRay" + ray_str
        elif self._mode == "ray":
            return "DirRay" + ray_str
        else:
            return "SegRay" + ray_str

    def to(self, device):
        self.rays = self.rays.to(device)

    def clone(self):
        return Rays(rays=self.rays.clone(), mode=self._mode)

    @property
    def shape(self):
        return self.rays.shape

    def visualize(self):
        directions = torch.nn.functional.normalize(self.get_directions(), dim=-1).cpu()
        moments = torch.nn.functional.normalize(self.get_moments(), dim=-1).cpu()
        return (directions + 1) / 2, (moments + 1) / 2

    def to_ray_bundle(self, length=0.3, recompute_origin=False, true_length=False):
        """
        Args:
            length (float): Length of the rays for visualization.
            recompute_origin (bool): If True, origin is set to the intersection point of
                all rays. If False, origins are the point along the ray closest
        """
        origins = self.get_origins(high_res=self.depth_resolution > 1)
        lengths = torch.ones_like(origins[..., :2]) * length
        lengths[..., 0] = 0
        p_intersect, p_closest, _, _ = intersect_skew_line_groups(
            origins.float(), self.get_directions().float()
        )
        if recompute_origin:
            centers = p_intersect
            centers = centers.unsqueeze(1).repeat(1, lengths.shape[1], 1)
        else:
            centers = p_closest

        if true_length:
            length = torch.norm(self.get_segments() - centers, dim=-1).unsqueeze(-1)
            lengths = torch.ones_like(origins[..., :2]) * length
            lengths[..., 0] = 0

        return RayBundle(
            origins=centers,
            directions=self.get_directions(),
            lengths=lengths,
            xys=self.get_directions(),
        )


def cameras_to_rays(
    cameras,
    crop_parameters,
    use_half_pix=True,
    use_plucker=True,
    num_patches_x=16,
    num_patches_y=16,
    no_crop_param_device="cpu",
    distortion_coeffs=None,
    depths=None,
    visualize=False,
    mode=None,
    depth_resolution=1,
    nearest_neighbor=True,
    distortion_coefficients=None,
):
    """
    Unprojects rays from camera center to grid on image plane.

    To match Moneish's code, set use_half_pix=False, use_plucker=True. Also, the
    arguments to meshgrid should be swapped (x first, then y). I'm following Pytorch3d
    convention to have y first.

    distortion_coeffs refers to Amy's distortion experiments
    distortion_coefficients refers to the fisheye parameters from colmap

    Args:
        cameras: Pytorch3D cameras to unproject. Can be batched.
        crop_parameters: Crop parameters in NDC (cc_x, cc_y, crop_width, scale).
            Shape is (B, 4).
        use_half_pix: If True, use half pixel offset (Default: True).
        use_plucker: If True, return rays in plucker coordinates (Default: False).
        num_patches_x: Number of patches in x direction (Default: 16).
        num_patches_y: Number of patches in y direction (Default: 16).
    """

    unprojected = []
    unprojected_ones = []
    crop_parameters_list = (
        crop_parameters if crop_parameters is not None else [None for _ in cameras]
    )
    depths_list = depths if depths is not None else [None for _ in cameras]
    if distortion_coeffs is None:
        zs = []
        for i, (camera, crop_param, depth) in enumerate(
            zip(cameras, crop_parameters_list, depths_list)
        ):
            xyd_grid = compute_ndc_coordinates(
                crop_parameters=crop_param,
                use_half_pix=use_half_pix,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
                no_crop_param_device=no_crop_param_device,
                depths=depth,
                return_zs=True,
                depth_resolution=depth_resolution,
                nearest_neighbor=nearest_neighbor,
            )

            xyd_grid, z, ones_grid = xyd_grid
            zs.append(z)

            if (
                distortion_coefficients is not None
                and (distortion_coefficients[i] != 0).any()
            ):
                xyd_grid = undistort_ndc_coordinates(
                    ndc_coordinates=xyd_grid,
                    principal_point=camera.principal_point[0],
                    focal_length=camera.focal_length[0],
                    distortion_coefficients=distortion_coefficients[i],
                )

            unprojected.append(
                camera.unproject_points(
                    xyd_grid.reshape(-1, 3), world_coordinates=True, from_ndc=True
                )
            )

            if depths is not None and mode == "plucker":
                unprojected_ones.append(
                    camera.unproject_points(
                        ones_grid.reshape(-1, 3), world_coordinates=True, from_ndc=True
                    )
                )

    else:
        for camera, crop_param, distort_coeff in zip(
            cameras, crop_parameters_list, distortion_coeffs
        ):
            xyd_grid = compute_ndc_coordinates(
                crop_parameters=crop_param,
                use_half_pix=use_half_pix,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
                no_crop_param_device=no_crop_param_device,
                distortion_coeffs=distort_coeff,
                depths=depths,
                nearest_neighbor=nearest_neighbor,
            )

            unprojected.append(
                camera.unproject_points(
                    xyd_grid.reshape(-1, 3), world_coordinates=True, from_ndc=True
                )
            )

    unprojected = torch.stack(unprojected, dim=0)  # (N, P, 3)
    origins = cameras.get_camera_center().unsqueeze(1)  # (N, 1, 3)
    origins = origins.repeat(1, num_patches_x * num_patches_y, 1)  # (N, P, 3)

    if depths is None:
        directions = unprojected - origins
        rays = Rays(
            origins=origins,
            directions=directions,
            crop_parameters=crop_parameters,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            distortion_coeffs=distortion_coeffs,
            mode="ray",
            unprojected=unprojected,
        )
        if use_plucker:
            return rays.to_plucker()
    elif mode == "segment":
        rays = Rays(
            origins=origins,
            segments=unprojected,
            crop_parameters=crop_parameters,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            distortion_coeffs=distortion_coeffs,
            depths=torch.stack(zs, dim=0),
            mode=mode,
            unprojected=unprojected,
            depth_resolution=depth_resolution,
        )
    elif mode == "plucker" or mode == "ray":
        unprojected_ones = torch.stack(unprojected_ones)
        directions = unprojected_ones - origins

        rays = Rays(
            origins=origins,
            directions=directions,
            crop_parameters=crop_parameters,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            distortion_coeffs=distortion_coeffs,
            depths=torch.stack(zs, dim=0),
            mode="ray",
            unprojected=unprojected,
        )

        if mode == "plucker":
            rays = rays.to_plucker()
    else:
        assert False

    if visualize:
        return rays, unprojected, torch.stack(zs, dim=0)

    return rays


def rays_to_cameras(
    rays,
    crop_parameters,
    num_patches_x=16,
    num_patches_y=16,
    use_half_pix=True,
    no_crop_param_device="cpu",
    sampled_ray_idx=None,
    cameras=None,
    focal_length=(3.453,),
    distortion_coeffs=None,
    calculate_distortion=False,
    depth_resolution=1,
    average_centers=False,
):
    """
    If cameras are provided, will use those intrinsics. Otherwise will use the provided
    focal_length(s). Dataset default is 3.32.

    Args:
        rays (Rays): (N, P, 6)
        crop_parameters (torch.Tensor): (N, 4)
    """

    device = rays.device
    origins = rays.get_origins(high_res=True)
    directions = rays.get_directions()

    if average_centers:
        camera_centers = torch.mean(origins, dim=1)
    else:
        camera_centers, _ = intersect_skew_lines_high_dim(origins, directions)

    # Retrieve target rays
    if cameras is None:
        if len(focal_length) == 1:
            focal_length = focal_length * rays.shape[0]
        I_camera = PerspectiveCameras(focal_length=focal_length, device=device)
    else:
        # Use same intrinsics but reset to identity extrinsics.
        I_camera = cameras.clone()
        I_camera.R[:] = torch.eye(3, device=device)
        I_camera.T[:] = torch.zeros(3, device=device)

    if distortion_coeffs is not None and not calculate_distortion:
        coeff = distortion_coeffs
    else:
        coeff = None

    I_patch_rays = cameras_to_rays(
        cameras=I_camera,
        num_patches_x=num_patches_x * depth_resolution,
        num_patches_y=num_patches_y * depth_resolution,
        use_half_pix=use_half_pix,
        crop_parameters=crop_parameters,
        no_crop_param_device=no_crop_param_device,
        distortion_coeffs=coeff,
        mode="plucker",
        depth_resolution=depth_resolution,
    ).get_directions()

    if sampled_ray_idx is not None:
        I_patch_rays = I_patch_rays[:, sampled_ray_idx]

    # Compute optimal rotation to align rays
    R = torch.zeros_like(I_camera.R)
    for i in range(len(I_camera)):
        R[i] = compute_optimal_rotation_alignment(
            I_patch_rays[i],
            directions[i],
        )

    # Construct and return rotated camera
    cam = I_camera.clone()
    cam.R = R
    cam.T = -torch.matmul(R.transpose(1, 2), camera_centers.unsqueeze(2)).squeeze(2)
    return cam


# https://www.reddit.com/r/learnmath/comments/v1crd7/linear_algebra_qr_to_ql_decomposition/
def ql_decomposition(A):
    P = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=A.device).float()
    A_tilde = torch.matmul(A, P)
    Q_tilde, R_tilde = torch.linalg.qr(A_tilde)
    Q = torch.matmul(Q_tilde, P)
    L = torch.matmul(torch.matmul(P, R_tilde), P)
    d = torch.diag(L)
    Q[:, 0] *= torch.sign(d[0])
    Q[:, 1] *= torch.sign(d[1])
    Q[:, 2] *= torch.sign(d[2])
    L[0] *= torch.sign(d[0])
    L[1] *= torch.sign(d[1])
    L[2] *= torch.sign(d[2])
    return Q, L


def rays_to_cameras_homography(
    rays,
    crop_parameters,
    num_patches_x=16,
    num_patches_y=16,
    use_half_pix=True,
    sampled_ray_idx=None,
    reproj_threshold=0.2,
    camera_coordinate_rays=False,
    average_centers=False,
    depth_resolution=1,
    directions_from_averaged_center=False,
):
    """
    Args:
        rays (Rays): (N, P, 6)
        crop_parameters (torch.Tensor): (N, 4)
    """
    device = rays.device
    origins = rays.get_origins(high_res=True)
    directions = rays.get_directions()

    if average_centers:
        camera_centers = torch.mean(origins, dim=1)
    else:
        camera_centers, _ = intersect_skew_lines_high_dim(origins, directions)

    if directions_from_averaged_center:
        assert rays.mode == "segment"
        directions = rays.get_segments() - camera_centers.unsqueeze(1).repeat(
            (1, num_patches_x * num_patches_y, 1)
        )

    # Retrieve target rays
    I_camera = PerspectiveCameras(focal_length=[1] * rays.shape[0], device=device)
    I_patch_rays = cameras_to_rays(
        cameras=I_camera,
        num_patches_x=num_patches_x * depth_resolution,
        num_patches_y=num_patches_y * depth_resolution,
        use_half_pix=use_half_pix,
        crop_parameters=crop_parameters,
        no_crop_param_device=device,
        mode="plucker",
    ).get_directions()

    if sampled_ray_idx is not None:
        I_patch_rays = I_patch_rays[:, sampled_ray_idx]

    # Compute optimal rotation to align rays
    if camera_coordinate_rays:
        directions_used = rays.get_camera_coordinate_rays()
    else:
        directions_used = directions

    Rs = []
    focal_lengths = []
    principal_points = []
    for i in range(rays.shape[-3]):
        R, f, pp = compute_optimal_rotation_intrinsics(
            I_patch_rays[i],
            directions_used[i],
            reproj_threshold=reproj_threshold,
        )
        Rs.append(R)
        focal_lengths.append(f)
        principal_points.append(pp)

    R = torch.stack(Rs)
    focal_lengths = torch.stack(focal_lengths)
    principal_points = torch.stack(principal_points)
    T = -torch.matmul(R.transpose(1, 2), camera_centers.unsqueeze(2)).squeeze(2)
    return PerspectiveCameras(
        R=R,
        T=T,
        focal_length=focal_lengths,
        principal_point=principal_points,
        device=device,
    )


def compute_optimal_rotation_alignment(A, B):
    """
    Compute optimal R that minimizes: || A - B @ R ||_F

    Args:
        A (torch.Tensor): (N, 3)
        B (torch.Tensor): (N, 3)

    Returns:
        R (torch.tensor): (3, 3)
    """
    # normally with R @ B, this would be A @ B.T
    H = B.T @ A
    U, _, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det(U @ Vh)
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    return U @ S_prime @ Vh


def compute_optimal_rotation_intrinsics(
    rays_origin, rays_target, z_threshold=1e-4, reproj_threshold=0.2
):
    """
    Note: for some reason, f seems to be 1/f.

    Args:
        rays_origin (torch.Tensor): (N, 3)
        rays_target (torch.Tensor): (N, 3)
        z_threshold (float): Threshold for z value to be considered valid.

    Returns:
        R (torch.tensor): (3, 3)
        focal_length (torch.tensor): (2,)
        principal_point (torch.tensor): (2,)
    """
    device = rays_origin.device
    z_mask = torch.logical_and(
        torch.abs(rays_target) > z_threshold, torch.abs(rays_origin) > z_threshold
    )[:, 2]

    rays_target = rays_target[z_mask]
    rays_origin = rays_origin[z_mask]
    rays_origin = rays_origin[:, :2] / rays_origin[:, -1:]
    rays_target = rays_target[:, :2] / rays_target[:, -1:]

    try:
        A, _ = cv2.findHomography(
            rays_origin.cpu().numpy(),
            rays_target.cpu().numpy(),
            cv2.RANSAC,
            reproj_threshold,
        )
    except:
        A, _ = cv2.findHomography(
            rays_origin.cpu().numpy(),
            rays_target.cpu().numpy(),
            cv2.RANSAC,
            reproj_threshold,
        )
    A = torch.from_numpy(A).float().to(device)

    if torch.linalg.det(A) < 0:
        # TODO: Find a better fix for this. This gives the correct R but incorrect
        # intrinsics.
        A = -A

    R, L = ql_decomposition(A)
    L = L / L[2][2]

    f = torch.stack((L[0][0], L[1][1]))
    # f = torch.stack(((L[0][0] + L[1][1]) / 2, (L[0][0] + L[1][1]) / 2))
    pp = torch.stack((L[2][0], L[2][1]))
    return R, f, pp


def compute_ndc_coordinates(
    crop_parameters=None,
    use_half_pix=True,
    num_patches_x=16,
    num_patches_y=16,
    no_crop_param_device="cpu",
    distortion_coeffs=None,
    depths=None,
    return_zs=False,
    depth_resolution=1,
    nearest_neighbor=True,
):
    """
    Computes NDC Grid using crop_parameters. If crop_parameters is not provided,
    then it assumes that the crop is the entire image (corresponding to an NDC grid
    where top left corner is (1, 1) and bottom right corner is (-1, -1)).
    """

    if crop_parameters is None:
        cc_x, cc_y, width = 0, 0, 2
        device = no_crop_param_device
    else:
        if len(crop_parameters.shape) > 1:
            if distortion_coeffs is None:
                return torch.stack(
                    [
                        compute_ndc_coordinates(
                            crop_parameters=crop_param,
                            use_half_pix=use_half_pix,
                            num_patches_x=num_patches_x,
                            num_patches_y=num_patches_y,
                            nearest_neighbor=nearest_neighbor,
                            depths=depths[i] if depths is not None else None,
                        )
                        for i, crop_param in enumerate(crop_parameters)
                    ],
                    dim=0,
                )
            else:
                patch_params = zip(crop_parameters, distortion_coeffs)
                return torch.stack(
                    [
                        compute_ndc_coordinates(
                            crop_parameters=crop_param,
                            use_half_pix=use_half_pix,
                            num_patches_x=num_patches_x,
                            num_patches_y=num_patches_y,
                            distortion_coeffs=distortion_coeff,
                            nearest_neighbor=nearest_neighbor,
                        )
                        for crop_param, distortion_coeff in patch_params
                    ],
                    dim=0,
                )
        device = crop_parameters.device
        cc_x, cc_y, width, _ = crop_parameters

    dx = 1 / num_patches_x
    dy = 1 / num_patches_y
    if use_half_pix:
        min_y = 1 - dy
        max_y = -min_y
        min_x = 1 - dx
        max_x = -min_x
    else:
        min_y = min_x = 1
        max_y = -1 + 2 * dy
        max_x = -1 + 2 * dx

    y, x = torch.meshgrid(
        torch.linspace(min_y, max_y, num_patches_y, dtype=torch.float32, device=device),
        torch.linspace(min_x, max_x, num_patches_x, dtype=torch.float32, device=device),
        indexing="ij",
    )

    x_prime = x * width / 2 - cc_x
    y_prime = y * width / 2 - cc_y

    if distortion_coeffs is not None:
        points = torch.cat(
            (x_prime.flatten().unsqueeze(-1), y_prime.flatten().unsqueeze(-1)),
            dim=-1,
        )
        new_points = apply_distortion_tensor(
            points, distortion_coeffs[0], distortion_coeffs[1]
        )
        x_prime = new_points[:, 0].reshape((num_patches_x, num_patches_y))
        y_prime = new_points[:, 1].reshape((num_patches_x, num_patches_y))

    if depths is not None:
        if depth_resolution > 1:
            high_res_grid = compute_ndc_coordinates(
                crop_parameters=crop_parameters,
                use_half_pix=use_half_pix,
                num_patches_x=num_patches_x * depth_resolution,
                num_patches_y=num_patches_y * depth_resolution,
                no_crop_param_device=no_crop_param_device,
            )
            x_prime = high_res_grid[..., 0]
            y_prime = high_res_grid[..., 1]

        z = depths
        xyd_grid = torch.stack([x_prime, y_prime, z], dim=-1)
    else:
        z = torch.ones_like(x)

    xyd_grid = torch.stack([x_prime, y_prime, z], dim=-1)
    xyd_grid_ones = torch.stack([x_prime, y_prime, torch.ones_like(x_prime)], dim=-1)

    if return_zs:
        return xyd_grid, z, xyd_grid_ones

    return xyd_grid


def undistort_ndc_coordinates(
    ndc_coordinates, principal_point, focal_length, distortion_coefficients
):
    """
    Given NDC coordinates from a fisheye camera, computes where the coordinates would
    have been for a pinhole camera.

    Args:
        ndc_coordinates (torch.Tensor): (H, W, 3)
        principal_point (torch.Tensor): (2,)
        focal_length (torch.Tensor): (2,)
        distortion_coefficients (torch.Tensor): (4,)

    Returns:
        torch.Tensor: (H, W, 3)
    """
    device = ndc_coordinates.device
    x = ndc_coordinates[..., 0]
    y = ndc_coordinates[..., 1]
    d = ndc_coordinates[..., 2]
    # Compute normalized coordinates (using opencv convention where negative is top-left
    x = -(x - principal_point[0]) / focal_length[0]
    y = -(y - principal_point[1]) / focal_length[1]
    distorted = torch.stack((x.flatten(), y.flatten()), 1).unsqueeze(1).cpu().numpy()
    undistorted = cv2.fisheye.undistortPoints(
        distorted, np.eye(3), distortion_coefficients.cpu().numpy(), np.eye(3)
    )
    u = torch.tensor(undistorted[:, 0, 0], device=device)
    v = torch.tensor(undistorted[:, 0, 1], device=device)
    new_x = -u * focal_length[0] + principal_point[0]
    new_y = -v * focal_length[1] + principal_point[1]
    return torch.stack((new_x.reshape(x.shape), new_y.reshape(y.shape), d), -1)


def get_identity_cameras_with_intrinsics(cameras):
    D = len(cameras)
    device = cameras.R.device

    new_cameras = cameras.clone()
    new_cameras.R = torch.eye(3, device=device).unsqueeze(0).repeat((D, 1, 1))
    new_cameras.T = torch.zeros((D, 3), device=device)

    return new_cameras


def normalize_cameras_batch(
    cameras,
    scale=1.0,
    normalize_first_camera=False,
    depths=None,
    crop_parameters=None,
    num_patches_x=16,
    num_patches_y=16,
    distortion_coeffs=[None],
    first_cam_mediod=False,
    return_scales=False,
):
    new_cameras = []
    undo_transforms = []
    scales = []
    for i, cam in enumerate(cameras):
        if normalize_first_camera:
            # Normalize cameras such that first camera is identity and origin is at
            # first camera center.

            s = 1
            if first_cam_mediod:
                s = scale_first_cam_mediod(
                    cam[0],
                    depths=depths[i][0].unsqueeze(0) if depths is not None else None,
                    crop_parameters=crop_parameters[i][0].unsqueeze(0),
                    num_patches_x=num_patches_x,
                    num_patches_y=num_patches_y,
                    distortion_coeffs=(
                        distortion_coeffs[i][0].unsqueeze(0)
                        if distortion_coeffs[i] is not None
                        else None
                    ),
                )
            scales.append(s)

            normalized_cameras = first_camera_transform(cam, s, rotation_only=False)
            undo_transform = None
        else:
            out = normalize_cameras(cam, scale=scale, return_scale=depths is not None)
            normalized_cameras, undo_transform, s = out

        if depths is not None:
            depths[i] *= s

            if depths.isnan().any():
                assert False

        new_cameras.append(normalized_cameras)
        undo_transforms.append(undo_transform)

    if return_scales:
        return new_cameras, undo_transforms, scales

    return new_cameras, undo_transforms


def scale_first_cam_mediod(
    cameras,
    scale=1.0,
    return_scale=False,
    depths=None,
    crop_parameters=None,
    num_patches_x=16,
    num_patches_y=16,
    distortion_coeffs=None,
):
    xy_grid = (
        compute_ndc_coordinates(
            depths=depths,
            crop_parameters=crop_parameters,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            distortion_coeffs=distortion_coeffs,
        )
        .reshape((-1, 3))
        .to(depths.device)
    )
    verts = cameras.unproject_points(xy_grid, from_ndc=True, world_coordinates=True)
    p_intersect = torch.median(
        verts.reshape((-1, 3))[: num_patches_x * num_patches_y].float(), dim=0
    ).values.unsqueeze(0)
    d = torch.norm(p_intersect - cameras.get_camera_center())

    if d < 0.001:
        return 1

    return 1 / d


def normalize_cameras(cameras, scale=1.0, return_scale=False):
    """
    Normalizes cameras such that the optical axes point to the origin, the rotation is
    identity, and the norm of the translation of the first camera is 1.

    Args:
        cameras (pytorch3d.renderer.cameras.CamerasBase).
        scale (float): Norm of the translation of the first camera.

    Returns:
        new_cameras (pytorch3d.renderer.cameras.CamerasBase): Normalized cameras.
        undo_transform (function): Function that undoes the normalization.
    """
    # Let distance from first camera to origin be unit
    new_cameras = cameras.clone()
    new_transform = (
        new_cameras.get_world_to_view_transform()
    )  # potential R is not valid matrix

    p_intersect, dist, _, _, _ = compute_optical_axis_intersection(cameras)

    if p_intersect is None:
        print("Warning: optical axes code has a nan. Returning identity cameras.")
        new_cameras.R[:] = torch.eye(3, device=cameras.R.device, dtype=cameras.R.dtype)
        new_cameras.T[:] = torch.tensor(
            [0, 0, 1], device=cameras.T.device, dtype=cameras.T.dtype
        )
        return new_cameras, lambda x: x, 1 / scale

    d = dist.squeeze(dim=1).squeeze(dim=0)[0]
    # Degenerate case
    if d == 0:
        print(cameras.T)
        print(new_transform.get_matrix()[:, 3, :3])
        assert False
    assert d != 0

    # Can't figure out how to make scale part of the transform too without messing up R.
    # Ideally, we would just wrap it all in a single Pytorch3D transform so that it
    # would work with any structure (eg PointClouds, Meshes).
    tR = Rotate(new_cameras.R[0].unsqueeze(0)).inverse()
    tT = Translate(p_intersect)
    t = tR.compose(tT)

    new_transform = t.compose(new_transform)
    new_cameras.R = new_transform.get_matrix()[:, :3, :3]
    new_cameras.T = new_transform.get_matrix()[:, 3, :3] / d * scale

    def undo_transform(cameras):
        cameras_copy = cameras.clone()
        cameras_copy.T *= d / scale
        new_t = (
            t.inverse().compose(cameras_copy.get_world_to_view_transform()).get_matrix()
        )
        cameras_copy.R = new_t[:, :3, :3]
        cameras_copy.T = new_t[:, 3, :3]
        return cameras_copy

    if return_scale:
        return new_cameras, undo_transform, scale / d

    return new_cameras, undo_transform


def first_camera_transform(cameras, s, rotation_only=True):
    new_cameras = cameras.clone()
    new_transform = new_cameras.get_world_to_view_transform()
    tR = Rotate(new_cameras.R[0].unsqueeze(0))
    if rotation_only:
        t = tR.inverse()
    else:
        tT = Translate(new_cameras.T[0].unsqueeze(0))
        t = tR.compose(tT).inverse()

    new_transform = t.compose(new_transform)
    new_cameras.R = new_transform.get_matrix()[:, :3, :3]
    new_cameras.T = new_transform.get_matrix()[:, 3, :3] * s

    return new_cameras
