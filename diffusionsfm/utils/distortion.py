import cv2
import ipdb
import numpy as np
from PIL import Image
import torch


# https://gist.github.com/davegreenwood/820d51ac5ec88a2aeda28d3079e7d9eb
def apply_distortion(pts, k1, k2):
    """
    Arguments:
        pts (N x 2): numpy array in NDC coordinates
        k1, k2 distortion coefficients
    Return:
        pts (N x 2): distorted points in NDC coordinates
    """
    r2 = np.square(pts).sum(-1)
    f = 1 + k1 * r2 + k2 * r2**2
    return f[..., None] * pts


# https://gist.github.com/davegreenwood/820d51ac5ec88a2aeda28d3079e7d9eb
def apply_distortion_tensor(pts, k1, k2):
    """
    Arguments:
        pts (N x 2): numpy array in NDC coordinates
        k1, k2 distortion coefficients
    Return:
        pts (N x 2): distorted points in NDC coordinates
    """
    r2 = torch.square(pts).sum(-1)
    f = 1 + k1 * r2 + k2 * r2**2
    return f[..., None] * pts


# https://gist.github.com/davegreenwood/820d51ac5ec88a2aeda28d3079e7d9eb
def remove_distortion_iter(points, k1, k2):
    """
    Arguments:
        pts (N x 2): numpy array in NDC coordinates
        k1, k2 distortion coefficients
    Return:
        pts (N x 2): distorted points in NDC coordinates
    """
    pts = ptsd = points
    for _ in range(5):
        r2 = np.square(pts).sum(-1)
        f = 1 + k1 * r2 + k2 * r2**2
        pts = ptsd / f[..., None]

    return pts


def make_square(im, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new("RGB", (size, size), fill_color)
    corner = (int((size - x) / 2), int((size - y) / 2))
    new_im.paste(im, corner)
    return new_im, corner


def pixel_to_ndc(coords, image_size):
    """
    Converts pixel coordinates to normalized device coordinates (Pytorch3D convention
    with upper left = (1, 1)) for a square image.

    Args:
        coords: Pixel coordinates UL=(0, 0), LR=(image_size, image_size).
        image_size (int): Image size.

    Returns:
        NDC coordinates UL=(1, 1) LR=(-1, -1).
    """
    coords = np.array(coords)
    return 1 - coords / image_size * 2


def ndc_to_pixel(coords, image_size):
    """
    Converts normalized device coordinates to pixel coordinates for a square image.
    """
    num_points = coords.shape[0]
    sizes = np.tile(np.array(image_size, dtype=np.float32)[None, ...], (num_points, 1))

    coords = np.array(coords, dtype=np.float32)
    return (1 - coords) * sizes / 2


def distort_image(image, bbox, k1, k2, modify_bbox=False):
    # We want to operate in -1 to 1 space using the padded square of the original image
    image, corner = make_square(image)
    bbox[:2] += np.array(corner)
    bbox[2:] += np.array(corner)

    # Construct grid points
    x = np.linspace(1, -1, image.width, dtype=np.float32)
    y = np.linspace(1, -1, image.height, dtype=np.float32)
    x, y = np.meshgrid(x, y, indexing="xy")
    xy_grid = np.stack((x, y), axis=-1)
    points = xy_grid.reshape((image.height * image.width, 2))
    new_points = ndc_to_pixel(apply_distortion(points, k1, k2), image.size)

    # Distort image by remapping
    map_x = new_points[:, 0].reshape((image.height, image.width))
    map_y = new_points[:, 1].reshape((image.height, image.width))
    distorted = cv2.remap(
        np.asarray(image),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
    )
    distorted = Image.fromarray(distorted)

    # Find distorted crop bounds - inverse process of above
    if modify_bbox:
        center = (bbox[:2] + bbox[2:]) / 2
        top, bottom = (bbox[0], center[1]), (bbox[2], center[1])
        left, right = (center[0], bbox[1]), (center[0], bbox[3])
        bbox_points = np.array(
            [
                pixel_to_ndc(top, image.size),
                pixel_to_ndc(left, image.size),
                pixel_to_ndc(bottom, image.size),
                pixel_to_ndc(right, image.size),
            ],
            dtype=np.float32,
        )
    else:
        bbox_points = np.array(
            [pixel_to_ndc(bbox[:2], image.size), pixel_to_ndc(bbox[2:], image.size)],
            dtype=np.float32,
        )

    # Inverse mapping
    distorted_bbox = remove_distortion_iter(bbox_points, k1, k2)

    if modify_bbox:
        p = ndc_to_pixel(distorted_bbox, image.size)
        distorted_bbox = np.array([p[0][0], p[1][1], p[2][0], p[3][1]])
    else:
        distorted_bbox = ndc_to_pixel(distorted_bbox, image.size).reshape(4)

    return distorted, distorted_bbox
