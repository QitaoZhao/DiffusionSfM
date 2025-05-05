import gzip
import json
import os.path as osp
import random
import socket
import time
import torch
import warnings

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy import ndimage as nd

from diffusionsfm.utils.distortion import distort_image


HOSTNAME = socket.gethostname()

CO3D_DIR = "../co3d_data"  # update this
CO3D_ANNOTATION_DIR = osp.join(CO3D_DIR, "co3d_annotations")
CO3D_DIR = CO3D_DEPTH_DIR = osp.join(CO3D_DIR, "co3d")
order_path = osp.join(
    CO3D_DIR, "co3d_v2_random_order_{sample_num}/{category}.json"
)


TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = [
    "ball",
    "book",
    "couch",
    "frisbee",
    "hotdog",
    "kite",
    "remote",
    "sandwich",
    "skateboard",
    "suitcase",
]

assert len(TRAINING_CATEGORIES) + len(TEST_CATEGORIES) == 51

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def fill_depths(data, invalid=None):
    data_list = []
    for i in range(data.shape[0]):
        data_item = data[i].numpy()
        # Invalid must be 1 where stuff is invalid, 0 where valid
        ind = nd.distance_transform_edt(
            invalid[i], return_distances=False, return_indices=True
        )
        data_list.append(torch.tensor(data_item[tuple(ind)]))
    return torch.stack(data_list, dim=0)


def full_scene_scale(batch):
    cameras = PerspectiveCameras(R=batch["R"], T=batch["T"], device="cuda")
    cc = cameras.get_camera_center()
    centroid = torch.mean(cc, dim=0)

    diffs = cc - centroid
    norms = torch.linalg.norm(diffs, dim=1)

    furthest_index = torch.argmax(norms).item()
    scale = norms[furthest_index].item()
    return scale


def square_bbox(bbox, padding=0.0, astype=None, tight=False):
    """
    Computes a square bounding box, with optional padding parameters.
    Args:
        bbox: Bounding box in xyxy format (4,).
    Returns:
        square_bbox in xyxy format (4,).
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2

    # No black bars if tight
    if tight:
        s = min(extents) * (1 + padding)
    else:
        s = max(extents) * (1 + padding)

    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=astype,
    )
    return square_bbox


def unnormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    return (image * 255.0).astype(np.uint8)


def unnormalize_image_for_vis(image):
    assert len(image.shape) == 5 and image.shape[2] == 3
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(image.device)
    image = image * std + mean
    image = (image - 0.5) / 0.5
    return image


def unnormalize_image_for_inference(image):
    assert len(image.shape) == 5 and image.shape[2] == 3
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(image.device)
    image = image * 0.5 + 0.5
    image = (image - mean) / std
    return image


def _transform_intrinsic(image, bbox, principal_point, focal_length):
    # Rescale intrinsics to match bbox
    half_box = np.array([image.width, image.height]).astype(np.float32) / 2
    org_scale = min(half_box).astype(np.float32)

    # Pixel coordinates
    principal_point_px = half_box - (np.array(principal_point) * org_scale)
    focal_length_px = np.array(focal_length) * org_scale
    principal_point_px -= bbox[:2]
    new_bbox = (bbox[2:] - bbox[:2]) / 2
    new_scale = min(new_bbox)

    # NDC coordinates
    new_principal_ndc = (new_bbox - principal_point_px) / new_scale
    new_focal_ndc = focal_length_px / new_scale

    principal_point = torch.tensor(new_principal_ndc.astype(np.float32))
    focal_length = torch.tensor(new_focal_ndc.astype(np.float32))

    return principal_point, focal_length


def construct_camera_from_batch(batch, device):
    if isinstance(device, int):
        device = f"cuda:{device}"

    return PerspectiveCameras(
        R=batch["R"].reshape(-1, 3, 3),
        T=batch["T"].reshape(-1, 3),
        focal_length=batch["focal_lengths"].reshape(-1, 2),
        principal_point=batch["principal_points"].reshape(-1, 2),
        image_size=batch["image_sizes"].reshape(-1, 2),
        device=device,
    )


def save_batch_images(images, fname):
    cmap = plt.get_cmap("hsv")
    num_frames = len(images)
    num_rows = len(images)
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows):
        for j in range(4):
            if i < num_frames:
                axs[i * 4 + j].imshow(unnormalize_image(images[i][j]))
                for s in ["bottom", "top", "left", "right"]:
                    axs[i * 4 + j].spines[s].set_color(cmap(i / (num_frames)))
                    axs[i * 4 + j].spines[s].set_linewidth(5)
                axs[i * 4 + j].set_xticks([])
                axs[i * 4 + j].set_yticks([])
            else:
                axs[i * 4 + j].axis("off")
    plt.tight_layout()
    plt.savefig(fname)


def jitter_bbox(
    square_bbox,
    jitter_scale=(1.1, 1.2),
    jitter_trans=(-0.07, 0.07),
    direction_from_size=None,
):

    square_bbox = np.array(square_bbox.astype(float))
    s = np.random.uniform(jitter_scale[0], jitter_scale[1])

    # Jitter only one dimension if center cropping
    tx, ty = np.random.uniform(jitter_trans[0], jitter_trans[1], size=2)
    if direction_from_size is not None:
        if direction_from_size[0] > direction_from_size[1]:
            tx = 0
        else:
            ty = 0

    side_length = square_bbox[2] - square_bbox[0]
    center = (square_bbox[:2] + square_bbox[2:]) / 2 + np.array([tx, ty]) * side_length
    extent = side_length / 2 * s
    ul = center - extent
    lr = ul + 2 * extent
    return np.concatenate((ul, lr))


class Co3dDataset(Dataset):
    def __init__(
        self,
        category=("all_train",),
        split="train",
        transform=None,
        num_images=2,
        img_size=224,
        mask_images=False,
        crop_images=True,
        co3d_dir=None,
        co3d_annotation_dir=None,
        precropped_images=False,
        apply_augmentation=True,
        normalize_cameras=True,
        no_images=False,
        sample_num=None,
        seed=0,
        load_extra_cameras=False,
        distort_image=False,
        load_depths=False,
        center_crop=False,
        depth_size=256,
        mask_holes=False,
        object_mask=True,
    ):
        """
        Args:
            num_images: Number of images in each batch.
            perspective_correction (str):
                "none": No perspective correction.
                "warp": Warp the image and label.
                "label_only": Correct the label only.
        """
        start_time = time.time()

        self.category = category
        self.split = split
        self.transform = transform
        self.num_images = num_images
        self.img_size = img_size
        self.mask_images = mask_images
        self.crop_images = crop_images
        self.precropped_images = precropped_images
        self.apply_augmentation = apply_augmentation
        self.normalize_cameras = normalize_cameras
        self.no_images = no_images
        self.sample_num = sample_num
        self.load_extra_cameras = load_extra_cameras
        self.distort = distort_image
        self.load_depths = load_depths
        self.center_crop = center_crop
        self.depth_size = depth_size
        self.mask_holes = mask_holes
        self.object_mask = object_mask

        if self.apply_augmentation:
            if self.center_crop:
                self.jitter_scale = (0.8, 1.1)
                self.jitter_trans = (0.0, 0.0)
            else:
                self.jitter_scale = (1.1, 1.2)
                self.jitter_trans = (-0.07, 0.07)
        else:
            # Note if trained with apply_augmentation, we should still use
            # apply_augmentation at test time.
            self.jitter_scale = (1, 1)
            self.jitter_trans = (0.0, 0.0)

        if self.distort:
            self.k1_max = 1.0
            self.k2_max = 1.0

        if co3d_dir is not None:
            self.co3d_dir = co3d_dir
            self.co3d_annotation_dir = co3d_annotation_dir
        else:
            self.co3d_dir = CO3D_DIR
            self.co3d_annotation_dir = CO3D_ANNOTATION_DIR
            self.co3d_depth_dir = CO3D_DEPTH_DIR

        if isinstance(self.category, str):
            self.category = [self.category]

        if "all_train" in self.category:
            self.category = TRAINING_CATEGORIES
        if "all_test" in self.category:
            self.category = TEST_CATEGORIES
        if "full" in self.category:
            self.category = TRAINING_CATEGORIES + TEST_CATEGORIES
        self.category = sorted(self.category)
        self.is_single_category = len(self.category) == 1

        # Fixing seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        print(f"Co3d ({split}):")

        self.low_quality_translations = [
            "411_55952_107659",
            "427_59915_115716",
            "435_61970_121848",
            "112_13265_22828",
            "110_13069_25642",
            "165_18080_34378",
            "368_39891_78502",
            "391_47029_93665",
            "20_695_1450",
            "135_15556_31096",
            "417_57572_110680",
        ]  # Initialized with sequences with poor depth masks
        self.rotations = {}
        self.category_map = {}
        for c in tqdm(self.category):
            annotation_file = osp.join(
                self.co3d_annotation_dir, f"{c}_{self.split}.jgz"
            )
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())

            counter = 0
            for seq_name, seq_data in annotation.items():
                counter += 1
                if len(seq_data) < self.num_images:
                    continue

                filtered_data = []
                self.category_map[seq_name] = c
                bad_seq = False
                for data in seq_data:
                    # Make sure translations are not ridiculous and rotations are valid
                    det = np.linalg.det(data["R"])
                    if (np.abs(data["T"]) > 1e5).any() or det < 0.99 or det > 1.01:
                        bad_seq = True
                        self.low_quality_translations.append(seq_name)
                        break

                    # Ignore all unnecessary information.
                    filtered_data.append(
                        {
                            "filepath": data["filepath"],
                            "bbox": data["bbox"],
                            "R": data["R"],
                            "T": data["T"],
                            "focal_length": data["focal_length"],
                            "principal_point": data["principal_point"],
                        },
                    )

                if not bad_seq:
                    self.rotations[seq_name] = filtered_data

        self.sequence_list = list(self.rotations.keys())

        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.img_size, antialias=True),
                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ]
            )

            self.transform_depth = transforms.Compose(
                [
                    transforms.Resize(
                        self.depth_size,
                        antialias=False,
                        interpolation=transforms.InterpolationMode.NEAREST_EXACT,
                    ),
                ]
            )

        print(
            f"Low quality translation sequences, not used: {self.low_quality_translations}"
        )
        print(f"Data size: {len(self)}")
        print(f"Data loading took {(time.time()-start_time)} seconds.")

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, index):
        num_to_load = self.num_images if not self.load_extra_cameras else 8

        sequence_name = self.sequence_list[index % len(self.sequence_list)]
        metadata = self.rotations[sequence_name]

        if self.sample_num is not None:
            with open(
                order_path.format(sample_num=self.sample_num, category=self.category[0])
            ) as f:
                order = json.load(f)
            ids = order[sequence_name][:num_to_load]
        else:
            replace = len(metadata) < 8
            ids = np.random.choice(len(metadata), num_to_load, replace=replace)

        return self.get_data(index=index, ids=ids, num_valid_frames=num_to_load)

    def _get_scene_scale(self, sequence_name):
        n = len(self.rotations[sequence_name])

        R = torch.zeros(n, 3, 3)
        T = torch.zeros(n, 3)

        for i, ann in enumerate(self.rotations[sequence_name]):
            R[i, ...] = torch.tensor(self.rotations[sequence_name][i]["R"])
            T[i, ...] = torch.tensor(self.rotations[sequence_name][i]["T"])

        cameras = PerspectiveCameras(R=R, T=T)
        cc = cameras.get_camera_center()
        centeroid = torch.mean(cc, dim=0)
        diff = cc - centeroid

        norm = torch.norm(diff, dim=1)
        scale = torch.max(norm).item()

        return scale

    def _crop_image(self, image, bbox):
        image_crop = transforms.functional.crop(
            image,
            top=bbox[1],
            left=bbox[0],
            height=bbox[3] - bbox[1],
            width=bbox[2] - bbox[0],
        )
        return image_crop

    def _transform_intrinsic(self, image, bbox, principal_point, focal_length):
        half_box = np.array([image.width, image.height]).astype(np.float32) / 2
        org_scale = min(half_box).astype(np.float32)

        # Pixel coordinates
        principal_point_px = half_box - (np.array(principal_point) * org_scale)
        focal_length_px = np.array(focal_length) * org_scale
        principal_point_px -= bbox[:2]
        new_bbox = (bbox[2:] - bbox[:2]) / 2
        new_scale = min(new_bbox)

        # NDC coordinates
        new_principal_ndc = (new_bbox - principal_point_px) / new_scale
        new_focal_ndc = focal_length_px / new_scale

        return new_principal_ndc.astype(np.float32), new_focal_ndc.astype(np.float32)

    def get_data(
        self,
        index=None,
        sequence_name=None,
        ids=(0, 1),
        no_images=False,
        num_valid_frames=None,
        load_using_order=None,
    ):
        if load_using_order is not None:
            with open(
                order_path.format(sample_num=self.sample_num, category=self.category[0])
            ) as f:
                order = json.load(f)
            ids = order[sequence_name][:load_using_order]

        if sequence_name is None:
            index = index % len(self.sequence_list)
            sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        category = self.category_map[sequence_name]

        # Read image & camera information from annotations
        annos = [metadata[i] for i in ids]
        images = []
        image_sizes = []
        PP = []
        FL = []
        crop_parameters = []
        filenames = []
        distortion_parameters = []
        depths = []
        depth_masks = []
        object_masks = []
        dino_images = []
        for anno in annos:
            filepath = anno["filepath"]

            if not no_images:
                image = Image.open(osp.join(self.co3d_dir, filepath)).convert("RGB")
                image_size = image.size

                # Optionally mask images with black background
                if self.mask_images:
                    black_image = Image.new("RGB", image_size, (0, 0, 0))
                    mask_name = osp.basename(filepath.replace(".jpg", ".png"))

                    mask_path = osp.join(
                        self.co3d_dir, category, sequence_name, "masks", mask_name
                    )
                    mask = Image.open(mask_path).convert("L")

                    if mask.size != image_size:
                        mask = mask.resize(image_size)
                    mask = Image.fromarray(np.array(mask) > 125)
                    image = Image.composite(image, black_image, mask)
                
                if self.object_mask:
                    mask_name = osp.basename(filepath.replace(".jpg", ".png"))
                    mask_path = osp.join(
                        self.co3d_dir, category, sequence_name, "masks", mask_name
                    )
                    mask = Image.open(mask_path).convert("L")

                    if mask.size != image_size:
                        mask = mask.resize(image_size)
                    mask = torch.from_numpy(np.array(mask) > 125)

                # Determine crop, Resnet wants square images
                bbox = np.array(anno["bbox"])
                good_bbox = ((bbox[2:] - bbox[:2]) > 30).all()
                bbox = (
                    anno["bbox"]
                    if not self.center_crop and good_bbox
                    else [0, 0, image.width, image.height]
                )

                # Distort image and bbox if desired
                if self.distort:
                    k1 = random.uniform(0, self.k1_max)
                    k2 = random.uniform(0, self.k2_max)

                    try:
                        image, bbox = distort_image(
                            image, np.array(bbox), k1, k2, modify_bbox=True
                        )

                    except:
                        print("INFO:")
                        print(sequence_name)
                        print(index)
                        print(ids)
                        print(k1)
                        print(k2)

                    distortion_parameters.append(torch.FloatTensor([k1, k2]))

                bbox = square_bbox(np.array(bbox), tight=self.center_crop)
                if self.apply_augmentation:
                    bbox = jitter_bbox(
                        bbox,
                        jitter_scale=self.jitter_scale,
                        jitter_trans=self.jitter_trans,
                        direction_from_size=image.size if self.center_crop else None,
                    )
                bbox = np.around(bbox).astype(int)

                # Crop parameters
                crop_center = (bbox[:2] + bbox[2:]) / 2
                principal_point = torch.tensor(anno["principal_point"])
                focal_length = torch.tensor(anno["focal_length"])

                # convert crop center to correspond to a "square" image
                width, height = image.size
                length = max(width, height)
                s = length / min(width, height)
                crop_center = crop_center + (length - np.array([width, height])) / 2

                # convert to NDC
                cc = s - 2 * s * crop_center / length
                crop_width = 2 * s * (bbox[2] - bbox[0]) / length
                crop_params = torch.tensor([-cc[0], -cc[1], crop_width, s])

                # Crop and normalize image
                if not self.precropped_images:
                    image = self._crop_image(image, bbox)

                try:
                    image = self.transform(image)
                except:
                    print("INFO:")
                    print(sequence_name)
                    print(index)
                    print(ids)
                    print(k1)
                    print(k2)

                images.append(image[:, : self.img_size, : self.img_size])
                crop_parameters.append(crop_params)

                if self.load_depths:
                    # Open depth map
                    depth_name = osp.basename(
                        filepath.replace(".jpg", ".jpg.geometric.png")
                    )
                    depth_path = osp.join(
                        self.co3d_depth_dir,
                        category,
                        sequence_name,
                        "depths",
                        depth_name,
                    )
                    depth_pil = Image.open(depth_path)

                    # 16 bit float type casting
                    depth = torch.tensor(
                        np.frombuffer(
                            np.array(depth_pil, dtype=np.uint16), dtype=np.float16
                        )
                        .astype(np.float32)
                        .reshape((depth_pil.size[1], depth_pil.size[0]))
                    )

                    # Crop and resize as with images
                    if depth_pil.size != image_size:
                        # bbox may have the wrong scale
                        bbox = depth_pil.size[0] * bbox / image_size[0]

                    if self.object_mask:
                        assert mask.shape == depth.shape

                    bbox = np.around(bbox).astype(int)
                    depth = self._crop_image(depth, bbox)

                    # Resize
                    depth = self.transform_depth(depth.unsqueeze(0))[
                        0, : self.depth_size, : self.depth_size
                    ]
                    depths.append(depth)

                    if self.object_mask:
                        mask = self._crop_image(mask, bbox)
                        mask = self.transform_depth(mask.unsqueeze(0))[
                            0, : self.depth_size, : self.depth_size
                        ]
                        object_masks.append(mask)

                PP.append(principal_point)
                FL.append(focal_length)
                image_sizes.append(torch.tensor([self.img_size, self.img_size]))
                filenames.append(filepath)

        if not no_images:
            if self.load_depths:
                depths = torch.stack(depths)

                depth_masks = torch.logical_or(depths <= 0, depths.isinf())
                depth_masks = (~depth_masks).long()

                if self.object_mask:
                    object_masks = torch.stack(object_masks, dim=0)

                if self.mask_holes:
                    depths = fill_depths(depths, depth_masks == 0)

                # Sometimes mask_holes misses stuff
                new_masks = torch.logical_or(depths <= 0, depths.isinf())
                new_masks = (~new_masks).long()
                depths[new_masks == 0] = -1

                assert torch.logical_or(depths > 0, depths == -1).all()
                assert not (depths.isinf()).any()
                assert not (depths.isnan()).any()

            if self.load_extra_cameras:
                # Remove the extra loaded image, for saving space
                images = images[: self.num_images]

            if self.distort:
                distortion_parameters = torch.stack(distortion_parameters)

            images = torch.stack(images)
            crop_parameters = torch.stack(crop_parameters)
            focal_lengths = torch.stack(FL)
            principal_points = torch.stack(PP)
            image_sizes = torch.stack(image_sizes)
        else:
            images = None
            crop_parameters = None
            distortion_parameters = None
            focal_lengths = []
            principal_points = []
            image_sizes = []

        # Assemble batch info to send back
        R = torch.stack([torch.tensor(anno["R"]) for anno in annos])
        T = torch.stack([torch.tensor(anno["T"]) for anno in annos])

        batch = {
            "model_id": sequence_name,
            "category": category,
            "n": len(metadata),
            "num_valid_frames": num_valid_frames, 
            "ind": torch.tensor(ids),
            "image": images,
            "depth": depths,
            "depth_masks": depth_masks,
            "object_masks": object_masks,
            "R": R,
            "T": T,
            "focal_length": focal_lengths,
            "principal_point": principal_points,
            "image_size": image_sizes,
            "crop_parameters": crop_parameters,
            "distortion_parameters": torch.zeros(4),
            "filename": filenames,
            "category": category,
            "dataset": "co3d",
        }

        return batch
