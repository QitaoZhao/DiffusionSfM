
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from diffusionsfm.dataset.co3d_v2 import square_bbox


class CustomDataset(Dataset):
    def __init__(
        self,
        image_list,
    ):
        self.images = []

        for image_path in sorted(image_list):
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img).convert("RGB")  # Apply EXIF rotation
            self.images.append(img)

        self.n = len(self.images)
        self.jitter_scale = [1, 1]
        self.jitter_trans = [0, 0]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform_for_vis = transforms.Compose(
            [
                transforms.Resize(224),
            ]
        )

    def __len__(self):
        return 1

    def _crop_image(self, image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new(
                "RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255)
            )
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image,
                top=bbox[1],
                left=bbox[0],
                height=bbox[3] - bbox[1],
                width=bbox[2] - bbox[0],
            )
        return image_crop

    def __getitem__(self):
        return self.get_data()

    def get_data(self):
        cmap = plt.get_cmap("hsv")
        ids = [i for i in range(len(self.images))]
        images = [self.images[i] for i in ids]
        images_transformed = []
        images_for_vis = []
        crop_parameters = []
        
        for i, image in enumerate(images):
            bbox = np.array([0, 0, image.width, image.height])
            bbox = square_bbox(bbox, tight=True)
            bbox = np.around(bbox).astype(int)
            image = self._crop_image(image, bbox)
            images_transformed.append(self.transform(image))
            image_for_vis = self.transform_for_vis(image)
            color_float = cmap(i / len(images))
            color_rgb = tuple(int(255 * c) for c in color_float[:3])
            image_for_vis = ImageOps.expand(image_for_vis, border=3, fill=color_rgb)
            images_for_vis.append(image_for_vis)

            width, height = image.size
            length = max(width, height)
            s = length / min(width, height)
            crop_center = (bbox[:2] + bbox[2:]) / 2
            crop_center = crop_center + (length - np.array([width, height])) / 2
            # convert to NDC
            cc = s - 2 * s * crop_center / length
            crop_width = 2 * s * (bbox[2] - bbox[0]) / length
            crop_params = torch.tensor([-cc[0], -cc[1], crop_width, s])

            crop_parameters.append(crop_params)
        images = images_transformed

        batch = {}
        batch["image"] = torch.stack(images)
        batch["image_for_vis"] = images_for_vis
        batch["n"] = len(images)
        batch["ind"] = torch.tensor(ids),
        batch["crop_parameters"] = torch.stack(crop_parameters)
        batch["distortion_parameters"] = torch.zeros(4)

        return batch
