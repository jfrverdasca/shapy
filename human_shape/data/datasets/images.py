import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.utils.data as dutils
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from human_shape.data.structures.bbox import BoundingBox
from human_shape.data.utils.bbox import bbox_to_center_scale
from human_shape.utils.typing import Array

EXTS = [".jpg", ".jpeg", ".png"]


def read_img(fname: str, dtype=np.float32) -> Array:
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if dtype == np.float32 and img.dtype == np.uint8:
        img = img.astype(dtype) / 255.0
    return img


def collate_fn(batch):
    output = defaultdict(list)

    for sample in batch:
        for k, v in sample.items():
            output[k].append(v)
    return output


def calculate_bounding_boxes(device, min_score, rcnn_dataloader, rcnn_model):
    img_paths = []
    bboxes = []
    for bidx, batch in enumerate(tqdm(rcnn_dataloader, desc="Processing with R-CNN")):
        batch["images"] = [x.to(device) for x in batch["images"]]

        output = rcnn_model(batch["images"])
        for i, x in enumerate(output):
            img_path = batch["paths"][i]
            _, fname = osp.splitext(img_path)
            fname, _ = os.path.splitext(fname)

            for j, bbox in enumerate(output[i]["boxes"]):
                bbox = bbox.detach().cpu().numpy()
                if output[i]["scores"][j].item() < min_score:
                    continue

                img_paths.append(img_path)
                bboxes.append(bbox)

    return bboxes, img_paths


class _ImagesDataset(dutils.Dataset):
    def __init__(self, data_folder: str, transforms=None, **kwargs):
        super(_ImagesDataset, self).__init__()

        self.transforms = transforms

        paths = []
        data_folder = osp.expandvars(data_folder)
        for fname in os.listdir(data_folder):
            if not any(fname.endswith(ext) for ext in EXTS):
                continue

            paths.append(osp.join(data_folder, fname))

        self.paths = np.stack(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = read_img(self.paths[index])

        if self.transforms is not None:
            img = self.transforms(img)

        return {"images": img, "paths": self.paths[index]}


class Images(dutils.Dataset):

    def __init__(
        self,
        data_folder="data/images",
        min_score=0.8,
        scale_factor=1.2,
        transforms=None,
        **kwargs,
    ):
        super(Images, self).__init__()

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.scale_factor = scale_factor
        self.transforms = transforms

        rcnn_model = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        rcnn_model.eval()
        rcnn_model = rcnn_model.to(device=self.device)

        images_dataset = _ImagesDataset(data_folder, transforms=Compose([ToTensor()]))
        rcnn_dataloader = dutils.DataLoader(
            images_dataset, batch_size=1, num_workers=0, collate_fn=collate_fn
        )

        bboxes, img_paths = calculate_bounding_boxes(
            self.device, min_score, rcnn_dataloader, rcnn_model
        )

        self.bboxes = np.stack(bboxes)
        self.paths = np.stack(img_paths)

    def name(self):
        return "Images"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = read_img(self.paths[index])

        bbox = self.bboxes[index]
        target = BoundingBox(bbox, img.shape)

        center, scale, bbox_size = bbox_to_center_scale(
            bbox, dset_scale_factor=self.scale_factor
        )
        target.add_field("center", center)
        target.add_field("orig_center", center)
        target.add_field("scale", scale)
        target.add_field("bbox_size", bbox_size)
        target.add_field("orig_bbox_size", bbox_size)

        _, fname = osp.split(self.paths[index])
        target.add_field("fname", f"{fname}_{index:03d}")

        if self.transforms is not None:
            img, cropped_image, target = self.transforms(img, target)

        return img, cropped_image, target, index
