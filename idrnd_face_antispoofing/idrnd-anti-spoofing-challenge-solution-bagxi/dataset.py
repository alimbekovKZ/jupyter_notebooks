# Copyright (c) 2019-present, Yauheni Kachan. All Rights Reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import collections
import functools
import math
from typing import Callable, Dict, List, Tuple, Type

import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from catalyst.utils.image import imread
import cv2
import pandas as pd
from pytorch_toolbelt.utils.torch_utils import (
    tensor_from_mask_image, tensor_from_rgb_image
)
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class AntispoofDataset(Dataset):
    def __init__(
        self,
        image_filenames: pd.DataFrame,
        mode: str = 'infer',
        transform: Callable = None,
        rootpath: str = None
    ):
        self.mode = mode
        self.images = image_filenames
        self.transform = transform
        self.rootpath = rootpath

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image_info = self._get_image_info(index)
        image = imread(image_info['path'], rootpath=self.rootpath)
        if self.transform is not None:
            image = self.transform(image=image)['image']

        if self.mode == 'infer':
            return image_info['id'], image_info['frame'], image
        return image, image_info['label']

    def _get_image_info(self, index):
        return self.images.iloc[index]


class AntispoofSubsampleDataset(AntispoofDataset):
    def __init__(
        self,
        image_filenames: pd.DataFrame,
        subsample_id: str,
        mode: str = 'infer',
        transform: Callable = None,
        rootpath: str = None
    ):
        super().__init__(
            image_filenames=image_filenames.groupby(subsample_id), mode=mode, transform=transform, rootpath=rootpath
        )
        self.image_keys = list(self.images.groups.keys())

    def __len__(self):
        return self.images.ngroups

    def _get_image_info(self, index):
        key = self.image_keys[index]
        images = self.images.get_group(key)
        image_info = images.sample(frac=1, random_state=None).iloc[0]  # shuffle data and choose first sample

        return image_info


def padding_for_rotation(image_size: Tuple[int, int], rotation):
    r = math.sqrt((image_size[0] / 2) ** 2 + (image_size[1] / 2) ** 2)

    rot_angle_rads = math.radians(45 - rotation)

    pad_h = int(r * math.cos(rot_angle_rads) - image_size[0] // 2)
    pad_w = int(r * math.cos(rot_angle_rads) - image_size[1] // 2)

    return pad_h, pad_w


class ToTensor(DualTransform):
    def __init__(self):
        super(ToTensor, self).__init__(always_apply=True, p=1.0)

    def __call__(self, force_apply=True, **kwargs):
        kwargs.update({'image': tensor_from_rgb_image(kwargs['image'])})
        if 'mask' in kwargs.keys():
            kwargs.update({
                'mask': tensor_from_mask_image(kwargs['mask'].float())
            })

        return kwargs


def light_augmentations(image_size: Tuple[int, int]):
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(),
        ToTensor()
    ])


def medium_augmentations(image_size: Tuple[int, int], rot_angle=15):
    return A.Compose([
        A.OneOf([
            A.RandomSizedCrop((image_size[0], int(image_size[0] * 1.25)), image_size[0], image_size[1], p=0.05),
            A.RandomSizedCrop((image_size[0], int(image_size[0] * 1.5)), image_size[0], image_size[1], p=0.10),
            A.RandomSizedCrop((image_size[0], int(image_size[0] * 2)), image_size[0], image_size[1], p=0.15),
            A.Resize(image_size[0], image_size[1], p=0.7),
        ], p=1.0),

        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.3),

        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.RandomGamma(gamma_limit=(85, 115), p=0.2),
        A.HueSaturationValue(p=0.2),
        A.CLAHE(p=0.2),
        A.JpegCompression(quality_lower=50, p=0.2),

        A.Normalize(),
        ToTensor()
    ])


def hard_augmentations(image_size: Tuple[int, int], rot_angle=30):
    pad_h, pad_w = padding_for_rotation(image_size, rot_angle)
    crop_height = int(image_size[0] + pad_h * 2)
    crop_width = int(image_size[1] + pad_w * 2)
    crop_transform = A.Compose([
        A.RandomSizedCrop((int(crop_height * 0.75), int(crop_height * 1.25)), crop_height, crop_width),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=rot_angle, border_mode=cv2.BORDER_CONSTANT),
        A.CenterCrop(image_size[0], image_size[1]),
    ])

    return A.Compose([
        # spatial transform
        A.PadIfNeeded(int(crop_height * 1.25), int(crop_height * 1.25)),
        A.OneOf([
            crop_transform,
            A.RandomSizedCrop((image_size[0], int(image_size[0] * 1.25)), image_size[0], image_size[1], p=0.25),
            A.RandomSizedCrop((image_size[0], int(image_size[0] * 1.5)), image_size[0], image_size[1], p=0.25),
            A.RandomSizedCrop((image_size[0], int(image_size[0] * 2)), image_size[0], image_size[1], p=0.25),
            A.Resize(image_size[0], image_size[1], p=0.75)
        ], p=1.0),

        # add occasion blur/sharpening
        A.OneOf([
            A.GaussianBlur(),
            A.MotionBlur(),
            A.IAASharpen(),
            A.JpegCompression(quality_lower=75, p=0.25),
        ]),

        # D4 augmentations
        A.Compose([
            A.HorizontalFlip(),
        ]),

        # spatial-preserving augmentations
        A.OneOf([
            A.Cutout(),
            A.GaussNoise(),
        ]),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.CLAHE(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.RandomGamma()
        ]),

        A.Normalize(),
        ToTensor()
    ])


def get_dataloaders(
    in_csv: str,
    in_dir: str,
    stages: List[str],
    fold: int = None,
    batch_size: int = 8,
    n_workers: int = 0,
    image_size: int = (224, 224),
    augmentation: str = 'light',
    fast: bool = False
) -> Dict[str, Type[DataLoader]]:
    df = pd.read_csv(in_csv)

    if augmentation == 'light':
        train_transform = light_augmentations
    elif augmentation == 'medium':
        train_transform = medium_augmentations
    elif augmentation == 'hard':
        train_transform = hard_augmentations
    else:
        raise ValueError(f'invalid augmentations: `{augmentation}`')

    TrainDataset = functools.partial(AntispoofSubsampleDataset, subsample_id='id') if fast else AntispoofDataset

    loaders = collections.OrderedDict()
    for mode, Dataset_, image_filenames, transform in [
        ('train', TrainDataset, df[df['fold'] != fold] if fold is not None else df, train_transform),
        ('valid', TrainDataset, df[df['fold'] == fold] if fold is not None else df, light_augmentations),
        ('infer', AntispoofDataset, df, light_augmentations)
    ]:
        if mode in stages:
            loaders[mode] = DataLoader(
                Dataset_(image_filenames=image_filenames, mode=mode, transform=transform(image_size), rootpath=in_dir),
                batch_size=batch_size,
                shuffle=(mode == 'train'),
                num_workers=n_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=(mode == 'train')
            )

    return loaders
