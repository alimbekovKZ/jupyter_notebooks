from typing import List, Optional

import albumentations as albu
from albumentations.core.transforms_interface import NoOp


def get_transforms(size: int, scope: str = 'geometric', crop='random'):
    augs = {'strong': albu.Compose([albu.HorizontalFlip(),
                                    albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=20, p=.4),
                                    albu.ElasticTransform(),
                                    albu.OpticalDistortion(),
                                    albu.OneOf([
                                        albu.CLAHE(clip_limit=2),
                                        albu.IAASharpen(),
                                        albu.IAAEmboss(),
                                        albu.RandomBrightnessContrast(),
                                        albu.RandomGamma()
                                    ], p=0.5),
                                    albu.OneOf([
                                        albu.RGBShift(),
                                        albu.HueSaturationValue(),
                                    ], p=0.5),
                                    ]),
            'weak': albu.Compose([albu.HorizontalFlip(),
                                  ]),
            'geometric': albu.OneOf([albu.HorizontalFlip(always_apply=True),
                                     albu.ShiftScaleRotate(always_apply=True, scale_limit=.5, rotate_limit=30),
                                     albu.Transpose(always_apply=True),
                                     albu.OpticalDistortion(always_apply=True, distort_limit=0.1, shift_limit=0.1),
                                     albu.ElasticTransform(always_apply=True),
                                     ]),
            'empty': NoOp(),
            }

    aug_fn = augs[scope]
    crop_fn = {'random': albu.RandomCrop(size, size, always_apply=True),
               'center': albu.CenterCrop(size, size, always_apply=True)}[crop]
    pad = albu.PadIfNeeded(size, size)

    pipeline = albu.Compose([aug_fn, crop_fn, pad])

    def process(a):
        r = pipeline(image=a)
        return r['image']

    return process


def get_normalize():
    normalize = albu.Normalize()

    def process(a):
        r = normalize(image=a)
        return r['image']

    return process


def _resolve_aug_fn(name):
    d = {
        'cutout': albu.Cutout,
        'rgb_shift': albu.RGBShift,
        'hsv_shift': albu.HueSaturationValue,
        'motion_blur': albu.MotionBlur,
        'median_blur': albu.MedianBlur,
        'snow': albu.RandomSnow,
        'shadow': albu.RandomShadow,
        'fog': albu.RandomFog,
        'brightness_contrast': albu.RandomBrightnessContrast,
        'gamma': albu.RandomGamma,
        'sun_flare': albu.RandomSunFlare,
        'sharpen': albu.IAASharpen,
        'jpeg': albu.JpegCompression,
        'gray': albu.ToGray,
        'channel_shuffle': albu.ChannelShuffle,
        'grid_distortion': albu.GridDistortion,

        # ToDo: pixelize
        # ToDo: partial gray
    }
    return d[name]


def get_corrupt_function(config: Optional[List[dict]]):
    if config is None:
        return
    augs = []
    for aug_params in config:
        name = aug_params.pop('name')
        cls = _resolve_aug_fn(name)
        prob = aug_params.pop('prob') if 'prob' in aug_params else .5
        augs.append(cls(p=prob, **aug_params))

    augs = albu.OneOf(augs)

    def process(x):
        return augs(image=x)['image']

    return process
