import os
import json
import collections
import configparser

from datetime import datetime
from .utils import ON_KAGGLE

data_path = '../input/aptos2019-blindness-detection'

dataset_map = {
    "train": os.path.join(data_path, 'train.csv'),
    "test": os.path.join(data_path, 'test.csv'),
    "fold": 'folds.csv',
    "train_images": os.path.join(data_path, 'train_images'),
    "test_images": os.path.join(data_path, 'test_images'),
    "submission": os.path.join(data_path, 'sample_submission.csv')
}

diabetic_retinopathy_path = '../input/diabetic-retinopathy-resized'

diabetic_retinopathy_map = {
    "train": os.path.join(diabetic_retinopathy_path, 'trainLabels_cropped.csv'),
    "train_images": os.path.join(diabetic_retinopathy_path, 'resized_train_cropped'),
}

if ON_KAGGLE:
    diabetic_retinopathy_map['train_images'] = \
        os.path.join(diabetic_retinopathy_path, 'resized_train_cropped', 'resized_train_cropped')


default_config = {
    'name': 'exp_{}'.format(datetime.now().replace(second=0, microsecond=0)),
        "input": {
        "width": 486,
        "height": 486,
        "transforms": [
            "random_resized_crop"
        ],
        "test_transforms": [
            "random_resized_crop"
        ]
    },
    "dataset": {
        "fold": 0,
        "num_class": 5,
        "batch_size": 1,
        "num_workers": 2,
        "method":"regression",
        "use_original": True,
        "use_diabetic_retinopathy": True,
        "valid_with_both": True,
        "valid_with_large": False,
        "valid_with_small": False,
        "upsampling": False,
        "use_class_ratio": True,
        "use_dataset_ratio": True,
        "class_ratio": [1, 1, 1, 1, 1], # diagnosis 0, 1, 2, 3, 4
        "dataset_ratio": [1, 1], # origin_train, diabetic
    },
    "model": {
        "name": "resnet50",
        "pretrained": False,
        "weight_path": "../input/resnet50/resnet50.pth"
    },
    "train_param": {
        "epoch": 30,
        "lr": 0.0001,
        "grad_clip_step": 20,
        "grad_clip": 100.0,
        "scheduler": "cosine",
        "cosine": {
            "t_max": 20,
            "eta_min": 0
        },
        "steplr": {
            "step_size": 4,
            "gamma": 0.1
        }
    }
}

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_cfg(cfg_path):
    default_config
    if cfg_path is None:
        return default_config
    with open(cfg_path, 'r') as f:
        cfg = json.loads(f.read())
    
    cfg = update(default_config, cfg)
    return cfg
