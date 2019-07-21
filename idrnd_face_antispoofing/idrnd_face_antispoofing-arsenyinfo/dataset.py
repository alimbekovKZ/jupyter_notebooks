import os
from copy import deepcopy
from functools import partial
from glob import glob
from hashlib import sha1
from typing import Callable, Iterable, Optional, Tuple

import cv2
import numpy as np
from glog import logger
from joblib import Parallel, cpu_count, delayed
from skimage.io import imread
from torch.utils.data import Dataset
from tqdm import tqdm

import aug


def subsample(data: Iterable, bounds: Tuple[float, float], hash_fn: Callable, n_buckets=100, salt='', verbose=True):
    data = list(data)
    buckets = split_into_buckets(data, n_buckets=n_buckets, salt=salt, hash_fn=hash_fn)

    lower_bound, upper_bound = [x * n_buckets for x in bounds]
    msg = f'Subsampling buckets from {lower_bound} to {upper_bound}, total buckets number is {n_buckets}'
    if salt:
        msg += f'; salt is {salt}'
    if verbose:
        logger.info(msg)
    return np.array([sample for bucket, sample in zip(buckets, data) if lower_bound <= bucket < upper_bound])


def split_into_buckets(data: Iterable, n_buckets: int, hash_fn: Callable, salt=''):
    hashes = map(partial(hash_fn, salt=salt), data)
    return np.array([int(x, 16) % n_buckets for x in hashes])


def _read_img(x: str):
    img = cv2.imread(x)
    if img is None:
        logger.warning(f'Can not read image {x} with OpenCV, switching to scikit-image')
        img = imread(x)
    return img


def create_soften_fn(v):
    def soften(x):
        r = np.ones(4) * (1 - v) / 4
        r[x] = v
        return r.astype('float32') / r.sum()

    return soften


class IdRndDataset(Dataset):
    def __init__(self,
                 files: Tuple[str],
                 transform_fn: Callable,
                 normalize_fn: Callable,
                 corrupt_fn: Optional[Callable] = None,
                 soften_fn: Optional[Callable] = None,
                 preload: bool = True,
                 preload_size: Optional[int] = 0,
                 mixup: float = 0,
                 verbose=True):

        self.size = preload_size
        self.preload = preload
        self.imgs = files
        self.labels = [self._get_label(f) for f in files]
        self.verbose = verbose
        self.corrupt_fn = corrupt_fn
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.soften_fn = soften_fn
        self.mixup_proba = mixup
        logger.info(f'Dataset has been created with {len(self.imgs)} samples')

        if preload:
            preload_fn = partial(self._bulk_preload, preload_size=preload_size)
            self.imgs = preload_fn(self.imgs)

    @staticmethod
    def _get_label(x):
        label, *_ = os.path.basename(x).split('_')
        labels = {'2dmask': 0,
                  'printed': 1,
                  'replay': 2,
                  'real': 3}
        return labels[label]

    def _bulk_preload(self, data: Iterable[str], preload_size: int):
        jobs = [delayed(self._preload)(x, preload_size=preload_size) for x in data]
        jobs = tqdm(jobs, desc='preloading images', disable=not self.verbose)
        return Parallel(n_jobs=cpu_count(), backend='threading')(jobs)

    @staticmethod
    def _preload(x: str, preload_size: int):
        img = _read_img(x)
        if preload_size:
            h, w, *_ = img.shape
            h_scale = preload_size / h
            w_scale = preload_size / w
            scale = max(h_scale, w_scale)
            img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
            assert min(img.shape[:2]) >= preload_size, f'weird img shape: {img.shape}'
        return img

    def _preprocess(self, img):
        def transpose(x):
            return np.transpose(x, (2, 0, 1))

        return transpose(self.normalize_fn(img))

    def __len__(self):
        return len(self.imgs)

    def get_raw(self, idx):
        img, label = self.imgs[idx], self.labels[idx]
        if not self.preload:
            img = self._preload(img, self.size)
        img = self.transform_fn(img)
        if self.soften_fn is not None:
            label = self.soften_fn(label)
        return img, label

    def mixup(self, idx):
        """
        https://arxiv.org/pdf/1710.09412.pdf
        """
        img_a, label_a = self.get_raw(idx)
        new_idx = idx
        while idx == new_idx:
            new_idx = np.random.randint(len(self))
        img_b, label_b = self.get_raw(new_idx)
        lmbda = np.random.beta(1, 1)

        img = img_a.astype('float32') * lmbda + img_b.astype('float32') * (1 - lmbda)
        label = label_a * lmbda + label_b * (1 - lmbda)
        return img.astype('uint8'), label

    def __getitem__(self, idx):
        get_fn = self.get_raw if np.random.rand() > self.mixup_proba else self.mixup
        img, label = get_fn(idx)
        if self.corrupt_fn is not None:
            img = self.corrupt_fn(img)

        img = self._preprocess(img)
        return {'img': img, 'label': label}

    @staticmethod
    def from_config(config):
        config = deepcopy(config)
        files = glob(f'{config.get("data_dir", "/home/arseny/datasets/idrnd_train")}/**/*.png', recursive=True)

        transform_fn = aug.get_transforms(size=config['size'], scope=config['scope'], crop=config['crop'])
        normalize_fn = aug.get_normalize()
        corrupt_fn = aug.get_corrupt_function(config['corrupt'])
        soften_fn = create_soften_fn(config.get('soften', 1))

        def hash_fn(x: str, salt: str = '') -> str:
            x = os.path.basename(x)
            label, video, frame = x.split('_')
            return sha1(f'{label}_{video}_{salt}'.encode()).hexdigest()

        verbose = config.get('verbose', True)
        n_fold = config['n_fold']
        total_folds = 10
        test = subsample(data=files,
                         bounds=(1 / total_folds * n_fold, 1 / total_folds * (n_fold + 1)),
                         hash_fn=hash_fn,
                         verbose=verbose,
                         salt='validation'
                         )

        if config['test']:
            data = test
        else:
            files = set(files) - set(test)
            data = subsample(data=files,
                             bounds=config.get('bounds', (0, 1)),
                             hash_fn=hash_fn,
                             verbose=verbose,
                             salt=config['salt'])

        return IdRndDataset(files=tuple(data),
                            preload=config['preload'],
                            preload_size=config['preload_size'],
                            corrupt_fn=corrupt_fn,
                            normalize_fn=normalize_fn,
                            transform_fn=transform_fn,
                            soften_fn=soften_fn,
                            mixup=config['mixup'],
                            verbose=verbose)

    def update_config(self, config):
        self.corrupt_fn = aug.get_corrupt_function(config['corrupt'])
        self.soften_fn = create_soften_fn(config['soften'])
        self.mixup_proba = config['mixup']
