# Copyright (c) 2019-present, Yauheni Kachan. All Rights Reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

"""
Anti-spoofing based on LBP:
- http://www.outex.oulu.fi/publications/pami_02_opm.pdf
- https://publications.idiap.ch/downloads/papers/2012/Chingovska_IEEEBIOSIG2012_2012.pdf
- https://arxiv.org/pdf/1511.06316.pdf

"""

import collections
from typing import Callable, Dict, Tuple

from albumentations.augmentations import functional as F
import cv2
import numpy as np
from scipy.stats.mstats import gmean
from skimage.feature import local_binary_pattern

from face_cropper import crop_faces


class LBPFeatureExtractor:
    """

    Args:
        n_features: Number of per channel features

    """

    def __init__(self, n_features: int = 59, crop_size: Tuple[int, int] = (64, 64)):
        self.per_channel_features = n_features
        self.height, self.width = crop_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = F.resize(image, height=self.height, width=self.width)
        color_features = self.color_spaces(image)
        features = self.extract_global_features(color_features)

        return features

    def extract_global_features(self, color_bands: Dict[str, np.ndarray]) -> np.ndarray:
        features = np.concatenate([
            self.lbp_hist(color_band=v, n_bins=self.per_channel_features)
            for k, v in color_bands.items()
        ], axis=0)

        return features

    # TODO:
    @staticmethod
    def extract_local_features(image: np.ndarray, kernel) -> np.ndarray:
        h, w, _ = image.shape
        h_step = h // kernel
        w_step = w // kernel
        for i in range(0, w, w_step):
            for j in range(0, h, h_step):
                raise NotImplementedError

    @staticmethod
    def lbp_hist(color_band: np.ndarray, n_bins: int = 59, kernel: int = 3, eps=1e-7) -> np.ndarray:
        """https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/"""
        neighbours = kernel ** 2 - 1
        lbp = local_binary_pattern(color_band, P=neighbours, R=1, method='uniform')
        hist, _ = np.histogram(lbp, bins=n_bins)

        # normalize hsit
        hist = hist / (hist.sum() + eps)

        return hist

    @staticmethod
    def color_spaces(image: np.ndarray) -> Dict:
        """RGB -> YCbCr, HSV"""
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        color_bands = collections.OrderedDict(
            y=image_ycbcr[..., 0], cb=image_ycbcr[..., 1], cr=image_ycbcr[..., 2],
            h=image_hsv[..., 0], s=image_hsv[..., 1], v=image_hsv[..., 2],
        )

        return color_bands


class CropClassificator:
    def __init__(
        self,
        face_detector,
        feature_extractor,
        classificator,
        agg_fn: Callable[[np.ndarray], float] = gmean
    ):
        self.face_detector = face_detector
        self.feature_extractor = feature_extractor
        self.classificator = classificator
        self.agg_fn = agg_fn

    def predict(self, image: np.ndarray) -> float:
        """Predict probability of spoofing"""
        face_crops, _ = crop_faces(image, self.face_detector)
        if face_crops:
            probs = np.array([self.predict_crop(crop) for crop, bbox in face_crops])
            p = self.agg_fn(probs, axis=None)
        else:
            p = 1

        return p

    def predict_crop(self, image: np.ndarray) -> float:
        features = self.feature_extractor(image)
        features = features.reshape(1, -1)  # for single sample prediction
        proba = self.classificator.predict_proba(features)[:, 1]

        return proba
