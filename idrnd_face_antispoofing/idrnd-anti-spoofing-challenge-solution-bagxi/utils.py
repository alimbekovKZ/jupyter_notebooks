# Copyright (c) 2019-present, Yauheni Kachan. All Rights Reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import collections
import os

import pandas as pd


def find_items(in_dir: str) -> pd.DataFrame:
    path_images = []
    for label in ['2dmask', 'real', 'printed', 'replay']:
        for sample in os.listdir(os.path.join(in_dir, label)):
            frames = sorted(os.listdir(os.path.join(in_dir, label, sample)))
            for i, frame in enumerate(frames):
                path_images.append(collections.OrderedDict(
                    id=sample,
                    frame=i,
                    path=os.path.join(label, sample, frame),
                    label=int(label != 'real'),
                ))

    return pd.DataFrame(path_images)
