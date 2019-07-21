# Copyright (c) 2019-present, Yauheni Kachan. All Rights Reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import os
import pickle

from catalyst.utils.image import imread
import cv2
import fire
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from sklearn import metrics, model_selection
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from face_cropper import crop_faces, FaceDetectionEngine
from lbp import CropClassificator, LBPFeatureExtractor
from utils import find_items


def infer(
    in_csv: str,
    in_dir: str,
    out_csv: str,
    weights_path: str,
    n_workers: int = 4,
    verbose: bool = False
):
    model = CropClassificator(
        face_detector=FaceDetectionEngine(weights_path='./models/s3fd_convert.pth'),
        feature_extractor=LBPFeatureExtractor(n_features=59, crop_size=(64, 64)),
        classificator=load_model(weights_path),
        agg_fn=gmean
    )
    df = pd.read_csv(in_csv)
    samples, frames, probabilities = [], [], []
    for idx, image_info in tqdm(df.iterrows(), total=df.shape[0], disable=not verbose):
        image = imread(image_info['path'], rootpath=in_dir)
        probability = model.predict(image)
        samples.append(image_info['id'])
        frames.append(image_info['frame'])
        probabilities.append(probability)

    df = pd.DataFrame.from_dict({'id': samples, 'frame': frames, 'prediction': probabilities})
    df = df.groupby('id')['prediction'].apply(gmean).reset_index()
    df[['id', 'prediction']].to_csv(out_csv, index=False)


def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    return model


def train(
    features_npy: str,
    targets_csv: str,
    n_splits: int = 5,
    n_repeats: int = 10,
    logdir: str = '.',
    random_seed=82
):
    model = LogisticRegression(
        penalty='elasticnet',
        C=1.0,
        class_weight='balanced',
        random_state=random_seed,
        solver='saga',
        max_iter=200,
        n_jobs=-1,
        l1_ratio=1.0
    )

    X = np.load(features_npy)
    df = pd.read_csv(targets_csv)
    y = df['label'].values

    logs = []
    splitter = model_selection.RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed
    )
    pbar = tqdm(
        splitter.split(X, y, groups=df['id']), desc='folds', total=splitter.get_n_splits()
    )
    for i, (train_index, valid_index) in enumerate(pbar):
        model_ = clone(model)
        X_train, X_test = X[train_index], X[valid_index]
        y_train, y_test = y[train_index], y[valid_index]

        model_.fit(X_train, y_train)
        preds = model_.predict_proba(X_test)[:, 1]
        logs.append({'auc': metrics.roc_auc_score(y_test, preds)})
        pbar.set_postfix(**logs[-1])
    auc_ = np.array([it['auc'] for it in logs])
    print(f'AUC (mean): {auc_.mean()}\tAUC (str): {auc_.std()}')
    with open(os.path.join(logdir, 'logs.pkl'), 'wb') as f:
        pickle.dump(logs, f)

    # train final model on all data
    model.fit(X, y)
    with open(os.path.join(logdir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)


def prepare_lbp_dataset(dirpath: str, features_npy: str, targets_csv: str, verbose: bool = True):
    feature_extractor = LBPFeatureExtractor(n_features=59, crop_size=(64, 64))

    features, targets = [], []
    df = find_items(in_dir=dirpath)
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], disable=not verbose):
        image = imread(row['path'], rootpath=dirpath)
        features.append(feature_extractor(image))
        targets.append(row)

    np.save(features_npy, np.stack(features, axis=0))
    pd.DataFrame(targets).to_csv(targets_csv, index=False)


def prepare_cutout_datasets(in_dir: str, out_dir_crops: str, out_dir_cutout: str, verbose: bool = False):
    df = find_items(in_dir=in_dir)
    face_detector = FaceDetectionEngine(weights_path='./models/s3fd_convert.pth')
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], disable=not verbose):
        image = imread(row['path'], rootpath=in_dir)
        face_crops, cutout_faces = crop_faces(image, face_detector)

        os.makedirs(os.path.join(out_dir_crops, os.path.dirname(row['path'])), exist_ok=True)
        for i, (crop, bbox) in enumerate(face_crops):
            root, ext = os.path.splitext(row['path'])
            cv2.imwrite(os.path.join(out_dir_crops, f'{root}_{i}{ext}'), crop[:, :, ::-1])  # RGB -> BGR

        os.makedirs(os.path.join(out_dir_cutout, os.path.dirname(row['path'])), exist_ok=True)
        cv2.imwrite(os.path.join(out_dir_cutout, row['path']), cutout_faces[:, :, ::-1])  # RGB -> BGR


if __name__ == '__main__':
    fire.Fire({
        'infer': infer,
        'train': train,
        'prepare-lbp-dataset': prepare_lbp_dataset,
        'prepare-cutout-datasets': prepare_cutout_datasets
    })
