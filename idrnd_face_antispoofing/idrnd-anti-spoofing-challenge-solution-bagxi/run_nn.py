# Copyright (c) 2019-present, Yauheni Kachan. All Rights Reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import collections

from catalyst.contrib import criterion, schedulers
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback
import fire
import pandas as pd
from pytorch_toolbelt.modules.backbone import senet
from scipy.stats.mstats import gmean
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
import torchvision

from dataset import get_dataloaders
from metrics import BinaryAUCCallback
from utils import find_items


def get_model(model: str, n_targets: int = 2, weights_path: str = None, device=None):
    if model.startswith('se'):  # pytorch-toolbelt SEResNet models
        model_constructor = getattr(senet, model)
        model = model_constructor(pretrained=('imagenet' if weights_path is None else None))
        model.last_linear = nn.Linear(model.last_linear.in_features, n_targets)
    else:  # torchvision models
        model_constructor = getattr(torchvision.models, model)
        model = model_constructor(pretrained=(weights_path is None))
        model.fc = torch.nn.Linear(model.fc.in_features, n_targets)

    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model.to(device=device)


def distil_model(model: str, in_weights: str, out_weights: str, device='cpu'):
    model = get_model(model=model, weights_path=in_weights, device=device)

    checkpoint = collections.OrderedDict(model_state_dict=model.state_dict())
    torch.save(checkpoint, out_weights)


def infer(
    in_csv: str,
    in_dir: str,
    out_csv: str,
    model: str,
    weights_path: str,
    image_size: int = 224,
    batch_size: int = 256,
    n_workers: int = 4
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_model(model=model, weights_path=weights_path, device=device).eval()

    dataloader = get_dataloaders(
        in_csv=in_csv,
        in_dir=in_dir,
        stages=['infer'],
        batch_size=batch_size,
        n_workers=n_workers,
        image_size=(image_size, image_size),
        fast=False
    )['infer']

    samples, frames, probabilities = [], [], []
    with torch.no_grad():
        for video, frame, batch in dataloader:
            batch = batch.to(device)
            probability = torch.sigmoid(model(batch)[:, 1].view(-1))

            samples.extend(video)
            frames.extend(frame.numpy())
            probabilities.extend(probability.cpu().numpy())

    df = pd.DataFrame.from_dict({'id': samples, 'frame': frames, 'prediction': probabilities})
    df = df.groupby('id')['prediction'].apply(gmean).reset_index()
    df[['id', 'prediction']].to_csv(out_csv, index=False)


def train(
    in_csv: str,
    in_dir: str,
    model: str = 'resnet18',
    fold: int = None,
    n_epochs: int = 30,
    image_size: int = 224,
    augmentation: str = 'medium',
    learning_rate: float = 3e-3,
    n_milestones: int = 5,
    batch_size: int = 256,
    n_workers: int = 4,
    fast: bool = False,
    logdir: str = '.',
    verbose: bool = False
):
    model = get_model(model=model)
    loss = criterion.FocalLossMultiClass()  # CrossEntropyLoss
    lr_scaled = learning_rate * (batch_size / 256)  # lr linear scaling
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_scaled)
    scheduler = schedulers.MultiStepLR(
        optimizer,
        milestones=[5, 10, 20, 30, 40],
        gamma=0.3
    )

    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=get_dataloaders(
            in_csv=in_csv,
            in_dir=in_dir,
            stages=['train', 'valid'],
            fold=fold,
            batch_size=batch_size,
            n_workers=n_workers,
            image_size=(image_size, image_size),
            augmentation=augmentation,
            fast=fast
        ),
        callbacks=[
            AccuracyCallback(accuracy_args=[1]),
            BinaryAUCCallback()
        ],
        logdir=logdir,
        num_epochs=n_epochs,
        verbose=verbose
    )


def prepare_folds(
    in_dir: str,
    out_csv: str,
    holdout_csv: str,
    n_folds: int = 5,
    holdout_size: float = 0.2,
    random_seed=82
):
    df = find_items(in_dir=in_dir)
    train_samples, holdout_samples = train_test_split(
        df['id'].unique(), test_size=holdout_size, random_state=random_seed, shuffle=True
    )

    # holdout
    df[df['id'].isin(holdout_samples)].to_csv(holdout_csv, index=False)

    # train-valid
    df = df[df['id'].isin(train_samples)]
    df['fold'] = -1
    folds = KFold(n_splits=n_folds, random_state=random_seed)
    for idx, (_, fold_samples) in enumerate(folds.split(train_samples)):
        df.loc[df['id'].isin(train_samples[fold_samples]), 'fold'] = idx

    df.to_csv(out_csv, index=False)


if __name__ == '__main__':
    fire.Fire({
        'infer': infer,
        'train-fold': train,
        'prepare-folds': prepare_folds,
        'distil-model': distil_model
    })
