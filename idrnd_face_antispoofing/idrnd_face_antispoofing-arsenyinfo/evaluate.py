import os
from functools import lru_cache
from glob import glob
from time import time

import numpy as np
import pandas as pd
import torch
import yaml
from fire import Fire
from glog import logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IdRndDataset
from metrics import accuracy, spoof_metric
from pred import TestAntispoofDataset

pd.options.display.float_format = lambda x: '{:.0f}'.format(x) if round(x, 0) == x else '{:,.4f}'.format(x)


@lru_cache(maxsize=10)
def make_test_dataset(n_fold=1):
    with open('config.yaml') as cfg:
        config = yaml.load(cfg)['test']
        config['n_fold'] = n_fold

    dataset = IdRndDataset.from_config(config)
    files = dataset.imgs
    labels = dataset.labels

    paths = [{'id': labels[idx],
              'path': files[idx],
              'frame': np.float32(0),
              }
             for idx in range(len(files))]

    test_dataset = TestAntispoofDataset(paths=paths)
    return test_dataset


def parse_tb(path):
    _dir = os.path.dirname(path)
    files = sorted(glob(f'{_dir}/*tfevents*'))
    if not files:
        return {}
    # fixme: it should pick proper metric file
    ea = EventAccumulator(files[0])
    ea.Reload()

    res = {}
    for k in ('train_acc', 'train_loss', 'val_acc', 'val_loss'):
        try:
            vals = [x.value for x in ea.Scalars(k)]
            f = np.min if 'loss' in k else np.max
            res[k] = f(vals)
        except Exception:
            logger.exception(f'Can not process {k} from {files[0]}')
            res[k] = None
    return res


def evaluate(model, dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    labels, preds, times = [], [], []

    with torch.no_grad():
        for gt, frames, batch in tqdm(dataloader):
            batch = batch.to(device)
            t1 = time()
            logits = model(batch)
            t2 = time()
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            labels.extend(gt)
            preds.extend(proba)
            times.append(t2 - t1)

    preds, labels = map(np.array, (preds, labels))

    return {'test_accuracy': accuracy(pred=preds, labels=labels),
            'test_metric': spoof_metric(pred=preds, labels=labels),
            'inference_time': np.mean(times),
            }


def explore_models(models, batch_size):
    logger.info(f'There are {len(models)} models to evaluate')
    for m in models:
        t0 = time()
        model = torch.jit.load(m).to('cuda:0')
        t1 = time()
        d = {'load_time': t1 - t0,
             'name': m
             }
        *_, n_fold = os.path.basename(m).split('_')
        n_fold, _ = n_fold.split('.')
        n_fold = int(n_fold)
        metrics = evaluate(model=model,
                           batch_size=batch_size,
                           dataset=make_test_dataset(n_fold))

        # tb_data = parse_tb(m)
        d.update(metrics)
        yield d


def main(pattern="./**/*_?.trcd", batch_size=64):
    models = glob(pattern, recursive=False)
    data = []
    for x in explore_models(models, batch_size=batch_size):
        data.append(x)

    df = pd.DataFrame(data).sort_values('test_metric')
    df.to_csv('scores.csv', index=False)
    print(df.set_index('name'))

    df['model'] = df['name'].apply(lambda x: x.split('/')[1])
    df = df.groupby('model').agg(np.mean)
    print(df)


if __name__ == '__main__':
    Fire(main)
