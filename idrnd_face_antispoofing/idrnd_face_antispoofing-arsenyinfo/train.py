import os
from functools import partial
from typing import Callable, Optional

import numpy as np
import torch
import yaml
from fire import Fire
from glog import logger
from joblib import cpu_count
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.serialization import save
from torch.utils.data import DataLoader
from tqdm import tqdm

from convert import main as convert_model
from dataset import IdRndDataset
from models import get_baseline


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train: Optional[DataLoader],
                 val: Optional[DataLoader],
                 epochs: int = 200,
                 early_stop: int = 10,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 loss_fn: Optional[Callable] = None,
                 scheduler: Optional[ReduceLROnPlateau] = None,
                 device: str = 'cuda:0',
                 checkpoint: str = './model.pt',
                 work_dir: str = '.'
                 ):
        self.epochs = epochs
        self.early_stop = early_stop
        self.model = model.to(device)
        self.device = device
        self.train = train
        self.val = val
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler if scheduler is not None else ReduceLROnPlateau(optimizer=self.optimizer,
                                                                                   verbose=True)
        self.loss_fn = loss_fn if loss_fn is not None else F.cross_entropy
        self.current_metric = -np.inf
        self.last_improvement = 0
        self.work_dir = work_dir
        self.checkpoint = os.path.join(self.work_dir, checkpoint)
        self.tb_writer = SummaryWriter(logdir=work_dir)

    def to_device(self, x):
        if type(x) == dict:
            return {k: self.to_device(v) for k, v in x.items()}
        return x.to(self.device)

    def get_acc(self, outputs, y):
        a = outputs.argmax(dim=1)
        b = y.argmax(dim=1)
        return (a == b).cpu().numpy().mean()

    def _train_epoch(self, n_epoch):
        self.model.train(True)
        losses, accs = [], []

        train = tqdm(self.train, desc=f'training, epoch {n_epoch}')
        for i, inputs in enumerate(train):
            inputs = self.to_device(inputs)
            x = inputs['img']
            y = inputs['label']
            self.optimizer.zero_grad()
            outputs = self.model(x)

            loss = self.loss_fn(outputs, y)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            acc = self.get_acc(outputs, y)
            accs.append(acc)
            train.set_postfix(loss=f'{loss.item():.3f}', accuracy=f'{acc:.3f}')
            self.optimizer.step()

        train.close()
        return losses, accs

    def _val_epoch(self, n_epoch):
        self.model.train(False)

        val_losses, val_accs = [], []

        with torch.no_grad():
            val = tqdm(self.val, f'validating, epoch {n_epoch}')
            for i, inputs in enumerate(val):
                inputs = self.to_device(inputs)
                x = inputs['img']
                y = inputs['label']
                outputs = self.model(x)

                loss = self.loss_fn(outputs, y)
                val_losses.append(loss.item())
                acc = self.get_acc(outputs, y)
                val_accs.append(acc)
                val.set_postfix(loss=f'{loss.item():.3f}', accuracy=f'{acc:.3f}')

            val.close()
        return val_losses, val_accs

    def fit_one_epoch(self, n_epoch):
        losses, accs = self._train_epoch(n_epoch)
        val_losses, val_accs = self._val_epoch(n_epoch)

        train_loss = np.mean(losses)
        val_loss = np.mean(val_losses)
        train_acc = np.mean(accs)
        val_acc = np.mean(val_accs)
        msg = f'Epoch {n_epoch}: train loss is {train_loss:.3f}, train accuracy {train_acc:.3f}, ' \
              f'val loss {val_loss:.3f}, val accuracy {val_acc:.3f}'
        logger.info(msg)

        self.scheduler.step(metrics=val_loss, epoch=n_epoch)

        metric = val_acc
        if metric > self.current_metric:
            self.current_metric = metric
            self.last_improvement = n_epoch
            save(self.model, f=self.checkpoint)
            logger.info(f'Best model has been saved at {n_epoch}, accuracy is {metric:.4f}')
        else:
            if self.last_improvement + self.early_stop < n_epoch:
                return True, (train_loss, val_loss, train_acc, val_acc)

        return False, (train_loss, val_loss, train_acc, val_acc)

    def fit(self, start_epoch: int):
        for i in range(self.epochs):
            finished, (train_loss, val_loss, train_acc, val_acc) = self.fit_one_epoch(i + start_epoch)
            for name, scalar in (('train_loss', train_loss),
                                 ('val_loss', val_loss),
                                 ('train_acc', train_acc),
                                 ('val_acc', val_acc)):
                self.tb_writer.add_scalar(name, scalar, global_step=i)
            if finished:
                return i
        return self.epochs


def make_dataloaders(train_cfg, val_cfg, batch_size, multiprocessing=False):
    train = IdRndDataset.from_config(train_cfg)
    val = IdRndDataset.from_config(val_cfg)

    shared_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': cpu_count() if multiprocessing else 0}

    train = DataLoader(train, drop_last=True, **shared_params)
    val = DataLoader(val, drop_last=False, **shared_params)
    return train, val


def update_config(config, params):
    for k, v in params.items():
        *path, key = k.split('.')
        conf = config
        for p in path:
            if p not in conf:
                logger.error(f'Overwriting non-existing attribute {k} = {v}')
            conf = conf[p]
        logger.info(f'Overwriting {k} = {v} (was {conf.get(key)})')
        conf[key] = v


def soft_cross_entropy(inputs, target, weights):
    raw = -target * F.log_softmax(inputs, dim=1)
    if weights is not None:
        raw = raw * weights.view(1, -1)
    res = (torch.sum(raw, dim=1))
    return res.mean()


def fit(parallel=False, **kwargs):
    with open('config.yaml') as cfg:
        config = yaml.load(cfg)
    update_config(config, kwargs)
    work_dir = config['name']
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, 'config.yaml'), 'w') as out:
        yaml.dump(config, out)

    config['train']['salt'] = config['val']['salt'] = config['name']
    config['train']['n_fold'] = config['val']['n_fold'] = config['n_fold']

    train, val = make_dataloaders(config['train'], config['val'], config['batch_size'], multiprocessing=parallel)
    model = DataParallel(get_baseline(config['model']))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    trainer = Trainer(model=model,
                      train=train,
                      val=val,
                      work_dir=work_dir,
                      loss_fn=None,
                      optimizer=optimizer,
                      scheduler=ReduceLROnPlateau(factor=.2, patience=5, optimizer=optimizer),
                      device='cuda:0',
                      )

    stages = config['stages']
    epochs_completed = 0
    for i, stage in enumerate(stages):
        logger.info(f'Starting stage {i}')
        # ToDo: update train properties: mixup, crop type
        trainer.train.dataset.update_config(stage['train'])
        trainer.epochs = stage['epochs']
        weights = torch.from_numpy(np.array(stage['loss_weights'], dtype='float32')).to('cuda:0')
        trainer.loss_fn = partial(soft_cross_entropy,
                                  weights=weights)
        epochs_completed = trainer.fit(epochs_completed)

    convert_model(model_path=os.path.join(work_dir, 'model.pt'),
                  out_name=os.path.join(work_dir, f'{config["name"]}_{config["n_fold"]}.trcd'),
                  name=config['model']
                  )


if __name__ == '__main__':
    Fire(fit)
