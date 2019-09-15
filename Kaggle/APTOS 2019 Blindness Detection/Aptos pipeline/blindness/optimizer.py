import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def build_optimizer(cfg, model, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer

def build_scheduler(cfg, optimizer):
    train_cfg = cfg['train_param']
    if train_cfg['scheduler'] == 'cosine':
        t_max = train_cfg['cosine']['t_max']
        eta_min = train_cfg['cosine']['eta_min']
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif train_cfg['scheduler'] == 'steplr':
        step_size = train_cfg['steplr']['step_size']
        gamma = train_cfg['steplr']['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        return
    return scheduler