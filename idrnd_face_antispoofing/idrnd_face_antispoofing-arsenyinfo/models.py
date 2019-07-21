import torch
from pytorchcv.model_provider import get_model
from torch import nn
from torch.nn import functional as F


class Baseline(nn.Module):
    def __init__(self, model_name, dropout):
        super().__init__()
        model = get_model(model_name, pretrained='imagenet')
        self.backbone = model.features
        _, n_features, *_ = self.backbone(torch.rand(1, 3, 256, 256)).size()
        self.linear = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(n_features, 4),
                                    )

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.linear(x)
        return x


def get_baseline(name='densenet121'):
    if '.' in name:
        model_name, dropout = name.split('.')
        dropout = int(dropout) / 100
    else:
        model_name, dropout = name, .5
    return Baseline(model_name, dropout)
