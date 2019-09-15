import torch

import torchvision.models as models

from torch import nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn


from .efficientnet_pytorch.model import EfficientNet

model_map = {
    'densenet121':models.densenet121,
    "resnet50": models.resnet50,
    'efficientnet-b0': EfficientNet.from_name('efficientnet-b0'),
    'efficientnet-b1': EfficientNet.from_name('efficientnet-b1'),
    'efficientnet-b2': EfficientNet.from_name('efficientnet-b2'),
    'efficientnet-b3': EfficientNet.from_name('efficientnet-b3'),
    'efficientnet-b4': EfficientNet.from_name('efficientnet-b4'),
    'efficientnet-b5': EfficientNet.from_name('efficientnet-b5'),
    'efficientnet-b6': EfficientNet.from_name('efficientnet-b6'),
    'efficientnet-b7': EfficientNet.from_name('efficientnet-b7'),
}

class Model(nn.Module):
    def __init__(self, cfg, device):
        super(Model, self).__init__()

        self.device = device
        self.mode = cfg['dataset']['method']
        weights_path = cfg['model']['weight_path']
        
        if self.mode == 'classification':
            out_feature=cfg['dataset']['num_class']
            self.criterion = nn.CrossEntropyLoss()
        elif self.mode == 'regression':
            out_feature=1
            self.criterion = nn.MSELoss()
        else: 
            raise ValueError

        args = (
            cfg['model']['pretrained'],
            cfg['dataset']['num_class']
        )
        if cfg['model']['name'].split('-')[0] == 'efficientnet':
            self.backbone = model_map[cfg['model']['name']]
        else:
            #print(cfg['model']['name'])
            #self.backbone = model_map[cfg['model']['name']](pretrained=True)
            self.backbone = models.densenet201(pretrained=False)
            #print(self.backbone)
            
        if weights_path is not None:
            self.backbone.load_state_dict(torch.load(weights_path))

        if cfg['model']['name'].split('-')[0] == 'efficientnet':
            in_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=512, bias=True),
                nn.BatchNorm1d(num_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=out_feature, bias=True),
            )
        else:
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features=1920, out_features=1024, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=out_feature, bias=True),
            )
            print(self.backbone)

    def loss(self, pred, label):
        if self.mode == 'regression':
            label = label.float().unsqueeze(0).transpose(0,1)
            
        loss = self.criterion(pred, label)
        return loss

    def forward(self, input_img, target=None, validate=False):
        input_img = input_img.to(self.device)
        if target is not None:
            target = target.to(self.device)
        pred = self.backbone(input_img)

        if self.training or validate: # training time return loss
            loss = self.loss(pred, target)

        if self.training:
            return loss
        elif validate:
            if self.mode == 'classification':
                pred =  F.softmax(pred,dim=1)
            return pred, loss
        else:
            if self.mode == 'classification':
                pred =  F.softmax(pred)
            return pred


def build_model(cfg, device):
    model = Model(cfg, device)
    model.to(device)
    cudnn.benchmark = True
    
    return model
