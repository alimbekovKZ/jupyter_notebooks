import json

from blindness.models import build_model
from blindness.dataset import build_dataset
from blindness.transforms import build_transforms

config_path = './blindness/configs/base_regression.json'

with open(config_path, 'r') as f:
    cfg = json.loads(f.read())

model = build_model(cfg, 'cpu')

train_transform  = build_transforms(cfg, split='train')
train_data = build_dataset(cfg, train_transform, split='train')

img, target = train_data.dataset[0]
print(model.training)
print(model(img.unsqueeze(0), target.unsqueeze(0)))
model.eval()
print(model.training)
print(model(img.unsqueeze(0), target.unsqueeze(0)))