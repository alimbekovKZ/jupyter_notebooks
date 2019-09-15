import json
import numpy as np

from PIL import Image
from torchvision import transforms
from blindness.dataset import build_dataset
from blindness.transforms import build_transforms
from pathlib import Path

inv_normalize = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])
    ])

config_path = 'blindness/configs/base_aug.json'

with open(config_path, 'r') as f:
    cfg = json.loads(f.read())

train_transform  = build_transforms(cfg, split='train')
train_data = build_dataset(cfg, train_transform, split='train')

valid_transform  = build_transforms(cfg, split='valid')
valida_data = build_dataset(cfg, valid_transform, split='valid')

print(len(train_data.dataset))
print(len(valida_data.dataset))
print(len(train_data.dataset)+len(valida_data.dataset))
output_dir = Path('output/vis_input')
output_dir.mkdir(exist_ok=True, parents=True)

data_iterator = iter(valida_data)
for i in range(50):
    img, target, ids = next(data_iterator)
    print(img.shape, target[0].cpu())
    target = target[0].cpu()
    img = img[0]
    img = np.transpose(inv_normalize(img.cpu()).numpy()*255,(1,2,0))
    img = Image.fromarray(np.uint8(img))
    img.save(f'{output_dir}/{str(i).zfill(3)}_{target}.png')