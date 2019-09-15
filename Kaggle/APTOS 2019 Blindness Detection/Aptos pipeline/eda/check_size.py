import os
import json
import pandas as pd

from PIL import Image

from blindness.configs import dataset_map, diabetic_retinopathy_map



train_df = pd.read_csv(dataset_map['fold'])
test_df = pd.read_csv(dataset_map['test'])
diabetic_df = pd.read_csv(diabetic_retinopathy_map['train'], index_col='Unnamed: 0')


def build_get_size(split='train'):
    if split == 'train':
        image_dir = dataset_map['train_images']
    elif split == 'test':
        image_dir = dataset_map['test_images']
    else:
        image_dir = diabetic_retinopathy_map['train_images']
    def get_size(row):
        if 'image' not in row.keys():
            image_path = os.path.join(image_dir, '{}.png'.format(row['id_code']))
        else:
            image_path = os.path.join(image_dir, '{}.jpeg'.format(row['image']))

        image = Image.open(image_path)
        w, h = image.size
        row['w'] = w 
        row['h'] = h
        return row

    return get_size

train_df = train_df.apply(build_get_size('train'), axis=1)
test_df = test_df.apply(build_get_size('test'), axis=1)
diabetic_df = diabetic_df.apply(build_get_size(''), axis=1)

train_df["area"] = train_df["w"] * train_df["h"]
test_df["area"] = test_df["w"] * test_df["h"]
diabetic_df["area"] = diabetic_df["w"] * diabetic_df["h"]
print("Train avg w:{2:} h:{:}".format(train_df["w"].mean(), train_df["h"].mean()))
print("Test avg w:{} h:{}".format(test_df["w"].mean(), test_df["h"].mean()))
print("Diabetic avg w:{} h:{}".format(diabetic_df["w"].mean(), diabetic_df["h"].mean()))


train_l = train_df[train_df["area"] > train_df["area"].mean()]
train_s = train_df[train_df["area"] <= train_df["area"].mean()]

train_l = train_df[train_df["area"] > train_df["area"].mean()]

import pdb; pdb.set_trace()