
import os
import argparse
import random
import pandas as pd
import numpy as np
import tqdm

from PIL import Image
from collections import defaultdict, Counter
from .configs import dataset_map, diabetic_retinopathy_map

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
        row['area'] = w * h
        return row

    return get_size

def make_folds(n_folds, with_size=False):
    df = pd.read_csv(dataset_map['train'], engine='python')
    if with_size:
        df = df.apply(build_get_size('train'), axis=1)
        df['large'] = df['area']> df['area'].mean()

    cls_counts = Counter([classes for classes in df['diagnosis']])
    fold_cls_counts = defaultdict()
    for class_index in cls_counts.keys():
        fold_cls_counts[class_index] = np.zeros(n_folds, dtype=np.int)

    df['fold'] = -1
    pbar = tqdm.tqdm(total=len(df))

    def get_fold(row):
        class_index = row['diagnosis']
        counts = fold_cls_counts[class_index]
        fold = np.argmin(counts)
        counts[fold] += 1
        fold_cls_counts[class_index] = counts
        row['fold']=fold
        pbar.update()
        return row
    
    df = df.apply(get_fold, axis=1)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--with_size', action='store_true')
    args = parser.parse_args()
    df = make_folds(n_folds=args.n_folds, with_size=args.with_size)
    df.to_csv('folds.csv', index=None)


if __name__ == '__main__':
    main()
