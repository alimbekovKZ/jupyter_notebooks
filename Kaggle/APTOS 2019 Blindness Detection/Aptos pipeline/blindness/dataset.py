import cv2
import torch
import pandas as pd

from glob import glob
from PIL import Image
from sklearn.utils import resample

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from .configs import dataset_map, diabetic_retinopathy_map
from .sampler import get_sampler

class BlindDataset(Dataset):
    """
    fold가 구분되어 train 또는 test로 split된 dataframe을 입력으로 받음.
    root: dataset root dir
    df: label을 포함한 dataframe
    num_class : label의 갯수
    output : label의 형태로 classification일 경우 one-hot 형태, regression일 경우 0, 1, 2, 3, 4 형태
    """
    def __init__(self,
                 image_dir,
                 df,
                 transforms,
                 num_class,
                 is_test,
                 num_tta=0):
        super().__init__()
        self.is_test = is_test
        self.image_dir = image_dir
        self.df = df
        self.transforms = transforms
        self.num_class = num_class
        self.num_tta = num_tta

    def __len__(self):
        if self.num_tta != 0:
            return len(self.df) * self.num_tta
        return len(self.df)

    def get_img(self, index):
        if 'id_code' not in self.df.keys():
            img_name = self.df.iloc[index]['image']
            img_path = self.image_dir + '/' + img_name + '.jpeg'
        else:
            img_name = self.df.iloc[index]['id_code']
            img_path = self.image_dir + '/' + img_name + '.png'
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        return image

    def __getitem__(self, index):
        if self.num_tta > 0:
            index = index % len(self.df) # find original index
        item = self.df.iloc[index]
        image = self.get_img(index)

        if self.transforms is not None:
            image = self.transforms(image)
        if 'id_code' not in self.df.keys():
            if not self.is_test:
                target = torch.tensor(item['level'])
            ids = item['image']
        else:
            if not self.is_test:
                target = torch.tensor(item['diagnosis'])
            ids = item['id_code']
        if self.is_test:
            return image, ids
        return image, target, ids


def upsampling(df):
    if 'diagnosis' in df.keys():
        gb = df.groupby(['diagnosis'])
    elif 'level' in df.keys():
        gb = df.groupby(['level'])
    groups = gb.groups
    max_count = max([len(gb.get_group(k)) for k in groups.keys()])
    
    df_list = list()
    for key in groups.keys():
        sub_df = resample(gb.get_group(key), n_samples=max_count, random_state=0)
        df_list.append(sub_df)

    return pd.concat(df_list)


def build_dataset(cfg, transforms,  split='train', num_tta=0):
    assert split in ['train', 'valid', 'test']
    dataset_config = cfg['dataset']

    num_class = dataset_config['num_class']
    fold = dataset_config['fold']
    batch_size = dataset_config['batch_size']
    num_workers = dataset_config['num_workers']
    use_upsampling = dataset_config['upsampling']
    is_test = split == 'test'

    if split == 'test':
        df = pd.read_csv(dataset_map['test'])
        image_dir = dataset_map['test_images']
    else:
        df = pd.read_csv(dataset_map['fold'])
        image_dir = dataset_map['train_images']
    
    if dataset_config['use_original']:
        if split == 'train':
            df = df[df['fold'] != fold]
            if use_upsampling: df = upsampling(df)
        elif split == 'valid':
            df = df[df['fold'] == fold]

    if split == 'valid':
        if not dataset_config['valid_with_both']:
            if dataset_config['valid_with_large']:
                df = df[df['large']]
            elif dataset_config['valid_with_small']:
                df = df[~df['large']]
    sampler_df = [df]
    dataset = BlindDataset(
        image_dir=image_dir,
        df=df,
        transforms=transforms,
        num_class=num_class,
        is_test=is_test,
        num_tta=num_tta
    )

    if split == 'train' and dataset_config['use_diabetic_retinopathy']:
        diabetic_df = pd.read_csv(diabetic_retinopathy_map['train'], index_col='Unnamed: 0')
        del diabetic_df['Unnamed: 0.1']
        if use_upsampling: diabetic_df = upsampling(diabetic_df) # up sampling for diabetic
        diabetic_dataset = BlindDataset(
            image_dir=diabetic_retinopathy_map['train_images'],
            df=diabetic_df,
            transforms=transforms,
            num_class=num_class,
            is_test=is_test
        )

        if not dataset_config['use_original']:
            dataset = diabetic_dataset
            sampler_df = [diabetic_df]
        else:
            sampler_df += [diabetic_df]
            dataset = ConcatDataset([dataset, diabetic_dataset])

    if split == 'train' and \
        (dataset_config['use_class_ratio'] or dataset_config['use_dataset_ratio']):
        sampler = get_sampler(
            dataset_config['use_class_ratio'],
            dataset_config['use_dataset_ratio'],
            dataset_config['class_ratio'],
            dataset_config['dataset_ratio'],
            sampler_df
        )
    else:
        sampler = None

    data_loader = DataLoader(
        dataset,
        shuffle=True if sampler is None else False,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
    )
    return data_loader