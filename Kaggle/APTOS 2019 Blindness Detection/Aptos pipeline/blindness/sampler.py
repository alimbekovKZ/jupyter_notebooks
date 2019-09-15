import torch

import numpy as np
import pandas as pd

from torch.utils.data import WeightedRandomSampler
from torch.utils.data.sampler import Sampler

class RatioSampler(Sampler):
    def __init__(self,
                 df_list,
                 class_ratio=None,
                 dataset_ratio=None,
                 replacement=False,
                 num_samples=False):


        self.epoch = 0
        self.replacement = replacement
        self.num_samples = num_samples

        self.df_length = [len(df) for df in df_list]
        _df_list = []
        for df in df_list:
            if 'diagnosis' in df.keys():
                _df_list.append(df[['diagnosis']])
            elif 'level' in df.keys():
                sub_df = df[['level']].copy()
                sub_df.columns = ['diagnosis']
                _df_list.append(sub_df)
        self.df = pd.concat(_df_list)

        if class_ratio is not None: 
            class_ratio = np.array(class_ratio)
            self.class_ratio = class_ratio/sum(class_ratio)

        if dataset_ratio is not None: 
            dataset_ratio = np.array(dataset_ratio)
            self.dataset_ratio = dataset_ratio/sum(dataset_ratio)
        else:
            self.dataset_ratio = np.array([1])

    def __iter__(self):
        return iter(self._fixed_ratio_indices())

    def __len__(self):
        return len(self._fixed_ratio_indices())
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def _get_fixed_class_ratio_weights(self, i):
        start_index = np.sum(self.df_length[:i])
        df = self.df[
            int(start_index):
            int(start_index+self.df_length[i])
        ]
        class_ratio = df['diagnosis'].value_counts()/len(df)

        df['weight'] = 0
        for k, v in class_ratio.items():
            df.loc[df['diagnosis'] == k, 'weight'] = v

        return torch.tensor(df['weight'].values)

    def _fixed_ratio_indices(self):
        num_to_keep_ratio = np.amin([l/r for l, r in zip(self.df_length, self.dataset_ratio)])
        sample_sizes_scaled = self.dataset_ratio*np.int(num_to_keep_ratio)

        index_list = []
        for i in range(len(self.df_length)):
            # add class weight
            if self.class_ratio is None:
                sub_df_index = torch.randperm(self.df_length[i])
            else:
                sub_weights = self._get_fixed_class_ratio_weights(i)
                sub_df_index = torch.multinomial(sub_weights, self.df_length[i], self.replacement)

            randomized_index = sub_df_index + np.sum(self.df_length[:i]) # add for previouse indexes
            index_list = index_list + randomized_index[:np.int(sample_sizes_scaled[i])].tolist()

        index_list_shuffled = [index_list[i] for i in torch.randperm(len(index_list))]
        return index_list_shuffled

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight

def get_sampler(use_class_ratio, use_dataset_ratio, class_ratio, dataset_ratio, df_list):
    if use_class_ratio: assert len(class_ratio) == 5
    else: class_ratio = None

    if use_dataset_ratio: assert len(dataset_ratio) == len(df_list)
    else: dataset_ratio = None

    sampler = RatioSampler(df_list, class_ratio, dataset_ratio)
    return sampler
