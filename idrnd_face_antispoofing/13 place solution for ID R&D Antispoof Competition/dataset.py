import torch
from torch.utils.data import Dataset
import cv2
import os


class AntispoofDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        image_info = self.paths[index]

        imgs = self.load_images(image_info['path'])

        return imgs, image_info['label']

    def __len__(self):
        return len(self.paths)

    def load_images(self, path):
        frames = sorted(os.listdir(path))
        imgs = []
        for p in frames:
            img = cv2.imread(os.path.join(path, p))
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        return torch.stack(imgs)
