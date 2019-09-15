import cv2
import random
import numpy as np

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, RandomApply, Resize, CenterCrop
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomGrayscale

from .autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy


class CropBlack(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            origin_w, origin_h = img.size
            cv_img = np.array(img)
            gray = cv2.cvtColor(cv_img,cv2.COLOR_RGB2GRAY)
            _,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
            area = (
                x, y, x+w, y+h
            )
            if w > origin_w/2 or h > origin_h/2:
                img = img.crop(area)
        return img


class RandomRotate(object):
    def __init__(self, p):
        self.p = p
        self.degrees = [0, 90, 180, 270]
    def __call__(self, img):
        if random.random() < self.p:
            degree = random.choice(self.degrees)
            img =  img.rotate(degree)
        return img


class BenGrahamAug(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            w,h = img.size
            cv_img = np.array(img)
            cv_img = cv2.addWeighted (cv_img, 4, cv2.GaussianBlur(cv_img , (0,0) , w/10) ,-4 ,128)
            img = Image.fromarray(cv_img)
        return img


def get_transforms(transforms_list,
                   width,
                   height,
                   is_train):
    transforms = []
    for transform in transforms_list:
        if transform == 'random_resized_crop':
            scale = (0.8, 1.2) if is_train else (1.0, 1.0)
            ratio = (1.0, 1.0) if is_train else (1.0, 1.0)
            transforms.append(
                RandomResizedCrop(
                    (width, height),
                    scale=scale,
                    ratio=ratio,
                )
                
            )
        elif transform == 'center_crop' :
            transforms.append(
                CenterCrop((700, 700))
            )
        elif transform == 'resize':
            transforms.append(
                Resize(
                    (width, height)
                )
            )
        elif transform == 'resize':
            transforms.append(
                Resize(
                    (width, height)
                )
            )
        elif transform == 'crop_black': # crop_black은 첫번째로 넣어줘야함.
            p = 1.0 if is_train else 1.0
            transforms.append(CropBlack(p))
        elif transform == 'random_rotate':
            p = 0.5 if is_train else 0.25
            transforms.append(RandomRotate(p))
        elif transform == 'random_vertical_flip':
            p = 0.5 if is_train else 0.25
            transforms.append(RandomVerticalFlip(p))
        elif transform == 'random_horizontal_flip':
            p = 0.5 if is_train else 0.25
            transforms.append(RandomHorizontalFlip(p))
        elif transform == 'random_color_jitter':
            brightness = 0.1 if is_train else 0.0
            contrast = 0.1 if is_train else 0.0
            transforms.append(ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=0,
                hue=0,
            ))
        elif transform == 'random_grayscale':
            p = 0.5 if is_train else 0.25
            transforms.append(RandomGrayscale(p))
        elif transform == 'ben_graham':
            p = 1 if is_train else 1
            transforms.append(BenGrahamAug(p))
        elif transform == 'imagenet_poilcy':
            transforms.append(ImageNetPolicy())
        elif transform == 'cifar_policy':
            transforms.append(CIFAR10Policy())
        elif transform == 'svhn_policy':
            transform.append(SVHNPolicy())
        else:
            print(transform)
            raise NotImplementedError
    return transforms

def build_transforms(cfg, split='train'):
    is_train = split == 'train'
    input_cfg = cfg['input']
    width = input_cfg['width']
    height = input_cfg['height']

    if split == 'train':
        transform_list = input_cfg['transforms']
    else:
        transform_list = input_cfg['test_transforms']

    transforms = get_transforms(
        input_cfg['transforms'],
        width,
        height,
        is_train)

    transforms += [
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return Compose(transforms)