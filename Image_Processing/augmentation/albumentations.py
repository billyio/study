ref
# https://github.com/albu/albumentations

import numpy as np
import cv2
from matplotlib import pyplot as plt
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)

aug = Compose([
    OneOf([RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=0.5),
          PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)], p=1),    
    VerticalFlip(p=0.5),              
    RandomRotate90(p=0.5),
    OneOf([
        ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(p=0.5),
        OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)                  
        ], p=0.8)])


from albumentations import *
from albumentations.pytorch import ToTensor

train_transform = Compose([
    Rotate(45, p=0.666),
    AverageCrop(resolution, resolution),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Transpose(p=0.5),
    ToTensor()
])

valid_transform = Compose([
    CenterCrop(resolution, resolution),
    ToTensor()
])