import os
import torch
import numpy as np

from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset

# pylint: disable=no-member

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda:0 " if torch.cuda.is_available() else "cpu")




class loader(Dataset):
    def __init__(self, mode, dir, gated=False):
        if mode == 'train':
            prefix = 'train'
        else:
            prefix = 'test'
        self.root_dir = dir
        self.mode = mode
        self.gated = gated

        # (*_mask.jpg, *_masked.jpg, *.jpg)
        self.data_dir = os.path.join(self.root_dir, prefix)
        self.target_dir = os.path.join(self.root_dir, prefix + '_gt')
        self.img_indices = [img.split('/')[-1].split('_')[0]
                            for img in os.listdir(self.data_dir) if '_mask.jpg' in img]
        self.path_lists = [(os.path.join(self.data_dir, index+'_mask.jpg'),
                            os.path.join(self.target_dir, index+'.jpg')) for index in self.img_indices]

        self.crop_shape = (400, 400)
        self.mask_threshold = 1600*1
        self.image_transforms = transforms.Compose([
            # transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.crop_shape),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W) [0,255]->[0,1.0]
        ])
        self.mask_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((0, 180)),
            transforms.RandomCrop(self.crop_shape),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.path_lists)

    def __getitem__(self, index):

        # mask_path = self.path_lists[index%len(self.path_lists)][0]
        # target_path = self.path_lists[index//len(self.path_lists)][1]
        mask_path = self.path_lists[random.randint(
            0, len(self.path_lists)-1)][0]
        target_path = self.path_lists[index][1]

        mask, gt = Image.open(mask_path).convert(
            'RGB'), Image.open(target_path).convert('RGB')

        ''' read mask '''
        mask = self.mask_transforms(mask)
        mask = torch.where(mask < 0.5, torch.zeros(1), torch.ones(1))
        while mask.sum() < self.mask_threshold:
            mask_path = self.path_lists[random.randint(
                0, len(self.path_lists)-1)][0]
            mask = self.mask_transforms(Image.open(mask_path))
            mask = torch.where(mask < 0.5, torch.zeros(1), torch.ones(1))

        ''' read ground truth image'''
        gt = self.image_transforms(gt)

        ''' read masked image '''
        masked = torch.where(mask > 0.5, gt, torch.ones(1))
        masked = self.normalize(masked)
        if self.gated is True:
            masked = torch.cat((masked, mask[0].unsqueeze(0)), dim=0)
        # print(masked.shape, mask.shape, gt.shape)
        return masked, mask, gt
