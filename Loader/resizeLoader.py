import os
import torch
import numpy as np
import random

from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset

# pylint: disable=no-member

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda:0 " if torch.cuda.is_available() else "cpu")


class loader(Dataset):

    def __init__(self, mode, dir, gated=False):
        ''' set up basic parameters for dataset '''
        self.mode = mode

        if mode == 'validation':
            mode = 'test'
        self.data_dir = os.path.join(dir, mode)
        self.data_name = os.listdir(self.data_dir)
        self.target_dir = os.path.join(dir, mode + '_gt')
        self.target_name = os.listdir(self.target_dir)
        self.gated = gated

        if self.mode == 'train':
            self.image_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ])
            self.mask_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.number = int(len(self.data_name)/2)
        ''' set up image trainsform '''
        print("Init dataloader in {} mode".format(self.mode))
        print("Number of {} data: {}".format(self.mode, self.number))

    def randomHorizontalFlip(self, img, mask, target):
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            target = TF.hflip(target)
        return img, mask, target

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        """ read image """

        masked_img = Image.open(
            os.path.join(self.data_dir, self.data_name[idx*2+1])).convert('RGB')
        mask = Image.open(
            os.path.join(self.data_dir, self.data_name[idx*2])).convert('RGB')

        if self.mode == 'train':
            target_img = Image.open(
                os.path.join(self.target_dir, self.target_name[idx]))

            masked_img, mask, target_img = self.randomHorizontalFlip(
                masked_img, mask, target_img)

        masked_img = self.image_transform(masked_img)

        if self.gated is not True:
            #  change mask to 0 and 1
            mask = self.mask_transform(mask)
            mask = torch.where(mask < 0.5, torch.zeros(1), torch.ones(1))
        else:
            #  change mask to 0 and 1
            mask = self.mask_transform(mask)
            mask = torch.where(mask < 0.5, torch.zeros(1), torch.ones(1))
            masked_img = torch.cat((masked_img, mask[0].unsqueeze(0)), dim=0)

        if self.mode == 'train':
            return masked_img, mask, self.image_transform(target_img)
        else:
            return masked_img, mask
