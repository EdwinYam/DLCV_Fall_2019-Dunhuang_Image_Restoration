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

        self.data_dir = os.path.join(dir, mode)
        self.data_name = os.listdir(self.data_dir)
        self.target_dir = os.path.join(dir, mode + '_gt')
        self.target_name = os.listdir(self.target_dir)
        self.gated = gated

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
        if self.mode == 'train':
            print("cropping image...")
            self.crop_image()

    def randomHorizontalFlip(self, img, mask, target):
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            target = TF.hflip(target)
        return img, mask, target

    def crop_image(self):
        self.image = []
        self.mask = []
        self.target = []
        for idx in range(self.number):
            mask = Image.open(
                os.path.join(self.data_dir, self.data_name[idx*2])).convert('RGB')

            masked_img = Image.open(
                os.path.join(self.data_dir, self.data_name[idx*2+1])).convert('RGB')

            target_img = Image.open(
                os.path.join(self.target_dir, self.target_name[idx])).convert('RGB')

            crop_mask = \
                transforms.functional.five_crop(mask, (256, 256))
            crop_image = \
                transforms.functional.five_crop(masked_img, (256, 256))
            crop_target = \
                transforms.functional.five_crop(target_img, (256, 256))
            for i in crop_mask:
                self.mask.append(i)
            for i in crop_image:
                self.image.append(i)
            for i in crop_target:
                self.target.append(i)
        self.number *= 5

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        """ read image """


        if self.mode == 'train':
            masked_img, mask, target_img = self.randomHorizontalFlip(
                self.image[idx], self.mask[idx], self.target[idx])

            masked_img = self.image_transform(masked_img)
            mask = self.mask_transform(mask)
            mask = torch.where(mask < 0.5, torch.zeros(1), torch.ones(1))
            target_img = self.image_transform(target_img)
            if self.gated is True:
                masked_img = torch.cat((masked_img, mask[0].unsqueeze(0)), dim=0)
                return masked_img, mask, target_img
            else:
                return masked_img, mask, target_img

        else:
            mask = Image.open(
                os.path.join(self.data_dir, self.data_name[idx*2])).convert('RGB')
            mask = self.mask_transform(mask)

            masked_img = Image.open(
                os.path.join(self.data_dir, self.data_name[idx*2+1]))

            masked_img = self.image_transform(masked_img)
            if self.gated is True:
                masked_img = torch.cat((masked_img, mask[0].unsqueeze(0)), dim=0)
            return masked_img, mask
