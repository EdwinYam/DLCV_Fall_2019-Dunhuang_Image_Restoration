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

    def __init__(self, dir, gated=False):
        ''' set up basic parameters for dataset '''
        self.img_path = []
        self.number = 100
        self.mask_path = []
        for i in range(self.number):
            self.img_path.append(os.path.join(
                dir,  str(i+400+1) + '_masked.jpg'))
            self.mask_path.append(os.path.join(
                dir,  str(i+400+1) + '_mask.jpg'))

        self.gated = gated

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        ''' set up image trainsform '''
        print("Init dataloader in {} mode".format('test'))
        print("Number of {} data: {}".format('test', self.number))

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        """ read image """

        masked_img = Image.open(self.img_path[idx]).convert('RGB')
        mask = Image.open(self.mask_path[idx]).convert('RGB')

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

        return masked_img, mask
