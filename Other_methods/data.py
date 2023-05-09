import os
import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from PIL import Image


class DunhuangDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.root_dir = args.test_path if mode == 'test' and args.test_path else args.data_path

        # (*_mask.jpg, *_masked.jpg, *.jpg)
        self.data_dir = os.path.join(self.root_dir, mode)
        self.target_dir = os.path.join(self.root_dir, mode + '_gt')
        self.path_lists = [ (os.path.join(self.data_dir, img.split('.')+'_mask.jpg'), 
                             os.path.join(self.data_dir, img.split('.')+'_masked.jpg'),
                             os.path.join(self.target_dir, img)) for img in os.listdir(self.target_dir) ]
        if args.verbose:
            print("[Preprocess] Image mean: {}".format(args.img_mean))
            print("[Preprocess] Image std: {}".format(args.img_std))
        self.image_transforms = transforms.Compose([
            # transforms.ColorJitter(),
            transforms.ToTensor(), # (H,W,C)->(C,H,W) [0,255]->[0,1.0]
            transforms.Normalize(args.img_mean, args.img_std)
        ])
        self.mask_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.path_lists)

    def __getitem__(self, index):
        mask_name = self.path_lists[index][0]
        masked_name = self.path_lists[index][1]
        target_name = self.path_lists[index][2]

        if self.mode == 'train':
            masked, mask, gt = self.randomHorizontalFlip(Image.open(masked_name).convert('RGB'), Image.open(mask_name), Image.open(target_name).convert('RGB'))
        else:
            masked, mask, gt = Image.open(masked_name).convert('RGB'), Image.open(mask_name), Image.open(target_name).convert('RGB')
        
        ''' read image '''
        img = self.transforms(img)
        
        ''' read segmentation '''
        seg = (self.transforms_seg(seg)*255).squeeze().long()
        return img_name.split('/')[-1], img, seg

    def randomHorizontalFlip(self, img, seg):
        if random.random() > 0.5:
            img = TF.hflip(img)
            seg = TF.hflip(seg)
        return img, seg

class SegTestset(Dataset):
    def __init__(self, args):
        self.root_dir = os.path.join(args.test_path) 
        self.path_lists = [ os.path.join(self.root_dir,img) for img in os.listdir(self.root_dir) ]
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.img_mean, args.img_std)])

    def __len__(self):
        return len(self.path_lists)

    def __getitem__(self, index):
        img_name = self.path_lists[index]
        img = Image.open(img_name).convert('RGB')
        ''' read image '''
        img = self.transforms(img)
        
        return img_name.split('/')[-1], img
