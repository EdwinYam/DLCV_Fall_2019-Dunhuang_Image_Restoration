import os
import logging

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity
from matplotlib import pyplot as plt


# pylint: disable=no-member

""" setup GPU """
DEVICE = torch.device("cuda:0 " if torch.cuda.is_available() else "cpu")


class ConvUnetTrainer():

    def __init__(self, model, save_dir, loss, learning_rate):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model = model.to(DEVICE)
        self.criterion = loss()
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=learning_rate)
        self.writer = SummaryWriter(os.path.join(save_dir, 'train_info'))

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def resume_model(self, resume_path):
        resume = torch.load(resume_path)
        self.model.load_state_dict(resume)

    def train(self, train_loader, val_loader, total_epoch):
        iters = 0
        for epoch in range(1, total_epoch + 1):
            self.model.train()
            for idx, (masked_img, mask, target) in enumerate(train_loader):
                masked_img, mask, target = \
                    masked_img.to(DEVICE), mask.to(DEVICE), target.to(DEVICE)
                iters += 1
                train_info = 'Epoch: [{0}][{1}/{2}]'.format(
                    epoch, idx+1, len(train_loader))

                output = self.model(masked_img)

                ''' compute loss, backpropagation, update parameters '''
                loss = self.criterion(output, mask, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
                train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())
                print(train_info, end='\r')

            ''' save model '''
            print('\n')
            self.save_model(os.path.join(self.save_dir,
                                         'model_{}.pth.tar'.format(epoch)))
            mse, ssim = self.validation(val_loader)
            self.writer.add_scalar('mse', mse, epoch)
            self.writer.add_scalar('ssim', ssim, epoch)
            print("validation: mse = {}, ssim = {}".format(mse, ssim))

    def validation(self, val_loader):
        img_gt_paths = []
        img_pred_paths = []
        self.inference(val_loader, 'log')

        for i in range(100):
            img_name = "{}.jpg".format(401 + i)
            img_gt_paths.append(os.path.join('Data_Challenge2/test_gt', img_name))
            img_pred_paths.append(os.path.join('log', img_name))

        mse = 0
        ssim = 0
        for i in range(len(img_gt_paths)):
            img_1 = plt.imread(img_gt_paths[i])
            img_2 = plt.imread(img_pred_paths[i])
            mse += np.mean((img_1 - img_2) ** 2)
            ssim += structural_similarity(img_1, img_2, multichannel=True)
        return mse/(i+1), ssim/(i+1)


    def inference(self, val_loader, target_dir):
        self.model.eval()
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with torch.no_grad():
            for idx, (img, mask) in enumerate(val_loader):
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                output = self.model(img).squeeze()
                print("Processing {}...".format(idx+1), end='\r')

                inv_normalize = transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225]
                )
                output = inv_normalize(output)
                img = img.squeeze()
                img = inv_normalize(img)

                #  out_img = mask*img + (1-mask)*output
                out_img = output
                out_img = out_img.cpu().squeeze()
                save_image(out_img, target_dir+'/'+str(idx+401)+'.jpg')
