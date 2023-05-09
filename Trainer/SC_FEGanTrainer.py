import os
import logging

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity
from matplotlib import pyplot as plt


# pylint: disable=no-member

""" setup GPU """
DEVICE = torch.device("cuda:0 " if torch.cuda.is_available() else "cpu")


class SC_FEGanTrainer():

    def __init__(self, generator, discriminator, save_dir,
                 impaint_loss, gen_loss, dis_loss, learning_rate):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.generator = generator.to(DEVICE)
        self.discriminator = discriminator.to(DEVICE)
        self.impaint_loss = impaint_loss()
        self.generator_loss = gen_loss()
        self.discriminator_loss = dis_loss()
        self.generator_opt = optim.Adam(params=self.generator.parameters(),
                                        lr=learning_rate)
        self.discriminator_opt = optim.Adam(params=self.discriminator.parameters(),
                                            lr=4*learning_rate)
        self.writer = SummaryWriter(os.path.join(save_dir, 'train_info'))

    def save_model(self, save_path):
        torch.save(self.generator.state_dict(), save_path)

    def resume_model(self, resume_path):
        resume = torch.load(resume_path)
        self.generator.load_state_dict(resume)

    def gradient_penalty(self, true_img, fake_img):
        batch_size = true_img.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(true_img).to(DEVICE)
        interpolated = alpha * true_img + (1 - alpha) * fake_img
        interpolated = interpolated.to(DEVICE)

        preds = self.discriminator(interpolated)

        gradients = grad(outputs=preds, inputs=interpolated,
                        grad_outputs=torch.ones_like(preds).to(DEVICE),
                        retain_graph=True, create_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0),  -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1)**2).mean()
        return gradient_penalty

    def train(self, train_loader, val_loader, total_epoch):
        iters = 0
        for epoch in range(1, total_epoch + 1):
            self.generator.train()
            for idx, (masked_img, mask, target) in enumerate(train_loader):
                masked_img, mask, target = \
                    masked_img.to(DEVICE), mask.to(DEVICE), target.to(DEVICE)
                iters += 1
                train_info = 'Epoch: [{0}][{1}/{2}]'.format(
                    epoch, idx+1, len(train_loader))

                output = self.generator(masked_img)
                out_img = mask*target + (1-mask)*output

                # update discriminator
                self.discriminator_opt.zero_grad()
                self.generator_opt.zero_grad()

                true_img = torch.cat([target, mask[:,0,:,:].unsqueeze(1)], dim=1)
                fake_img = torch.cat([out_img, mask[:,0,:,:].unsqueeze(1)], dim=1)
                true_fake_img = torch.cat([true_img, fake_img], dim=0)

                true_fake_pred = self.discriminator(true_fake_img)
                true_pred, fake_pred = torch.chunk(true_fake_pred, 2, dim=0)
                dis_loss = self.discriminator_loss(true_pred, fake_pred)
                gp_loss = self.gradient_penalty(true_img, fake_img)
                dis_loss = dis_loss + 10*gp_loss
                dis_loss.backward(retain_graph=True)
                self.discriminator_opt.step()

                # update generator
                self.discriminator_opt.zero_grad()
                self.generator_opt.zero_grad()

                fake_pred = self.discriminator(fake_img)
                true_pred = self.discriminator(true_img)
                impaint_loss = self.impaint_loss(output, mask, target)
                gen_loss = self.generator_loss(fake_pred, true_pred)
                gen_loss = impaint_loss + 0.001*gen_loss
                gen_loss.backward()
                self.generator_opt.step()

                self.writer.add_scalar('loss', gen_loss.data.cpu().numpy(), iters)
                train_info += 'impaint_loss: {:.4f}, gen_loss: {:.4f}, dis_loss: {:.4f}, gp: {:.4f}'\
                    .format(impaint_loss.detach().cpu().numpy(), gen_loss.data.cpu().numpy(),
                            dis_loss.data.cpu().numpy(), gp_loss.data.cpu().numpy())
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
        self.generator.eval()
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with torch.no_grad():
            for idx, (img, mask) in enumerate(val_loader):
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                output = self.generator(img).squeeze()
                print("Processing {}...".format(idx+1), end='\r')

                inv_normalize = transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225]
                )
                output = inv_normalize(output)
                img = img.squeeze()
                img = inv_normalize(img)

                out_img = mask*img[:-1] + (1-mask)*output
                #  out_img = output
                out_img = out_img.cpu().squeeze()
                save_image(out_img, target_dir+'/'+str(idx+401)+'.jpg')
