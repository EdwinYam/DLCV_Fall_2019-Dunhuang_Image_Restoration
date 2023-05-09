import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import upscaleConcat

# pylint: disable=no-member


class GatedConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False,
                 batch_norm=False, non_linearity='relu'):
        super(GatedConvUnit, self).__init__()

        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        self.mask_conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)

        self.gated = nn.Sigmoid()
        if (batch_norm == True):
            self.batch_norm = nn.BatchNorm2d(out_channels)
        else:
            self.batch_norm = None

        if (non_linearity == 'leaky_relu'):
            self.act = nn.LeakyReLU(0.2)
        elif (non_linearity == 'relu'):
            self.act = nn.ReLU()
        else:
            self.act = None

    def forward(self, img):
        x = self.conv2d(img)
        masked = self.mask_conv2d(img)
        if self.act is not None:
            x = self.act(x) * self.gated(masked)
        else:
            x = x * self.gated(masked)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        return x




class GatedConvUnet(nn.Module):

    def __init__(self):
        super(GatedConvUnet, self).__init__()
        # Encoder
        self.encoder1 = GatedConvUnit(4, 64, 5, 1, 1, batch_norm=False)
        self.encoder2 = GatedConvUnit(64, 128, 5, 2, 2, batch_norm=True)
        self.encoder3 = GatedConvUnit(128, 256, 5, 2, 2, batch_norm=True)
        self.encoder4 = GatedConvUnit(256, 512, 3, 2, 1, batch_norm=True)
        self.encoder5 = GatedConvUnit(512, 512, 3, 2, 1, batch_norm=True)
        self.encoder6 = GatedConvUnit(512, 512, 3, 2, 1, batch_norm=True)
        self.encoder7 = GatedConvUnit(512, 512, 3, 2, 1, batch_norm=True)

        # dilated convolution
        self.encoder8 = GatedConvUnit(512, 512, 3, 1, 2, dilation=2, batch_norm=True)
        self.encoder9 = GatedConvUnit(512, 512, 3, 1, 4, dilation=4, batch_norm=True)
        self.encoder10 = GatedConvUnit(512, 512, 3, 1, 8, dilation=8, batch_norm=True)
        self.encoder11 = GatedConvUnit(512, 512, 3, 1, 16, dilation=16, batch_norm=True)

        # Decoder
        self.decoder1 = GatedConvUnit(512+512, 512, 3, 1, 1,
                                      batch_norm=True, non_linearity='leaky_relu')
        self.decoder2 = GatedConvUnit(512+512, 512, 3, 1, 1,
                                      batch_norm=True, non_linearity='leaky_relu')
        self.decoder3 = GatedConvUnit(512+512, 512, 3, 1, 1,
                                      batch_norm=True, non_linearity='leaky_relu')
        self.decoder4 = GatedConvUnit(512+512, 512, 3, 1, 1,
                                      batch_norm=True, non_linearity='leaky_relu')
        self.decoder5 = GatedConvUnit(512+256, 256, 3, 1, 1,
                                      batch_norm=True, non_linearity='leaky_relu')
        self.decoder6 = GatedConvUnit(256+128, 128, 3, 1, 1,
                                      batch_norm=True, non_linearity='leaky_relu')
        self.decoder7 = GatedConvUnit(128+64, 64, 3, 1, 1,
                                      batch_norm=True, non_linearity='leaky_relu')
        self.decoder8 = GatedConvUnit(64+4, 3, 3, 1, 1, bias=True,
                                      batch_norm=False, non_linearity=None)
        self.act = nn.Tanh()

    def forward(self, input_img):
        # encoder
        img_1 = self.encoder1(input_img)
        img_2 = self.encoder2(img_1)
        img_3 = self.encoder3(img_2)
        img_4 = self.encoder4(img_3)
        img_5 = self.encoder5(img_4)
        img_6 = self.encoder6(img_5)
        img_7 = self.encoder7(img_6)

        # dilation convolution
        img_8 = self.encoder8(img_7)
        img_8 = self.encoder9(img_8)
        img_8 = self.encoder10(img_8)
        img_8 = self.encoder11(img_8)

        # decoder
        img_9 = self.decoder1(upscaleConcat(img_8, img_7))
        img_10 = self.decoder2(upscaleConcat(img_9, img_6))
        img_11 = self.decoder3(upscaleConcat(img_10, img_5))
        img_12 = self.decoder4(upscaleConcat(img_11, img_4))
        img_13 = self.decoder5(upscaleConcat(img_12, img_3))
        img_14 = self.decoder6(upscaleConcat(img_13, img_2))
        img_15 = self.decoder7(upscaleConcat(img_14, img_1))
        img_16 = self.decoder8(upscaleConcat(img_15, input_img))

        return self.act(img_16)

if __name__ == "__main__":
    from torchsummary import summary
    net = GatedConvUnet()
    summary(net, input_size=(4, 512, 512), device='cpu')
