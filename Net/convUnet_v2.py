import torch
import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=no-member


class ConvUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False,
                 batch_norm=False, non_linearity='relu'):
        super(ConvUnit, self).__init__()
        self.Conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=bias)

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
        out = self.Conv(img)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


def upscaleConcat(img, img_cat):
    x = F.interpolate(
        img, size=img_cat.shape[-2:], mode='nearest')
    out = torch.cat((x, img_cat), dim=1)
    return out


class ConvUnet(nn.Module):

    def __init__(self):
        super(ConvUnet, self).__init__()
        # Encoder
        self.encoder1 = ConvUnit(3, 64, 7, 2, 3, batch_norm=False)
        self.encoder2 = ConvUnit(64, 128, 5, 2, 2, batch_norm=True)
        self.encoder3 = ConvUnit(128, 256, 3, 2, 1, batch_norm=True)
        self.encoder4 = ConvUnit(256, 512, 3, 2, 1, batch_norm=True)
        self.encoder5 = ConvUnit(512, 512, 3, 2, 1, batch_norm=True)
        self.encoder6 = ConvUnit(512, 512, 3, 2, 1, batch_norm=True)
        self.encoder7 = ConvUnit(512, 512, 3, 2, 1, batch_norm=True)

        # dilated convolution
        self.encoder8 = ConvUnit(512, 512, 3, 1, 2, dilation=2, batch_norm=True)
        self.encoder9 = ConvUnit(512, 512, 3, 1, 4, dilation=4, batch_norm=True)
        self.encoder10 = ConvUnit(512, 512, 3, 1, 8, dilation=8, batch_norm=True)
        self.encoder11 = ConvUnit(512, 512, 3, 1, 16, dilation=16, batch_norm=True)

        # Decoder
        self.decoder1 = ConvUnit(512+512, 512, 3, 1, 1,
                                 batch_norm=True, non_linearity='leaky_relu')
        self.decoder2 = ConvUnit(512+512, 512, 3, 1, 1,
                                 batch_norm=True, non_linearity='leaky_relu')
        self.decoder3 = ConvUnit(512+512, 512, 3, 1, 1,
                                 batch_norm=True, non_linearity='leaky_relu')
        self.decoder4 = ConvUnit(512+512, 512, 3, 1, 1,
                                 batch_norm=True, non_linearity='leaky_relu')
        self.decoder5 = ConvUnit(512+256, 256, 3, 1, 1,
                                 batch_norm=True, non_linearity='leaky_relu')
        self.decoder6 = ConvUnit(256+128, 128, 3, 1, 1,
                                 batch_norm=True, non_linearity='leaky_relu')
        self.decoder7 = ConvUnit(128+64, 64, 3, 1, 1,
                                 batch_norm=True, non_linearity='leaky_relu')
        self.decoder8 = ConvUnit(64+3, 3, 3, 1, 1, bias=True,
                                 batch_norm=False, non_linearity=None)

    def forward(self, input_img):
        # encoder
        img_1 = self.encoder1(input_img)
        img_2 = self.encoder2(img_1)
        img_3 = self.encoder3(img_2)
        img_4 = self.encoder4(img_3)
        img_5 = self.encoder5(img_4)
        img_6 = self.encoder6(img_5)
        img_7 = self.encoder7(img_6)
        img_8 = self.encoder8(img_7)

        # decoder
        img_9 = self.decoder1(upscaleConcat(img_8, img_7))
        img_10 = self.decoder2(upscaleConcat(img_9, img_6))
        img_11 = self.decoder3(upscaleConcat(img_10, img_5))
        img_12 = self.decoder4(upscaleConcat(img_11, img_4))
        img_13 = self.decoder5(upscaleConcat(img_12, img_3))
        img_14 = self.decoder6(upscaleConcat(img_13, img_2))
        img_15 = self.decoder7(upscaleConcat(img_14, img_1))
        img_16 = self.decoder8(upscaleConcat(img_15, input_img))

        return img_16


if __name__ == "__main__":
    from torchsummary import summary
    net = ConvUnet()
    summary(net, input_size=[(3, 512, 512)], device='cpu')
