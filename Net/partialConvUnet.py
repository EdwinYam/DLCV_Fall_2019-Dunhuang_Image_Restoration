import torch
import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=no-member


class PartialConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super(PartialConv, self).__init__()

        self.input_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.mask_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        # set weight for mask_conv as 1.0
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        # avoid calculation for mask_conv
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, masked_img, mask):
        """ perform partical convolution
        Args:
            W: input_conv's weight
            M: Mask input(should be turned into multi channel(Ex:3*64*64))
            sum(M): ouput of mask_conv(M) because we initialize mask_conv.weight for 1.0
            X: Image input
            B: input_conv's bias

        Returns:
            x' = if sum(M) != 0 : W^T * (M * X) / sum(M) + B
                 else : 0
            m' = if sum(M) > 0 : 1
                 else : 0
        """
        output = self.input_conv(masked_img * mask)

        # create bias
        if self.input_conv.bias is not None:
            bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)

        else:
            bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        # change where Sum(M) == 0 to avoid divided by 0
        zero_index = output_mask == 0
        output_mask = output_mask.masked_fill_(zero_index, 1.0)

        # input_conv(x) = W^T * (x) + B => minus bias to create W^T * (x)
        output = (output - bias)/output_mask + bias
        output = output.masked_fill_(zero_index, 0.0)

        new_mask = torch.ones(output.shape).cuda()
        new_mask = new_mask.masked_fill(zero_index, 0.0)

        return output, new_mask


class PartialConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False,
                 batch_norm=False, non_linearity='relu'):
        super(PartialConvUnit, self).__init__()
        self.PConv = PartialConv(
            in_channels, out_channels, kernel_size, stride, padding, bias)

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

    def forward(self, img, mask):
        out, new_mask = self.PConv(img, mask)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        if self.act is not None:
            out = self.act(out)
        return out, new_mask


def upscaleConcat(img, mask, img_cat, mask_cat):
    x = F.interpolate(img, size=img_cat.shape[-2:], mode='nearest')
    mask_x = F.interpolate(mask, size=mask_cat.shape[-2:], mode='nearest')
    out = torch.cat((x, img_cat), dim=1)
    out_mask = torch.cat((mask_x, mask_cat), dim=1)
    return out, out_mask


class PartialConvUnet(nn.Module):

    def __init__(self):
        super(PartialConvUnet, self).__init__()
        # Encoder
        self.encoder1 = PartialConvUnit(3, 64, 7, 2, 3, batch_norm=False)
        self.encoder2 = PartialConvUnit(64, 128, 5, 2, 3, batch_norm=True)
        self.encoder3 = PartialConvUnit(128, 256, 5, 2, 2, batch_norm=True)
        self.encoder4 = PartialConvUnit(256, 512, 3, 2, 1, batch_norm=True)
        self.encoder5 = PartialConvUnit(512, 512, 3, 2, 1, batch_norm=True)
        self.encoder6 = PartialConvUnit(512, 512, 3, 2, 1, batch_norm=True)
        self.encoder7 = PartialConvUnit(512, 512, 3, 2, 1, batch_norm=True)
        self.encoder8 = PartialConvUnit(512, 512, 3, 2, 1, batch_norm=True)

        # Decoder
        self.decoder1 = PartialConvUnit(512+512, 512, 3, 1, 1,
                                        batch_norm=True, non_linearity='leaky_relu')
        self.decoder2 = PartialConvUnit(512+512, 512, 3, 1, 1,
                                        batch_norm=True, non_linearity='leaky_relu')
        self.decoder3 = PartialConvUnit(512+512, 512, 3, 1, 1,
                                        batch_norm=True, non_linearity='leaky_relu')
        self.decoder4 = PartialConvUnit(512+512, 512, 3, 1, 1,
                                        batch_norm=True, non_linearity='leaky_relu')
        self.decoder5 = PartialConvUnit(512+256, 256, 3, 1, 1,
                                        batch_norm=True, non_linearity='leaky_relu')
        self.decoder6 = PartialConvUnit(256+128, 128, 3, 1, 1,
                                        batch_norm=True, non_linearity='leaky_relu')
        self.decoder7 = PartialConvUnit(128+64, 64, 3, 1, 1,
                                        batch_norm=True, non_linearity='leaky_relu')
        self.decoder8 = PartialConvUnit(64+3, 3, 3, 1, 1, bias=True,
                                        batch_norm=False, non_linearity=None)

    def forward(self, input_img, input_mask):
        # encoder
        img_1, mask_1 = self.encoder1(input_img, input_mask)
        img_2, mask_2 = self.encoder2(img_1, mask_1)
        img_3, mask_3 = self.encoder3(img_2, mask_2)
        img_4, mask_4 = self.encoder4(img_3, mask_3)
        img_5, mask_5 = self.encoder5(img_4, mask_4)
        img_6, mask_6 = self.encoder6(img_5, mask_5)
        img_7, mask_7 = self.encoder7(img_6, mask_6)
        img_8, mask_8 = self.encoder8(img_7, mask_7)

        # decoder
        img_9, mask_9 = self.decoder1(
            *upscaleConcat(img_8, mask_8, img_7, mask_7))
        img_10, mask_10 = self.decoder2(
            *upscaleConcat(img_9, mask_9, img_6, mask_6))
        img_11, mask_11 = self.decoder3(
            *upscaleConcat(img_10, mask_10, img_5, mask_5))
        img_12, mask_12 = self.decoder4(
            *upscaleConcat(img_11, mask_11, img_4, mask_4))
        img_13, mask_13 = self.decoder5(
            *upscaleConcat(img_12, mask_12, img_3, mask_3))
        img_14, mask_14 = self.decoder6(
            *upscaleConcat(img_13, mask_13, img_2, mask_2))
        img_15, mask_15 = self.decoder7(
            *upscaleConcat(img_14, mask_14, img_1, mask_1))
        img_16, _ = self.decoder8(
            *upscaleConcat(img_15, mask_15, input_img, input_mask))

        return img_16

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                module.eval()

