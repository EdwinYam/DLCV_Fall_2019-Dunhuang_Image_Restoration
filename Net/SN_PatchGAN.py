import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SNConvUnit(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvUnit, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x



class SNDiscriminator(nn.Module):
    """ discriminator definition """
    def __init__(self):
        super(SNDiscriminator, self).__init__()
        self.discriminator_net = nn.Sequential(
            SNConvUnit(4, 64, 4, 2, 1),
            SNConvUnit(64, 128, 4, 2, 1),
            SNConvUnit(128, 256, 4, 2, 1),
            SNConvUnit(256, 256, 4, 2, 1),
        )

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        return x

if __name__ == "__main__":
    from torchsummary import summary
    net = SNDiscriminator()
    summary(net, input_size=(4, 256, 256), device='cpu')
