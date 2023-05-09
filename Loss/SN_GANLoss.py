import torch
import numpy as np
import torch.nn.functional as F

# pylint: disable=no-member

class SNDisLoss(torch.nn.Module):
    """
    The loss for sngan discriminator
    """
    def __init__(self, weight=1):
        super(SNDisLoss, self).__init__()

    def forward(self, pos, neg):
        return torch.mean(1-pos) + torch.mean(1+neg)

class SNGenLoss(torch.nn.Module):
    """
    The loss for sngan generator
    """
    def __init__(self, weight=1):
        super(SNGenLoss, self).__init__()

    def forward(self, fake, true):
        return -torch.mean(fake) + torch.mean(true**2)

