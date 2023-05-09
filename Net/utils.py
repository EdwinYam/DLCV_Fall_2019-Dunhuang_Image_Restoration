import torch
import torch.nn.functional as F
import numpy as np
#pylint: disable=no-member


def upscaleConcat(img, img_cat):
    x = F.interpolate(img, size=img_cat.shape[-2:], mode='nearest')
    out = torch.cat((x, img_cat), dim=1)
    return out


