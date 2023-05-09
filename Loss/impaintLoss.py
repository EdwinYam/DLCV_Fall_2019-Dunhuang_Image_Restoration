import torch
import torch.nn as nn
from torchvision import models

# pylint: disable=no-member


class ImpaintLoss(nn.Module):

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        vgg.eval()
        self.extractor_1 = nn.Sequential(*vgg.features[:5]).cuda()
        self.extractor_2 = nn.Sequential(*vgg.features[5:10]).cuda()
        self.extractor_3 = nn.Sequential(*vgg.features[10:17]).cuda()
        for param in [self.extractor_1, self.extractor_2, self.extractor_3]:
            param.requires_grad = False
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def pixel_loss(self, mask, Iout, Igt):
        """
        pixel_loss = 6*l1loss[(1-mask) * (Iout-Igt)] + l1loss[mask * (Iout-Igt)]

        Args:
            mask (torch.Tensors): input mask
            Iout (torch.Tensors): model output
            Igt (torch.Tensors): ground truth target
        """
        loss_hole = self.l1loss((1-mask)*Iout, (1-mask)*Igt)
        loss_valid = self.l1loss(mask*Iout, mask*Igt)

        return 6*loss_hole + loss_valid

    def feature_extract(self, img):
        """
        extract feature using vgg16
        """
        feature = []
        feature_1 = self.extractor_1(img)
        feature.append(feature_1)
        feature_2 = self.extractor_2(feature_1)
        feature.append(feature_2)
        feature_3 = self.extractor_3(feature_2)
        feature.append(feature_3)
        return feature

    def gram_matrix(self, feat):
        # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
        (b, ch, h, w) = feat.size()
        feat = feat.view(b, ch, h * w)
        feat_t = feat.transpose(1, 2)
        gram = torch.bmm(feat, feat_t) / (ch * h * w)
        return gram

    def perceptual_loss(self, Iout_feature, Igt_feature, Icomp_feature):
        """
        perceptual_loss = Σ(l1loss(Ψ(Iout)-Ψ(Igt))) + Σ(l1loss(Ψ(Icomp)-Ψ(Igt)))
        Ψ: vgg feature extractor

        Args:
            Iout_feature (torch.Tensors): vgg extracted model output
            Igt_feature (torch.Tensors): vgg extracted ground truth target
            Icomp_feature (torch.Tensors): vgg extracted model output w non-hole pixel set to ground truth
        """
        perceptual_loss = 0.0
        for i, _ in enumerate(Igt_feature):
            perceptual_loss += self.l1loss(Iout_feature[i], Igt_feature[i])
            perceptual_loss += self.l1loss(Icomp_feature[i], Igt_feature[i])
        return perceptual_loss

    def style_loss(self, Iout_feature, Igt_feature, Icomp_feature):
        """
        style_loss_out = Σ(l1loss(gram_matrix(Ψ(Iout)) - gram_matrix(Ψ(Igt))))
        style_loss_comp = Σ(l1loss(gram_matrix(Ψ(Icomp)) - gram_matrix(Ψ(Igt))))
        """
        style_loss_out = 0.0
        style_loss_comp = 0.0
        for i, _ in enumerate(Igt_feature):
            style_loss_out += self.l1loss(self.gram_matrix(Iout_feature[i]),
                                          self.gram_matrix(Igt_feature[i]))
            style_loss_comp += self.l1loss(self.gram_matrix(Icomp_feature[i]),
                                           self.gram_matrix(Igt_feature[i]))
        return style_loss_out + style_loss_comp

    def total_variation_loss(self, comp):
        """
        TODO: doctstring for total_variation_loss
        """
        # shift one pixel and get difference (for both x and y direction)
        loss = torch.mean(torch.abs(comp[:, :, :, :-1] - comp[:, :, :, 1:])) + \
            torch.mean(torch.abs(comp[:, :, :-1, :] - comp[:, :, 1:, :]))
        return loss

    def forward(self, output, mask, target):
        pixel_loss = self.pixel_loss(mask, output, target)

        comp = mask*target + (1-mask)*output
        output_feature = self.feature_extract(output)
        target_feature = self.feature_extract(target)
        comp_feature = self.feature_extract(comp)
        perceptual_loss = \
            self.perceptual_loss(output_feature, target_feature, comp_feature)
        style_loss = \
            self.style_loss(output_feature, target_feature, comp_feature)

        total_variation_loss = self.total_variation_loss(comp)
        return pixel_loss + 0.05*perceptual_loss + 120*style_loss + 0.1*total_variation_loss
        #  return 0.05*perceptual_loss + 120*style_loss + 0.1*total_variation_loss
