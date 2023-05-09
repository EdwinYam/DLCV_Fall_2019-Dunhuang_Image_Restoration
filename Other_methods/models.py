import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class baselineNet(nn.Module):
    def __init__(self, args):
        super(baselineNet, self).__init__()
        ''' declare layers used in this network '''

        self.class_num = 9
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.filter_num = [512,256,128,64,32,16]

        self.pretrained = resnet18(pretrained=True)      
        self.decoder = self._make_decoder()
        self.__setattr__('exclusive', ['decoder'])
        # modules=list(baseline_resnet18.children())[:-2]
        # self.baseline_resnet18 = nn.Sequential(*modules)
        if args.verbose:
            print(self.pretrained)
            print(self.decoder)

    def _make_decoder(self):
        layers = []
        for i in range(len(self.filter_num)-1):
            layers.append(nn.ConvTranspose2d(self.filter_num[i], 
                                             self.filter_num[i+1], 
                                             kernel_size=self.kernel_size, 
                                             stride=self.stride, 
                                             padding=self.padding,
                                             bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(self.filter_num[-1], 
                                self.class_num,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        out = self.pretrained.layer4(x)
        out = self.decoder(out)
        return out


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out

class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)

class improvedNet(nn.Module):
    def __init__(self, args):
        super(improvedNet, self).__init__()
        self.class_num = 9       
        self.aux = args.aux
        if args.backbone == 'resnet50':
            self.filter_num = 2048
            self.pretrained = resnet50(pretrained=True)      
        elif args.backbone == 'resnet18':
            self.filter_num = 512
            self.pretrained = resnet18(pretrained=True)

        self.decoder = _DAHead(self.filter_num, self.class_num, self.aux)
        self.__setattr__('exclusive', ['decoder'])
        if args.verbose:
            print(self.pretrained)
            print(self.decoder)


    def forward(self, x):
        shape = x.shape[2:]
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        out = self.pretrained.layer4(x)
        outputs = list()
        out = self.decoder(out)
        out_0 = F.interpolate(out[0], shape, mode='bilinear', align_corners=True)
        outputs.append(out_0)
        if self.aux:
            out_1 = F.interpolate(out[1], shape, mode='bilinear', align_corners=True)
            out_2 = F.interpolate(out[2], shape, mode='bilinear', align_corners=True)
            outputs.append(out_1)
            outputs.append(out_2)
        return outputs
