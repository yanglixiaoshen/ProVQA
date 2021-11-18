import torch
from torch import nn as nn
from torch.nn import functional as F
import os
from basicsr.models.archs.arch_util import (DCNv2Pack, ResidualBlockNoBN,
                                            make_layer)
import matplotlib.pyplot as plt
from SphereNet.spherenet.sphere_cnn import SphereConv2D, SphereMaxPool2D
from NonLocalNet.lib.non_local_dot_product import NONLocalBlock3D_tem
from utils.antialias import Downsample as downsamp

a=0
class Channel3DAttention(nn.Module):
    def __init__(self, in_planes, num_frame=10,  ratio=16):
        super(Channel3DAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((num_frame, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((num_frame, 1, 1))

        self.fc1   = nn.Conv3d(in_planes, in_planes // 10, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv3d(in_planes // 10, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        #self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv1 = SphereConv2D(2, 1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SphereAttentionBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SphereAttentionBlock, self).__init__()
        self.leaky = 0.1
        self.conv1 = SphereConv2D(inplanes, planes, stride=1)
        self.pool1 = SphereMaxPool2D(stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride
        self.inplanes = inplanes

    def forward(self, x):

        out = self.conv1(x)
        #out = self.pool1(out)
        out = self.relu(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        out = self.relu(out)

        return out
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = SphereConv2D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2 = SphereConv2D(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride
        self.inplanes = inplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)


        out += residual
        out = self.relu(out)

        return out
class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride
        self.inplanes = inplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)


        out += residual
        out = self.relu(out)

        return out



class SFI(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SFI, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1) # (b, 3c, h, w) (36, 3*64, 240, 480)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3]) # (b, 3, c, h, w)

        feats_U = torch.sum(inp_feats, dim=1) #  (b, c, h, w) 如果加上keepdim=True, 则会保持dim的维度不被squeeze
        feats_S = self.avg_pool(feats_U) # (b, c, 1, 1)
        feats_Z = self.conv_du(feats_S) # (b, 8, 1 , 1)

        attention_vectors = [fc(feats_Z) for fc in self.fcs] # (b, c, 1, 1)
        attention_vectors = torch.cat(attention_vectors, dim=1) # (b, 3c, 1, 1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1) # (b, 3, c, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors) # (b, 3, c, 1, 1) # 第一个维度上做softmax, 三个特征块

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1) # (b, 3, c, h, w) * (b, 3, c, 1, 1)

        return feats_V  # (b, 3, c, h, w)

import math
class ResNet_iqa(nn.Module):
    def __init__(self, in_channel, block, layers, out_channels):
        self.inplanes = in_channel
        super(ResNet_iqa, self).__init__()

        self.bot1 = nn.Sequential(downsamp(channels=self.inplanes, filt_size=3, stride=2),
                                 nn.Conv2d(self.inplanes, out_channels[0], kernel_size=1, stride=1, padding=0, bias=False))

        self.bot2 = nn.Sequential(downsamp(channels=out_channels[0], filt_size=3, stride=2),
                                  nn.Conv2d(out_channels[0], out_channels[1], 1, stride=1, padding=0, bias=False))

        self.bot3 = nn.Sequential(downsamp(channels=out_channels[1], filt_size=3, stride=2),
                                  nn.Conv2d(out_channels[1], out_channels[2], 1, stride=1, padding=0, bias=False))

        self.layer1 = self._make_layer(block, out_channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, out_channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, out_channels[2], layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3])
        #self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        downsample = None

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(self.inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        self.inplanes = planes

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )


        layers.append(self.conv1x1)
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x1 = self.layer1(x) + self.bot1(x)
        #print(x.size())
        x2 = self.layer2(x1) + self.bot2(x1)

        x3 = self.layer3(x2) + self.bot3(x2)
        #x = self.layer4(x)

        return x1, x2, x3


class SpatialFeatureModule(nn.Module):
    def __init__(self, in_channel=32, out_channels=(64, 64, 64), res_blocks=(3, 4, 6), spa3_in=64, spa3_out=32, comb_in=64): # c5:48 c4:128 c6:64 c3:96 c2:256 [3, 4, 6]
        super(SpatialFeatureModule, self).__init__()
        self.leaky = 0.1
        self.shared_layers1 = nn.Sequential(
            #nn.Conv2d(3, c5, 3, stride=1, padding=1, bias=False),
            SphereConv2D(3, in_channel, stride=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.shared_layers2 = ResNet_iqa(in_channel, BasicBlock, res_blocks, out_channels)

        self.skff_block = SFI(comb_in)

        self.shared_layers3 = nn.Sequential(
            #nn.Conv2d(c4, c6, 3, stride=1, padding=1, bias=False),
            SphereConv2D(spa3_in, spa3_out, stride=1),
            nn.BatchNorm2d(spa3_out),
            nn.ReLU(inplace=True))
        self.low_downsamp = nn.Sequential(downsamp(channels=out_channels[0], filt_size=3, stride=2),
                                 nn.Conv2d(out_channels[0], comb_in, 1, stride=1, padding=0, bias=False))

        self.high_upsamp = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(out_channels[2], comb_in, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        #x = self.shared_layers3(self.shared_layers2(self.shared_layers1(x)))
        x = self.shared_layers1(x)
        x_low, x_mid, x_high = self.shared_layers2(x)
        x1 = self.skff_block([self.low_downsamp(x_low), x_mid, self.high_upsamp(x_high)])
        x1 = self.shared_layers3(x1)
        #x = self.conv1(x)
        return x1

class MotionFeatureExtr(nn.Module):
    def __init__(self, motion_comb1=32, motion_comb2=64, motion_in=6, layers=(2, 2, 2, 2)): # c6:32 c7:64 c66:6
        super(MotionFeatureExtr, self).__init__()

        self.leaky = 0.1
        self.inplanes = motion_in
        self.inplanes1 = motion_comb2
        self.spatial_layer1 = self._make_layer(BasicBlock, motion_comb2, layers[0], stride=2)
        self.spatial_layer2 = self._make_layer(BasicBlock, motion_comb1, layers[1], stride=2)
        #self.inplanes = c7
        self.sa = SpatialAttention()

        self.spatial_layer3 = self._make_layer(BasicBlock, motion_comb1, layers[2])
        #self.spatial_layer4 = self._make_layer(BasicBlock, c6, 2)
        self.ca = ChannelAttention(motion_comb2)
        self.feat_comb_layer = nn.Sequential(
            nn.Conv2d(motion_comb2, motion_comb1, 1, bias=False),
            nn.BatchNorm2d(motion_comb1),
            nn.ReLU(inplace=True),
        )
        #self.inplanes = motion_comb2
        self.spatial_layer4 = self._make_layer1(BasicBlock, motion_comb1, layers[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )


        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def _make_layer1(self, block, planes, blocks, stride=1):
        layers = []
        downsample = None

        if stride != 1 or self.inplanes1 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes1, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )


        layers.append(block(self.inplanes1, planes, stride, downsample))
        self.inplanes1 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes1, planes))

        return nn.Sequential(*layers)

    def forward(self, x): # x [x_centers, x_motions]

        x_cent = x[0] # 32
        x_motion = x[1] #torch.cat((x[1], x[2]), dim=1) 6
        x_motion_spat = self.spatial_layer2(self.spatial_layer1(x_motion))
        x_motion_spat = x_motion_spat * self.sa(x_motion_spat)
        #print(x_motion_spat.size())
        x_mix = x_cent * x_motion_spat # 32
        #print(x_mix.size())
        x_mix_temp = torch.cat((x_mix, x_cent), dim=1)
        x_mix1 = self.feat_comb_layer(x_mix_temp * self.ca(x_mix_temp)) # 32
        #print(x_mix1.size())
        x_top = self.spatial_layer3(x_mix) # 64
        #print(x_top.size())
        x_mix2 = self.spatial_layer4(torch.cat((x_top, x_mix1), dim=1))
        x_final = x_mix2 + x_mix1


        return x_final



class TemporalNonLocalModule(nn.Module):
    def __init__(self, nonlocal_in=32):
        super(TemporalNonLocalModule, self).__init__()

        self.out = NONLocalBlock3D_tem(nonlocal_in, sub_sample=False, bn_layer=True)

    def forward(self, x):

        return self.out(x) # (b, 32, 5, 60, 120)


class TemporalScoresAggModule(nn.Module):
    def __init__(self, mos_in=32, num_frame=5, fc_dim1 = 20):
        super(TemporalScoresAggModule, self).__init__()
        self.leaky = 0.1
        self.spatial_layer1 = nn.Sequential(
            nn.Conv3d(mos_in, mos_in // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(mos_in // 2),
            nn.ReLU(inplace=True),
        )
        self.maxpool1 = nn.AdaptiveMaxPool3d((num_frame, 30, 60))
        self.avgpool1 = nn.AdaptiveAvgPool3d((num_frame, 30, 60))

        self.spatial_layer2 = nn.Sequential(
            nn.Conv3d(2*mos_in // 2, mos_in // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(mos_in // 4),
            nn.ReLU(inplace=True),
        )
        self.maxpool2 = nn.AdaptiveMaxPool3d((num_frame, 15, 30))
        self.avgpool2 = nn.AdaptiveAvgPool3d((num_frame, 15, 30))

        # self.spatial_layer3 = nn.Sequential(
        #     nn.Conv3d(2*c9, c9, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm3d(c9),
        #     nn.ReLU(inplace=True),
        # )
        self.pool1 = nn.AdaptiveAvgPool3d((num_frame, 10, 10))

        #self.ca3d = Channel3DAttention(c9*2, num_frame)

        self.pool2 = nn.AdaptiveAvgPool3d((num_frame, 1, 1))

        self.fc = nn.Sequential(
            nn.Linear(mos_in // 2*num_frame, fc_dim1, bias=False),
            #nn.BatchNorm1d(fc_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim1, 1, bias=False)
            #nn.LeakyReLU(self.leaky, inplace=True),
        )

    def forward(self, x):  # (b, 60, 10, 180, 320)
        x1 = self.spatial_layer1(x)
        x11 = self.maxpool1(x1)
        x12 = self.avgpool1(x1)
        x_1 = torch.cat((x11, x12), dim=1)  # (b, 40, 10, 90, 160)

        x2 = self.spatial_layer2(x_1)
        x21 = self.maxpool2(x2)
        x22 = self.avgpool2(x2)
        x_2 = torch.cat((x21, x22), dim=1)  # (b, 20, 10, 45, 80)
        #x_2 = self.spatial_layer3(x_2)
        x_2 = self.pool1(x_2) # (b, 20, 10, 20, 20)

        #x_2 = self.ca3d(x_2) * x_2 # (b, 20, 10, 20, 20)

        x_2 = self.pool2(x_2) # (b, 20, 10, 1, 1)
        x_2 = x_2.view(x_2.size(0), -1) # (b, 200)
        x_2 = self.fc(x_2) # score: 1

        return x_2


class BVQA360v240(nn.Module):
    def __init__(self, in_channel=32, out_channels=(64, 64, 64), res_blocks=(3, 4, 6), spa3_in=64, spa3_out=32, comb_in=64,
                       motion_comb1=32, motion_comb2=64, motion_in=6, layers=(2, 2, 2, 2),
                       nonlocal_in=32,
                       mos_in=32, num_frame=5, fc_dim1=20):
        super(BVQA360v240, self).__init__()

        self.spatial_module = SpatialFeatureModule(in_channel, out_channels, res_blocks, spa3_in, spa3_out, comb_in)
        self.motion_module = MotionFeatureExtr(motion_comb1, motion_comb2, motion_in, layers)
        self.nonlocal_module = TemporalNonLocalModule(nonlocal_in)
        self.tempotalAgg_module = TemporalScoresAggModule(mos_in, num_frame, fc_dim1)
        self.l_index = [i*3 for i in list(range(0, num_frame))] # (6, 6, 3, 240, 480) 所有的左边帧 也是supporting frame
        self.cen_index = [i * 3+1 for i in list(range(0, num_frame))] # (6, 6, 3, 240, 480) 所有的中间帧 也是reference frame
        self.r_index = [i * 3+2 for i in list(range(0, num_frame))] # (6, 6, 3, 240, 480) 所有的右边帧 也是supporting frame

    def forward(self, x):
        # input: x shape (b, t, c, h, w ) (6, 6, 3, 240, 480)
        x1 = x[:, self.cen_index, :, :, :].contiguous()
        x_spa = x1.view(-1, x.size(2), x.size(3), x.size(4)) # bt, c, h, w (36, 3, 240, 480)
        x_l = x[:, self.l_index, :, :, :].contiguous() # left frame: bt, c, h, w (36, 3, 240, 480)
        motion_l = x_spa - x_l.view(-1, x.size(2), x.size(3), x.size(4)) # motion: bt, c, h, w (36, 3, 240, 480)
        x_r = x[:, self.r_index, :, :, :].contiguous() # right frame: bt, c, h, w (36, 3, 240, 480)
        motion_r = x_r.view(-1, x.size(2), x.size(3), x.size(4)) - x_spa # motion: bt, c, h, w (36, 3, 240, 480)

        x_spa = self.spatial_module(x_spa)
        x_motion = [x_spa, torch.cat((motion_l, motion_r), dim=1)] # bt, c, h, w

        x_motion = self.motion_module(x_motion) # (bt, c, w, h)
        x_motion = x_motion.view(x.size(0), len(self.l_index), x_motion.size(1), x_motion.size(2), x_motion.size(3)) # (b, t, c, w, h)
        #print(x_motion.size())
        x_motion = x_motion.permute(0, 2, 1, 3, 4)  # (b, c, t, w, h)

        x_motion = self.nonlocal_module(x_motion)
        # print('>>>>>>>>>>>Nonlocal')
        score = self.tempotalAgg_module(x_motion)
        # print('>>>>>>>>>>>>>>>>score', score)
        # self.xx = x
        return score


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    while True:
        img = torch.randn(6, 18, 3, 240, 480) #
        net = BVQA360v240(in_channel=32, out_channels=(64, 64, 64), res_blocks=(3, 4, 6), spa3_in=64, spa3_out=32, comb_in=64,
                           motion_comb1=32, motion_comb2=64, motion_in=6, layers=(2, 2, 2, 2),
                           nonlocal_in=32,
                           mos_in=32, num_frame=6, fc_dim1=20)
        #print(net)
        score = net(img)

        print('!!!!!!!!!!!!!!!!!!!!!!!!The score is ', score)

