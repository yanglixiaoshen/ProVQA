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
        img = torch.randn(1, 18, 3, 240, 480) #
        net = BVQA360v240(in_channel=32, out_channels=(64, 64, 64), res_blocks=(3, 4, 6), spa3_in=64, spa3_out=32, comb_in=64,
                           motion_comb1=32, motion_comb2=64, motion_in=6, layers=(2, 2, 2, 2),
                           nonlocal_in=32,
                           mos_in=32, num_frame=5, fc_dim1=20)
        #print(net)
        score = net(img)

        print('!!!!!!!!!!!!!!!!!!!!!!!!The score is ', score)

























        # # if self.xx != x:
        # #     self.id += 1
        # all_frames_feat = []
        # #num = 0
        # # global a
        # # a +=1
        # # input: x shape (b, t, c, h, w ) (2, 6, 3, 720, 1280)
        # #x1 = x[:, i:i + 3, :, :, :].contiguous()
        # for i in range(0, self.num_frame*3, 3):
        #     #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>enter!')
        #     #self.id += 1
        #     print(a)
        #     #num+=1
        #     #print('>>>>>>>>>>>>>>>>>>>> x shape is ', x.size())
        #     x1 = x[:, i:i+3, :, :, :].contiguous()
        #     #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>!!!!!!!!!!!!!!!!', x1.size())
        #     #print(x1[0, 1, :, :, :].size())
        #     #plt.imsave(r'/media/yl/yl_8t/bvqa360_experiments/input_map'+ '/' + '{}_map.jpg'.format(a), x1[0, 1, :, :, :].permute(1,2,0).detach().cpu().numpy())
        #     x1_spa = x1[:, 1, :, :, :] # (b, c, h, w) 取出中间那帧
        #     motion_l = x1_spa - x1[:, 0, :, :, :]
        #     motion_r = x1[:, 2, :, :, :] - x1_spa
        #
        #     #x2 = x1_spa.view(x1_spa.size(0)*x1_spa.size(1), x1_spa.size(2), x1_spa.size(3), x1_spa.size(4))
        #     x3 = self.spatial_module(x1_spa)
        #     #print(x3.size())
        #     #print(x1.size(0), x1.size(1), x3.size(1), x3.size(2), x3.size(3))
        #     #x3 = x3.view(x1_spa.size(0), x1_spa.size(1), x3.size(1), x3.size(2), x3.size(3))
        #     x4 = [x3, motion_l, motion_r]
        #     #x4 = x3.permute(1, 0, 2, 3, 4).contiguous() # (t, b, c, w, h)
        #     x4 = self.motion_module(x4) # (b, c, w, h)
        #     #print(x4.size())
        #     # for c in range(0, x4.size(1)):
        #     #     motion = x4[0, c].detach().cpu().numpy()
        #     #     #print(motion.shape)
        #     #     if not os.path.exists(os.path.join(r'/media/yl/yl_8t/bvqa360_experiments/motion_map', '{}'.format(a))):
        #     #         os.mkdir(os.path.join(r'/media/yl/yl_8t/bvqa360_experiments/motion_map', '{}'.format(a)))
        #     #     plt.imsave(os.path.join(r'/media/yl/yl_8t/bvqa360_experiments/motion_map', '{}'.format(a)) + '/' + '{}_map.jpg'.format(c), motion, format="jpg", cmap="rainbow")
        #     all_frames_feat.append(x4)
        #     #print(x4.size())
        #     #print('>>>>>>>>>>>>>sussess!!!!!!!!!!')
        #     #print(len(all_frames_feat))
        # all_frames = torch.stack(all_frames_feat, dim = 0) # (t, b, c, w, h)
        # #print('>>>>>>>>>>>temporal')
        # all_frames = all_frames.permute(1, 2, 0, 3, 4) # (b, c, t, w, h)
        # all_frames = self.nonlocal_module(all_frames)
        # #print('>>>>>>>>>>>Nonlocal')
        # score = self.tempotalAgg_module(all_frames)
        # #print('>>>>>>>>>>>>>>>>score', score)
        # #self.xx = x
        # return score











# 3.22 修改 因为太慢 把t,b合成batch 这是之前的循环
#    def forward(self, x):
#         all_frames_feat = []
#         num = 0
#         # input: x shape (b, t, c, h, w ) (2, 6, 3, 720, 1280)
#         for i in range(0, self.num_frame*3, 3):
#             num+=1
#             #print('>>>>>>>>>>>>>>>>>>>> x shape is ', x.size())
#             x1 = x[:, i:i+3, :, :, :].contiguous()
#             x2 = x1.view(x1.size(0)*x1.size(1), x1.size(2), x1.size(3), x1.size(4))
#             x3 = self.spatial_module(x2)
#             #print(x3.size())
#             #print(x1.size(0), x1.size(1), x3.size(1), x3.size(2), x3.size(3))
#             x3 = x3.view(x1.size(0), x1.size(1), x3.size(1), x3.size(2), x3.size(3))
#             x4 = x3.permute(1, 0, 2, 3, 4).contiguous() # (t, b, c, w, h)
#             x4 = self.motion_module(x4) # (b, c, w, h)
#             all_frames_feat.append(x4)
#             #print(x4.size())
#             #print('>>>>>>>>>>>>>sussess!!!!!!!!!!')
#             #print(len(all_frames_feat))
#         all_frames = torch.stack(all_frames_feat, dim = 0) # (t, b, c, w, h)
#         #print('>>>>>>>>>>>temporal')
#         all_frames = all_frames.permute(1, 2, 0, 3, 4) # (b, c, t, w, h)
#         all_frames = self.nonlocal_module(all_frames)
#         #print('>>>>>>>>>>>Nonlocal')
#         score = self.tempotalAgg_module(all_frames)
#         #print('>>>>>>>>>>>>>>>>score', score)
#         return score





 # self.sp_1_block = 1
 #        self.sp_2_block = 1
 #        self.sp_3_block = 1
 #        self.c1 = 16
 #        self.c2 = 16
 #        self.c3 = 16
 #        self.c4 = 120 #
 #        self.c5 = 60
 #        self.c6 = 60
 #        self.c7 = 120
 #        self.c8 = 20
 #        self.c9 = 10
 #        self.fc_dim1 = 20
 #        self.num_frame = 1
 #        self.spatial_module = SpatialFeatureModule(SphereAttentionBlock, [self.sp_1_block, self.sp_2_block, self.sp_3_block], self.c1, self.c2, self.c3, self.c4, self.c5)
 #        self.motion_module = MotionFeatureExtr(self.c6, self.c7)
 #        self.nonlocal_module = TemporalNonLocalModule()
 #        self.tempotalAgg_module = TemporalScoresAggModule(self.c6, self.c8, self.c9, self.num_frame, self.fc_dim1)


# class PCDAlignment(nn.Module):
#     """Alignment module using Pyramid, Cascading and Deformable convolution
#     (PCD). It is used in EDVR.
#
#     Ref:
#         EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
#
#     Args:
#         num_feat (int): Channel number of middle features. Default: 64.
#         deformable_groups (int): Deformable groups. Defaults: 8.
#     """
#
#     def __init__(self, num_feat=64, deformable_groups=8):
#         super(PCDAlignment, self).__init__()
#
#         # Pyramid has three levels:
#         # L3: level 3, 1/4 spatial size
#         # L2: level 2, 1/2 spatial size
#         # L1: level 1, original spatial size
#         self.offset_conv1 = nn.ModuleDict()
#         self.offset_conv2 = nn.ModuleDict()
#         self.offset_conv3 = nn.ModuleDict()
#         self.dcn_pack = nn.ModuleDict()
#         self.feat_conv = nn.ModuleDict()
#
#         # Pyramids
#         for i in range(3, 0, -1):
#             level = f'l{i}'
#             self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
#                                                  1)
#             if i == 3:
#                 self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
#                                                      1)
#             else:
#                 self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3,
#                                                      1, 1)
#                 self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
#                                                      1)
#             self.dcn_pack[level] = DCNv2Pack(
#                 num_feat,
#                 num_feat,
#                 3,
#                 padding=1,
#                 deformable_groups=deformable_groups)
#
#             if i < 3:
#                 self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
#                                                   1)
#
#         # Cascading dcn
#         self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
#         self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.cas_dcnpack = DCNv2Pack(
#             num_feat,
#             num_feat,
#             3,
#             padding=1,
#             deformable_groups=deformable_groups)
#
#         self.upsample = nn.Upsample(
#             scale_factor=2, mode='bilinear', align_corners=False)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#
#     def forward(self, nbr_feat_l, ref_feat_l):
#         """Align neighboring frame features to the reference frame features.
#
#         Args:
#             nbr_feat_l (list[Tensor]): Neighboring feature list. It
#                 contains three pyramid levels (L1, L2, L3),
#                 each with shape (b, c, h, w).
#             ref_feat_l (list[Tensor]): Reference feature list. It
#                 contains three pyramid levels (L1, L2, L3),
#                 each with shape (b, c, h, w).
#
#         Returns:
#             Tensor: Aligned features.
#         """
#         # Pyramids
#         upsampled_offset, upsampled_feat = None, None
#         for i in range(3, 0, -1):
#             level = f'l{i}'
#             offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
#             offset = self.lrelu(self.offset_conv1[level](offset))
#             if i == 3:
#                 offset = self.lrelu(self.offset_conv2[level](offset))
#             else:
#                 offset = self.lrelu(self.offset_conv2[level](torch.cat(
#                     [offset, upsampled_offset], dim=1)))
#                 offset = self.lrelu(self.offset_conv3[level](offset))
#
#             feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
#             if i < 3:
#                 feat = self.feat_conv[level](
#                     torch.cat([feat, upsampled_feat], dim=1))
#             if i > 1:
#                 feat = self.lrelu(feat)
#
#             if i > 1:  # upsample offset and features
#                 # x2: when we upsample the offset, we should also enlarge
#                 # the magnitude.
#                 upsampled_offset = self.upsample(offset) * 2
#                 upsampled_feat = self.upsample(feat)
#
#         # Cascading
#         offset = torch.cat([feat, ref_feat_l[0]], dim=1)
#         offset = self.lrelu(
#             self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
#         feat = self.lrelu(self.cas_dcnpack(feat, offset))
#         return feat
#
#
# class TSAFusion(nn.Module):
#     """Temporal Spatial Attention (TSA) fusion module.
#
#     Temporal: Calculate the correlation between center frame and
#         neighboring frames;
#     Spatial: It has 3 pyramid levels, the attention is similar to SFT.
#         (SFT: Recovering realistic texture in image super-resolution by deep
#             spatial feature transform.)
#
#     Args:
#         num_feat (int): Channel number of middle features. Default: 64.
#         num_frame (int): Number of frames. Default: 5.
#         center_frame_idx (int): The index of center frame. Default: 2.
#     """
#
#     def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
#         super(TSAFusion, self).__init__()
#         self.center_frame_idx = center_frame_idx
#         # temporal attention (before fusion conv)
#         self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
#
#         # spatial attention (after fusion conv)
#         self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
#         self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
#         self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
#         self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
#         self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
#         self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
#         self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
#         self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
#         self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#         self.upsample = nn.Upsample(
#             scale_factor=2, mode='bilinear', align_corners=False)
#
#     def forward(self, aligned_feat):
#         """
#         Args:
#             aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).
#
#         Returns:
#             Tensor: Features after TSA with the shape (b, c, h, w).
#         """
#         b, t, c, h, w = aligned_feat.size()
#         # temporal attention
#         embedding_ref = self.temporal_attn1(
#             aligned_feat[:, self.center_frame_idx, :, :, :].clone())
#         embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
#         embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)
#
#         corr_l = []  # correlation list
#         for i in range(t):
#             emb_neighbor = embedding[:, i, :, :, :]
#             corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
#             corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
#         corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
#         corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
#         corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
#         aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob
#
#         # fusion
#         feat = self.lrelu(self.feat_fusion(aligned_feat))
#
#         # spatial attention
#         attn = self.lrelu(self.spatial_attn1(aligned_feat))
#         attn_max = self.max_pool(attn)
#         attn_avg = self.avg_pool(attn)
#         attn = self.lrelu(
#             self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
#         # pyramid levels
#         attn_level = self.lrelu(self.spatial_attn_l1(attn))
#         attn_max = self.max_pool(attn_level)
#         attn_avg = self.avg_pool(attn_level)
#         attn_level = self.lrelu(
#             self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
#         attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
#         attn_level = self.upsample(attn_level)
#
#         attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
#         attn = self.lrelu(self.spatial_attn4(attn))
#         attn = self.upsample(attn)
#         attn = self.spatial_attn5(attn)
#         attn_add = self.spatial_attn_add2(
#             self.lrelu(self.spatial_attn_add1(attn)))
#         attn = torch.sigmoid(attn)
#
#         # after initialization, * 2 makes (attn * 2) to be close to 1.
#         feat = feat * attn * 2 + attn_add
#         return feat
#
#
# class PredeblurModule(nn.Module):
#     """Pre-dublur module.
#
#     Args:
#         num_in_ch (int): Channel number of input image. Default: 3.
#         num_feat (int): Channel number of intermediate features. Default: 64.
#         hr_in (bool): Whether the input has high resolution. Default: False.
#     """
#
#     def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
#         super(PredeblurModule, self).__init__()
#         self.hr_in = hr_in
#
#         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
#         if self.hr_in:
#             # downsample x4 by stride conv
#             self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
#             self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
#
#         # generate feature pyramid
#         self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
#         self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
#
#         self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
#         self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
#         self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
#         self.resblock_l1 = nn.ModuleList(
#             [ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])
#
#         self.upsample = nn.Upsample(
#             scale_factor=2, mode='bilinear', align_corners=False)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#
#     def forward(self, x):
#         feat_l1 = self.lrelu(self.conv_first(x))
#         if self.hr_in:
#             feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
#             feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))
#
#         # generate feature pyramid
#         feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
#         feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))
#
#         feat_l3 = self.upsample(self.resblock_l3(feat_l3))
#         feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
#         feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))
#
#         for i in range(2):
#             feat_l1 = self.resblock_l1[i](feat_l1)
#         feat_l1 = feat_l1 + feat_l2
#         for i in range(2, 5):
#             feat_l1 = self.resblock_l1[i](feat_l1)
#         return feat_l1
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# class EDVR(nn.Module):
#     """EDVR network structure for video super-resolution.
#
#     Now only support X4 upsampling factor.
#     Paper:
#         EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
#
#     Args:
#         num_in_ch (int): Channel number of input image. Default: 3.
#         num_out_ch (int): Channel number of output image. Default: 3.
#         num_feat (int): Channel number of intermediate features. Default: 64.
#         num_frame (int): Number of input frames. Default: 5.
#         deformable_groups (int): Deformable groups. Defaults: 8.
#         num_extract_block (int): Number of blocks for feature extraction.
#             Default: 5.
#         num_reconstruct_block (int): Number of blocks for reconstruction.
#             Default: 10.
#         center_frame_idx (int): The index of center frame. Frame counting from
#             0. Default: 2.
#         hr_in (bool): Whether the input has high resolution. Default: False.
#         with_predeblur (bool): Whether has predeblur module.
#             Default: False.
#         with_tsa (bool): Whether has TSA module. Default: True.
#     """
#
#     def __init__(self,
#                  num_in_ch=3,
#                  num_out_ch=3,
#                  num_feat=64,
#                  num_frame=5,
#                  deformable_groups=8,
#                  num_extract_block=5,
#                  num_reconstruct_block=10,
#                  center_frame_idx=2,
#                  hr_in=False,
#                  with_predeblur=False,
#                  with_tsa=True):
#         super(EDVR, self).__init__()
#         if center_frame_idx is None:
#             self.center_frame_idx = num_frame // 2
#         else:
#             self.center_frame_idx = center_frame_idx
#         self.hr_in = hr_in
#         self.with_predeblur = with_predeblur
#         self.with_tsa = with_tsa
#
#         # extract features for each frame
#         if self.with_predeblur:
#             self.predeblur = PredeblurModule(
#                 num_feat=num_feat, hr_in=self.hr_in)
#             self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
#         else:
#             self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
#
#         # extrat pyramid features
#         self.feature_extraction = make_layer(
#             ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
#         self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
#         self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
#         self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#
#         # pcd and tsa module
#         self.pcd_align = PCDAlignment(
#             num_feat=num_feat, deformable_groups=deformable_groups)
#         if self.with_tsa:
#             self.fusion = TSAFusion(
#                 num_feat=num_feat,
#                 num_frame=num_frame,
#                 center_frame_idx=self.center_frame_idx)
#         else:
#             self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
#
#         # reconstruction
#         self.reconstruction = make_layer(
#             ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
#         # upsample
#         self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
#         self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
#         self.pixel_shuffle = nn.PixelShuffle(2)
#         self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
#         self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
#
#         # activation function
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#
#     def forward(self, x):
#         b, t, c, h, w = x.size()
#         if self.hr_in:
#             assert h % 16 == 0 and w % 16 == 0, (
#                 'The height and width must be multiple of 16.')
#         else:
#             assert h % 4 == 0 and w % 4 == 0, (
#                 'The height and width must be multiple of 4.')
#
#         x_center = x[:, self.center_frame_idx, :, :, :].contiguous()
#
#         # extract features for each frame
#         # L1
#         if self.with_predeblur:
#             feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
#             if self.hr_in:
#                 h, w = h // 4, w // 4
#         else:
#             feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
#
#         feat_l1 = self.feature_extraction(feat_l1)
#         # L2
#         feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
#         feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
#         # L3
#         feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
#         feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))
#
#         feat_l1 = feat_l1.view(b, t, -1, h, w)
#         feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
#         feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)
#
#         # PCD alignment
#         ref_feat_l = [  # reference feature list
#             feat_l1[:, self.center_frame_idx, :, :, :].clone(),
#             feat_l2[:, self.center_frame_idx, :, :, :].clone(),
#             feat_l3[:, self.center_frame_idx, :, :, :].clone()
#         ]
#         aligned_feat = []
#         for i in range(t):
#             nbr_feat_l = [  # neighboring feature list
#                 feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
#                 feat_l3[:, i, :, :, :].clone()
#             ]
#             aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
#         aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)
#
#         if not self.with_tsa:
#             aligned_feat = aligned_feat.view(b, -1, h, w)
#         feat = self.fusion(aligned_feat)
#
#         out = self.reconstruction(feat)
#         out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
#         out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
#         out = self.lrelu(self.conv_hr(out))
#         out = self.conv_last(out)
#         if self.hr_in:
#             base = x_center
#         else:
#             base = F.interpolate(
#                 x_center, scale_factor=4, mode='bilinear', align_corners=False)
#         out += base
#         return out













































# import torch
# from torch import nn as nn
# from torch.nn import functional as F
#
# from basicsr.models.archs.arch_util import (DCNv2Pack, ResidualBlockNoBN,
#                                             make_layer)
# from SphereNet.spherenet.sphere_cnn import SphereConv2D, SphereMaxPool2D
# from NonLocalNet.lib.non_local_dot_product import NONLocalBlock3D_tem
#
# class Channel3DAttention(nn.Module):
#     def __init__(self, in_planes, num_frame=10,  ratio=16):
#         super(Channel3DAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool3d((num_frame, 1, 1))
#         self.max_pool = nn.AdaptiveMaxPool3d((num_frame, 1, 1))
#
#         self.fc1   = nn.Conv3d(in_planes, in_planes // 10, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv3d(in_planes // 10, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=3):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         #self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.conv1 = SphereConv2D(2, 1, stride=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
#
# class SphereAttentionBlock(nn.Module):
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(SphereAttentionBlock, self).__init__()
#         self.leaky = 0.1
#         self.conv1 = SphereConv2D(inplanes, planes, stride=1)
#         self.pool1 = SphereMaxPool2D(stride=2)
#         self.relu = nn.ReLU(inplace=True)
#
#         # self.ca = ChannelAttention(planes)
#         # self.sa = SpatialAttention()
#
#         self.downsample = downsample
#         self.stride = stride
#         self.inplanes = inplanes
#
#     def forward(self, x):
#
#         out = self.conv1(x)
#         #out = self.pool1(out)
#         out = self.relu(out)
#
#         # out = self.ca(out) * out
#         # out = self.sa(out) * out
#         #
#         # out = self.relu(out)
#
#         return out
#
#
# class SpatialFeatureModule(nn.Module):
#     def __init__(self, block, layers, c1, c2, c3, c4, c5):
#         super(SpatialFeatureModule, self).__init__()
#         self.leaky = 0.1
#         self.inplanes = 32
#         self.spatial_layer1 = nn.Sequential(
#             SphereConv2D(3, 32, stride=1),
#             SphereMaxPool2D(stride=2),
#             nn.ReLU(inplace=True),
#         )
#
#         #self.conv1 = SphereConv2D(3, 32, stride=1)
#         self.layer1 = self._make_layer(block, c1, layers[0])
#         self.ca1 = ChannelAttention(c1)
#         self.sa1 = SpatialAttention()
#         self.layer2 = self._make_layer(block, c2, layers[1])
#         self.ca2 = ChannelAttention(c2)
#         self.sa2 = SpatialAttention()
#         self.layer3 = self._make_layer(block, c3, layers[2])
#         self.ca3 = ChannelAttention(c3)
#         self.sa3 = SpatialAttention()
#
#
#
#         # Sphere3*3: Reduce the channel number and pooling.
#         self.spatial_layer2 = nn.Sequential(
#             SphereConv2D(c1+c2+c3, c4, stride=1),
#             SphereMaxPool2D(stride=2),
#             nn.LeakyReLU(self.leaky, inplace=True),
#         )
#         # conv1*1: Reduce the channel number
#         self.spatial_layer3 = nn.Sequential(
#             nn.Conv2d(c4, c5, 1, bias=False),
#             #SphereMaxPool2D(stride=2),
#             nn.ReLU(inplace=True),
#         )
#
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         #downsample = None
#
#         layers = []
#
#         layers.append(block(self.inplanes, planes, stride))
#         self.inplanes = planes
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, stride))
#
#         return nn.Sequential(*layers)
#
#
#     def forward(self, x):
#         x = self.spatial_layer1(x)
#         out1 = self.layer1(x) # (b, 96, 360, 640)
#         out1 = self.ca1(out1) * out1
#         out1 = self.sa1(out1) * out1
#
#         out2 = self.layer2(out1) # (b, 128, )
#         out2 = self.ca2(out2) * out2
#         out2 = self.sa2(out2) * out2
#
#         out3 = self.layer3(out2) # (b, 256, )
#         out3 = self.ca3(out3) * out3
#         out3 = self.sa3(out3) * out3
#
#         out = torch.cat((out1, out2, out3), dim=1)
#         out = self.spatial_layer2(out)
#         out = self.spatial_layer3(out)
#
#         #x = self.conv1(x)
#         return out
#
# class MotionFeatureExtr(nn.Module):
#     def __init__(self, c6, c7):
#         super(MotionFeatureExtr, self).__init__()
#         self.leaky = 0.1
#         self.spatial_layer1 = nn.Sequential(
#             SphereConv2D(c6, c7, stride=1),
#             #SphereMaxPool2D(stride=2),
#             nn.ReLU(inplace=True),
#             SphereConv2D(c7, c6, stride=1),
#             #SphereMaxPool2D(stride=2),
#             nn.ReLU(inplace=True),
#         )
#         self.spatial_layer2 = nn.Sequential(
#             SphereConv2D(c6, c7, stride=1),
#             #SphereMaxPool2D(stride=2),
#             nn.ReLU(inplace=True),
#             SphereConv2D(c7, c6, stride=1),
#             #SphereMaxPool2D(stride=2),
#             nn.ReLU(inplace=True),
#         )
#         self.spatial_layer3 = nn.Sequential(
#             SphereConv2D(c7, c6, stride=1),
#             #SphereMaxPool2D(stride=2),
#             nn.ReLU(inplace=True),
#         )
#         self.spatial_layer4 = nn.Sequential(
#             SphereConv2D(c7, c6, stride=1),
#             #SphereMaxPool2D(stride=2),
#             nn.ReLU(inplace=True),
#         )
#         self.feat_comb_layer = nn.Sequential(
#             nn.Conv2d(3*c6, c6, 1, bias=False),
#             #SphereMaxPool2D(stride=2),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         x_down = x[0, :, :, :, :]
#         x_cent = x[1, :, :, :, :]
#         x_upon = x[2, :, :, :, :]
#
#         x_dow = x_cent - x_down
#         x_up  = x_upon - x_cent
#
#         x_dow = self.spatial_layer1(x_dow)
#         x_up  = self.spatial_layer2(x_up)
#
#         x_combin = torch.cat((x_down, x_cent, x_upon), dim=1)
#         x_combin_sqz = self.feat_comb_layer(x_combin)
#
#         x_com_dow = torch.cat((x_dow, x_combin_sqz), dim=1)
#         x_com_up  = torch.cat((x_up, x_combin_sqz), dim=1)
#
#         x_com_dow_sqz = self.spatial_layer3(x_com_dow)
#         x_com_up_sqz = self.spatial_layer4(x_com_up)
#
#         #x_out = x_com_dow_sqz + x_com_up_sqz + x_combin_sqz
#
#         return x_com_dow_sqz + x_com_up_sqz + x_combin_sqz
#
#
#
# class TemporalNonLocalModule(nn.Module):
#     def __init__(self):
#         super(TemporalNonLocalModule, self).__init__()
#
#         self.out = NONLocalBlock3D_tem(60, sub_sample=False, bn_layer=False)
#
#     def forward(self, x):
#
#         return self.out(x) # (b, 60, 10, 90, 180)
#
#
# class TemporalScoresAggModule(nn.Module):
#     def __init__(self, c6=60, c8=20, c9 = 10, num_frame=10, fc_dim1 = 20):
#         super(TemporalScoresAggModule, self).__init__()
#         self.leaky = 0.1
#         self.spatial_layer1 = nn.Sequential(
#             nn.Conv3d(c6, c8, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#         )
#         self.maxpool1 = nn.AdaptiveMaxPool3d((num_frame, 45, 90))
#         self.avgpool1 = nn.AdaptiveAvgPool3d((num_frame, 45, 90))
#
#         self.spatial_layer2 = nn.Sequential(
#             nn.Conv3d(c8*2, c9, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#         )
#         self.maxpool2 = nn.AdaptiveMaxPool3d((num_frame, 45, 90))
#         self.avgpool2 = nn.AdaptiveAvgPool3d((num_frame, 45, 90))
#
#         self.pool1 = nn.AdaptiveAvgPool3d((num_frame, 20, 20))
#
#         self.ca3d = Channel3DAttention(c9*2, num_frame)
#
#         self.pool2 = nn.AdaptiveAvgPool3d((num_frame, 1, 1))
#
#         self.fc = nn.Sequential(
#             nn.Linear(c9*2*num_frame, fc_dim1, bias=False),
#             #nn.LeakyReLU(self.leaky, inplace=True),
#             nn.Linear(fc_dim1, 1),
#             #nn.LeakyReLU(self.leaky, inplace=True),
#         )
#
#     def forward(self, x):  # (b, 60, 10, 180, 320)
#         x1 = self.spatial_layer1(x)
#         x11 = self.maxpool1(x1)
#         x12 = self.avgpool1(x1)
#         x_1 = torch.cat((x11, x12), dim=1)  # (b, 40, 10, 90, 160)
#
#         x2 = self.spatial_layer2(x_1)
#         x21 = self.maxpool1(x2)
#         x22 = self.avgpool1(x2)
#         x_2 = torch.cat((x21, x22), dim=1)  # (b, 20, 10, 45, 80)
#
#         x_2 = self.pool1(x_2) # (b, 20, 10, 20, 20)
#
#         x_2 = self.ca3d(x_2) * x_2 # (b, 20, 10, 20, 20)
#
#         x_2 = self.pool2(x_2) # (b, 20, 10, 1, 1)
#         x_2 = x_2.view(x_2.size(0), -1) # (b, 200)
#         x_2 = self.fc(x_2) # score: 1
#
#         return x_2
#
#
#
#
#
# class BVQA360(nn.Module):
#     def __init__(self):
#         super(BVQA360, self).__init__()
#         self.sp_1_block = 2
#         self.sp_2_block = 2
#         self.sp_3_block = 3
#         self.c1 = 32
#         self.c2 = 64
#         self.c3 = 128
#         self.c4 = 120
#         self.c5 = 60
#         self.c6 = 60
#         self.c7 = 120
#         self.c8 = 20
#         self.c9 = 10
#         self.fc_dim1 = 20
#         self.num_frame = 1
#         self.spatial_module = SpatialFeatureModule(SphereAttentionBlock, [self.sp_1_block, self.sp_2_block, self.sp_3_block], self.c1, self.c2, self.c3, self.c4, self.c5)
#         self.motion_module = MotionFeatureExtr(self.c6, self.c7)
#         self.nonlocal_module = TemporalNonLocalModule()
#         self.tempotalAgg_module = TemporalScoresAggModule(self.c6, self.c8, self.c9, self.num_frame, self.fc_dim1)
#     def forward(self, x):
#         all_frames_feat = []
#         num = 0
#         # input: x shape (b, t, c, h, w ) (2, 6, 3, 720, 1280)
#         for i in range(0, self.num_frame*3, 3):
#             num+=1
#             #print('>>>>>>>>>>>>>>>num is ', num)
#             x1 = x[:, i:i+3, :, :, :].contiguous()
#             x2 = x1.view(x1.size(0)*x1.size(1), x1.size(2), x1.size(3), x1.size(4))
#             x3 = self.spatial_module(x2)
#             #print(x3.size())
#             #print(x1.size(0), x1.size(1), x3.size(1), x3.size(2), x3.size(3))
#             x3 = x3.view(x1.size(0), x1.size(1), x3.size(1), x3.size(2), x3.size(3))
#             x4 = x3.permute(1, 0, 2, 3, 4).contiguous() # (t, b, c, w, h)
#             x4 = self.motion_module(x4) # (b, c, w, h)
#             all_frames_feat.append(x4)
#             #print(x4.size())
#             #print('>>>>>>>>>>>>>sussess!!!!!!!!!!')
#             #print(len(all_frames_feat))
#         all_frames = torch.stack(all_frames_feat, dim = 0) # (t, b, c, w, h)
#         #print('>>>>>>>>>>>temporal')
#         all_frames = all_frames.permute(1, 2, 0, 3, 4) # (b, c, t, w, h)
#         all_frames = self.nonlocal_module(all_frames)
#         #print('>>>>>>>>>>>Nonlocal')
#         score = self.tempotalAgg_module(all_frames)
#         print('>>>>>>>>>>>>>>>>score', score)
#         return score
#
#
#  # self.sp_1_block = 1
#  #        self.sp_2_block = 1
#  #        self.sp_3_block = 1
#  #        self.c1 = 16
#  #        self.c2 = 16
#  #        self.c3 = 16
#  #        self.c4 = 120 #
#  #        self.c5 = 60
#  #        self.c6 = 60
#  #        self.c7 = 120
#  #        self.c8 = 20
#  #        self.c9 = 10
#  #        self.fc_dim1 = 20
#  #        self.num_frame = 1
#  #        self.spatial_module = SpatialFeatureModule(SphereAttentionBlock, [self.sp_1_block, self.sp_2_block, self.sp_3_block], self.c1, self.c2, self.c3, self.c4, self.c5)
#  #        self.motion_module = MotionFeatureExtr(self.c6, self.c7)
#  #        self.nonlocal_module = TemporalNonLocalModule()
#  #        self.tempotalAgg_module = TemporalScoresAggModule(self.c6, self.c8, self.c9, self.num_frame, self.fc_dim1)
#
#
# # class PCDAlignment(nn.Module):
# #     """Alignment module using Pyramid, Cascading and Deformable convolution
# #     (PCD). It is used in EDVR.
# #
# #     Ref:
# #         EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
# #
# #     Args:
# #         num_feat (int): Channel number of middle features. Default: 64.
# #         deformable_groups (int): Deformable groups. Defaults: 8.
# #     """
# #
# #     def __init__(self, num_feat=64, deformable_groups=8):
# #         super(PCDAlignment, self).__init__()
# #
# #         # Pyramid has three levels:
# #         # L3: level 3, 1/4 spatial size
# #         # L2: level 2, 1/2 spatial size
# #         # L1: level 1, original spatial size
# #         self.offset_conv1 = nn.ModuleDict()
# #         self.offset_conv2 = nn.ModuleDict()
# #         self.offset_conv3 = nn.ModuleDict()
# #         self.dcn_pack = nn.ModuleDict()
# #         self.feat_conv = nn.ModuleDict()
# #
# #         # Pyramids
# #         for i in range(3, 0, -1):
# #             level = f'l{i}'
# #             self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
# #                                                  1)
# #             if i == 3:
# #                 self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
# #                                                      1)
# #             else:
# #                 self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3,
# #                                                      1, 1)
# #                 self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
# #                                                      1)
# #             self.dcn_pack[level] = DCNv2Pack(
# #                 num_feat,
# #                 num_feat,
# #                 3,
# #                 padding=1,
# #                 deformable_groups=deformable_groups)
# #
# #             if i < 3:
# #                 self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
# #                                                   1)
# #
# #         # Cascading dcn
# #         self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
# #         self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #         self.cas_dcnpack = DCNv2Pack(
# #             num_feat,
# #             num_feat,
# #             3,
# #             padding=1,
# #             deformable_groups=deformable_groups)
# #
# #         self.upsample = nn.Upsample(
# #             scale_factor=2, mode='bilinear', align_corners=False)
# #         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
# #
# #     def forward(self, nbr_feat_l, ref_feat_l):
# #         """Align neighboring frame features to the reference frame features.
# #
# #         Args:
# #             nbr_feat_l (list[Tensor]): Neighboring feature list. It
# #                 contains three pyramid levels (L1, L2, L3),
# #                 each with shape (b, c, h, w).
# #             ref_feat_l (list[Tensor]): Reference feature list. It
# #                 contains three pyramid levels (L1, L2, L3),
# #                 each with shape (b, c, h, w).
# #
# #         Returns:
# #             Tensor: Aligned features.
# #         """
# #         # Pyramids
# #         upsampled_offset, upsampled_feat = None, None
# #         for i in range(3, 0, -1):
# #             level = f'l{i}'
# #             offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
# #             offset = self.lrelu(self.offset_conv1[level](offset))
# #             if i == 3:
# #                 offset = self.lrelu(self.offset_conv2[level](offset))
# #             else:
# #                 offset = self.lrelu(self.offset_conv2[level](torch.cat(
# #                     [offset, upsampled_offset], dim=1)))
# #                 offset = self.lrelu(self.offset_conv3[level](offset))
# #
# #             feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
# #             if i < 3:
# #                 feat = self.feat_conv[level](
# #                     torch.cat([feat, upsampled_feat], dim=1))
# #             if i > 1:
# #                 feat = self.lrelu(feat)
# #
# #             if i > 1:  # upsample offset and features
# #                 # x2: when we upsample the offset, we should also enlarge
# #                 # the magnitude.
# #                 upsampled_offset = self.upsample(offset) * 2
# #                 upsampled_feat = self.upsample(feat)
# #
# #         # Cascading
# #         offset = torch.cat([feat, ref_feat_l[0]], dim=1)
# #         offset = self.lrelu(
# #             self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
# #         feat = self.lrelu(self.cas_dcnpack(feat, offset))
# #         return feat
# #
# #
# # class TSAFusion(nn.Module):
# #     """Temporal Spatial Attention (TSA) fusion module.
# #
# #     Temporal: Calculate the correlation between center frame and
# #         neighboring frames;
# #     Spatial: It has 3 pyramid levels, the attention is similar to SFT.
# #         (SFT: Recovering realistic texture in image super-resolution by deep
# #             spatial feature transform.)
# #
# #     Args:
# #         num_feat (int): Channel number of middle features. Default: 64.
# #         num_frame (int): Number of frames. Default: 5.
# #         center_frame_idx (int): The index of center frame. Default: 2.
# #     """
# #
# #     def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
# #         super(TSAFusion, self).__init__()
# #         self.center_frame_idx = center_frame_idx
# #         # temporal attention (before fusion conv)
# #         self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #         self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #         self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
# #
# #         # spatial attention (after fusion conv)
# #         self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
# #         self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
# #         self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
# #         self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
# #         self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #         self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
# #         self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #         self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
# #         self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
# #         self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #         self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
# #         self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)
# #
# #         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
# #         self.upsample = nn.Upsample(
# #             scale_factor=2, mode='bilinear', align_corners=False)
# #
# #     def forward(self, aligned_feat):
# #         """
# #         Args:
# #             aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).
# #
# #         Returns:
# #             Tensor: Features after TSA with the shape (b, c, h, w).
# #         """
# #         b, t, c, h, w = aligned_feat.size()
# #         # temporal attention
# #         embedding_ref = self.temporal_attn1(
# #             aligned_feat[:, self.center_frame_idx, :, :, :].clone())
# #         embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
# #         embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)
# #
# #         corr_l = []  # correlation list
# #         for i in range(t):
# #             emb_neighbor = embedding[:, i, :, :, :]
# #             corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
# #             corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
# #         corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
# #         corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
# #         corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
# #         aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob
# #
# #         # fusion
# #         feat = self.lrelu(self.feat_fusion(aligned_feat))
# #
# #         # spatial attention
# #         attn = self.lrelu(self.spatial_attn1(aligned_feat))
# #         attn_max = self.max_pool(attn)
# #         attn_avg = self.avg_pool(attn)
# #         attn = self.lrelu(
# #             self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
# #         # pyramid levels
# #         attn_level = self.lrelu(self.spatial_attn_l1(attn))
# #         attn_max = self.max_pool(attn_level)
# #         attn_avg = self.avg_pool(attn_level)
# #         attn_level = self.lrelu(
# #             self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
# #         attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
# #         attn_level = self.upsample(attn_level)
# #
# #         attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
# #         attn = self.lrelu(self.spatial_attn4(attn))
# #         attn = self.upsample(attn)
# #         attn = self.spatial_attn5(attn)
# #         attn_add = self.spatial_attn_add2(
# #             self.lrelu(self.spatial_attn_add1(attn)))
# #         attn = torch.sigmoid(attn)
# #
# #         # after initialization, * 2 makes (attn * 2) to be close to 1.
# #         feat = feat * attn * 2 + attn_add
# #         return feat
# #
# #
# # class PredeblurModule(nn.Module):
# #     """Pre-dublur module.
# #
# #     Args:
# #         num_in_ch (int): Channel number of input image. Default: 3.
# #         num_feat (int): Channel number of intermediate features. Default: 64.
# #         hr_in (bool): Whether the input has high resolution. Default: False.
# #     """
# #
# #     def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
# #         super(PredeblurModule, self).__init__()
# #         self.hr_in = hr_in
# #
# #         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
# #         if self.hr_in:
# #             # downsample x4 by stride conv
# #             self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
# #             self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
# #
# #         # generate feature pyramid
# #         self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
# #         self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
# #
# #         self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
# #         self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
# #         self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
# #         self.resblock_l1 = nn.ModuleList(
# #             [ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])
# #
# #         self.upsample = nn.Upsample(
# #             scale_factor=2, mode='bilinear', align_corners=False)
# #         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
# #
# #     def forward(self, x):
# #         feat_l1 = self.lrelu(self.conv_first(x))
# #         if self.hr_in:
# #             feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
# #             feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))
# #
# #         # generate feature pyramid
# #         feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
# #         feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))
# #
# #         feat_l3 = self.upsample(self.resblock_l3(feat_l3))
# #         feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
# #         feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))
# #
# #         for i in range(2):
# #             feat_l1 = self.resblock_l1[i](feat_l1)
# #         feat_l1 = feat_l1 + feat_l2
# #         for i in range(2, 5):
# #             feat_l1 = self.resblock_l1[i](feat_l1)
# #         return feat_l1
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # class EDVR(nn.Module):
# #     """EDVR network structure for video super-resolution.
# #
# #     Now only support X4 upsampling factor.
# #     Paper:
# #         EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
# #
# #     Args:
# #         num_in_ch (int): Channel number of input image. Default: 3.
# #         num_out_ch (int): Channel number of output image. Default: 3.
# #         num_feat (int): Channel number of intermediate features. Default: 64.
# #         num_frame (int): Number of input frames. Default: 5.
# #         deformable_groups (int): Deformable groups. Defaults: 8.
# #         num_extract_block (int): Number of blocks for feature extraction.
# #             Default: 5.
# #         num_reconstruct_block (int): Number of blocks for reconstruction.
# #             Default: 10.
# #         center_frame_idx (int): The index of center frame. Frame counting from
# #             0. Default: 2.
# #         hr_in (bool): Whether the input has high resolution. Default: False.
# #         with_predeblur (bool): Whether has predeblur module.
# #             Default: False.
# #         with_tsa (bool): Whether has TSA module. Default: True.
# #     """
# #
# #     def __init__(self,
# #                  num_in_ch=3,
# #                  num_out_ch=3,
# #                  num_feat=64,
# #                  num_frame=5,
# #                  deformable_groups=8,
# #                  num_extract_block=5,
# #                  num_reconstruct_block=10,
# #                  center_frame_idx=2,
# #                  hr_in=False,
# #                  with_predeblur=False,
# #                  with_tsa=True):
# #         super(EDVR, self).__init__()
# #         if center_frame_idx is None:
# #             self.center_frame_idx = num_frame // 2
# #         else:
# #             self.center_frame_idx = center_frame_idx
# #         self.hr_in = hr_in
# #         self.with_predeblur = with_predeblur
# #         self.with_tsa = with_tsa
# #
# #         # extract features for each frame
# #         if self.with_predeblur:
# #             self.predeblur = PredeblurModule(
# #                 num_feat=num_feat, hr_in=self.hr_in)
# #             self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
# #         else:
# #             self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
# #
# #         # extrat pyramid features
# #         self.feature_extraction = make_layer(
# #             ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
# #         self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
# #         self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #         self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
# #         self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #
# #         # pcd and tsa module
# #         self.pcd_align = PCDAlignment(
# #             num_feat=num_feat, deformable_groups=deformable_groups)
# #         if self.with_tsa:
# #             self.fusion = TSAFusion(
# #                 num_feat=num_feat,
# #                 num_frame=num_frame,
# #                 center_frame_idx=self.center_frame_idx)
# #         else:
# #             self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
# #
# #         # reconstruction
# #         self.reconstruction = make_layer(
# #             ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
# #         # upsample
# #         self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
# #         self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
# #         self.pixel_shuffle = nn.PixelShuffle(2)
# #         self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
# #         self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
# #
# #         # activation function
# #         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
# #
# #     def forward(self, x):
# #         b, t, c, h, w = x.size()
# #         if self.hr_in:
# #             assert h % 16 == 0 and w % 16 == 0, (
# #                 'The height and width must be multiple of 16.')
# #         else:
# #             assert h % 4 == 0 and w % 4 == 0, (
# #                 'The height and width must be multiple of 4.')
# #
# #         x_center = x[:, self.center_frame_idx, :, :, :].contiguous()
# #
# #         # extract features for each frame
# #         # L1
# #         if self.with_predeblur:
# #             feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
# #             if self.hr_in:
# #                 h, w = h // 4, w // 4
# #         else:
# #             feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
# #
# #         feat_l1 = self.feature_extraction(feat_l1)
# #         # L2
# #         feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
# #         feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
# #         # L3
# #         feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
# #         feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))
# #
# #         feat_l1 = feat_l1.view(b, t, -1, h, w)
# #         feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
# #         feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)
# #
# #         # PCD alignment
# #         ref_feat_l = [  # reference feature list
# #             feat_l1[:, self.center_frame_idx, :, :, :].clone(),
# #             feat_l2[:, self.center_frame_idx, :, :, :].clone(),
# #             feat_l3[:, self.center_frame_idx, :, :, :].clone()
# #         ]
# #         aligned_feat = []
# #         for i in range(t):
# #             nbr_feat_l = [  # neighboring feature list
# #                 feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
# #                 feat_l3[:, i, :, :, :].clone()
# #             ]
# #             aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
# #         aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)
# #
# #         if not self.with_tsa:
# #             aligned_feat = aligned_feat.view(b, -1, h, w)
# #         feat = self.fusion(aligned_feat)
# #
# #         out = self.reconstruction(feat)
# #         out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
# #         out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
# #         out = self.lrelu(self.conv_hr(out))
# #         out = self.conv_last(out)
# #         if self.hr_in:
# #             base = x_center
# #         else:
# #             base = F.interpolate(
# #                 x_center, scale_factor=4, mode='bilinear', align_corners=False)
# #         out += base
# #         return out
