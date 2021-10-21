import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.models.archs.arch_util import (DCNv2Pack, ResidualBlockNoBN,
                                            make_layer)
from SphereNet.spherenet.sphere_cnn import SphereConv2D, SphereMaxPool2D
from NonLocalNet.lib.non_local_dot_product import NONLocalBlock3D_tem

class Channel3DAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Channel3DAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((10, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((10, 1, 1))

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
        self.conv1 = SphereConv2D(inplanes, planes, stride=1)
        self.pool1 = SphereMaxPool2D(stride=2)
        self.relu = nn.LeakyReLU(self.leaky, inplace=True)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride
        self.inplanes = inplanes

    def forward(self, x):

        out = self.conv1(x)
        out = self.pool1(out)
        out = self.relu(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        out = self.relu(out)

        return out


class SpatialFeatureModule(nn.Module):
    def __init__(self, block, layers):
        super(SpatialFeatureModule, self).__init__()
        self.inplanes = 32
        self.spatial_layer1 = nn.Sequential(
            SphereConv2D(3, 32, stride=1),
            SphereMaxPool2D(stride=2),
            nn.LeakyReLU(self.leaky, inplace=True),
        )

        #self.conv1 = SphereConv2D(3, 32, stride=1)
        self.layer1 = self._make_layer(block, 96, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])

        # Sphere3*3: Reduce the channel number and pooling.
        self.spatial_layer2 = nn.Sequential(
            SphereConv2D(480, 120, stride=1),
            SphereMaxPool2D(stride=2),
            nn.LeakyReLU(self.leaky, inplace=True),
        )
        # conv1*1: Reduce the channel number
        self.spatial_layer3 = nn.Sequential(
            nn.Conv2d(120, 60, 1, bias=False),
            #SphereMaxPool2D(stride=2),
            nn.LeakyReLU(self.leaky, inplace=True),
        )


    def _make_layer(self, block, planes, blocks, stride=1):
        #downsample = None

        layers = []

        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.spatial_layer1(x)
        out1 = self.layer1(x) # (b, 96, 360, 640)
        out2 = self.layer2(out1) # (b, 128, )
        out3 = self.layer3(out2) # (b, 256, )
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.spatial_layer2(out)
        out = self.spatial_layer3(out)

        #x = self.conv1(x)
        return out

class MotionFeatureExtr(nn.Module):
    def __init__(self):
        super(MotionFeatureExtr, self).__init__()
        self.spatial_layer1 = nn.Sequential(
            SphereConv2D(60, 120, stride=1),
            SphereMaxPool2D(stride=2),
            nn.LeakyReLU(self.leaky, inplace=True),
            SphereConv2D(120, 60, stride=1),
            SphereMaxPool2D(stride=2),
            nn.LeakyReLU(self.leaky, inplace=True),
        )
        self.spatial_layer2 = nn.Sequential(
            SphereConv2D(60, 120, stride=1),
            SphereMaxPool2D(stride=2),
            nn.LeakyReLU(self.leaky, inplace=True),
            SphereConv2D(120, 60, stride=1),
            SphereMaxPool2D(stride=2),
            nn.LeakyReLU(self.leaky, inplace=True),
        )
        self.spatial_layer3 = nn.Sequential(
            SphereConv2D(120, 60, stride=1),
            SphereMaxPool2D(stride=2),
            nn.LeakyReLU(self.leaky, inplace=True),
        )
        self.spatial_layer4 = nn.Sequential(
            SphereConv2D(120, 60, stride=1),
            SphereMaxPool2D(stride=2),
            nn.LeakyReLU(self.leaky, inplace=True),
        )
        self.feat_comb_layer = nn.Sequential(
            nn.Conv2d(180, 60, 1, bias=False),
            #SphereMaxPool2D(stride=2),
            nn.LeakyReLU(self.leaky, inplace=True),
        )
    def forward(self, x):
        x_down = x[0, :, :, :, :]
        x_cent = x[1, :, :, :, :]
        x_upon = x[2, :, :, :, :]

        x_dow = x_cent - x_down
        x_up  = x_upon - x_cent

        x_dow = self.spatial_layer1(x_dow)
        x_up  = self.spatial_layer2(x_up)

        x_combin = torch.cat((x_down, x_cent, x_upon), dim=1)
        x_combin_sqz = self.feat_comb_layer(x_combin)

        x_com_dow = torch.cat((x_dow, x_combin_sqz), dim=1)
        x_com_up  = torch.cat((x_up, x_combin_sqz), dim=1)

        x_com_dow_sqz = self.spatial_layer3(x_com_dow)
        x_com_up_sqz = self.spatial_layer4(x_com_up)

        #x_out = x_com_dow_sqz + x_com_up_sqz + x_combin_sqz

        return x_com_dow_sqz + x_com_up_sqz + x_combin_sqz



class TemporalNonLocalModule(nn.Module):
    def __init__(self):
        super(TemporalNonLocalModule, self).__init__()

        self.out = NONLocalBlock3D_tem(60, sub_sample=False, bn_layer=False)

    def forward(self, x):

        return self.out(x) # (b, 60, 10, 180, 320)


class TemporalScoresAggModule(nn.Module):
    def __init__(self):
        super(TemporalScoresAggModule, self).__init__()
        self.spatial_layer1 = nn.Sequential(
            nn.Conv3d(60, 20, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(self.leaky, inplace=True),
        )
        self.maxpool1 = nn.AdaptiveMaxPool3d((10, 90, 160))
        self.avgpool1 = nn.AdaptiveAvgPool3d((10, 90, 160))

        self.spatial_layer2 = nn.Sequential(
            nn.Conv3d(40, 10, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(self.leaky, inplace=True),
        )
        self.maxpool2 = nn.AdaptiveMaxPool3d((10, 90, 160))
        self.avgpool2 = nn.AdaptiveAvgPool3d((10, 90, 160))

        self.pool1 = nn.AdaptiveAvgPool3d((10, 20, 20))

        self.ca3d = Channel3DAttention(20)

        self.pool2 = nn.AdaptiveAvgPool3d((10, 1, 1))

        self.fc = nn.Sequential(
            nn.Linear(200, 16, bias=False),
            nn.LeakyReLU(self.leaky, inplace=True),
            nn.Linear(16, 1),
            nn.LeakyReLU(self.leaky, inplace=True),
        )

    def forward(self, x):  # (b, 60, 10, 180, 320)
        x1 = self.spatial_layer1(x)
        x11 = self.maxpool1(x1)
        x12 = self.avgpool1(x1)
        x_1 = torch.cat((x11, x12), dim=1)  # (b, 40, 10, 90, 160)

        x2 = self.spatial_layer1(x_1)
        x21 = self.maxpool1(x2)
        x22 = self.avgpool1(x2)
        x_2 = torch.cat((x21, x22), dim=1)  # (b, 20, 10, 45, 80)

        x_2 = self.pool1(x_2) # (b, 20, 10, 20, 20)

        x_2 = self.ca3d(x_2) * x_2 # (b, 20, 10, 20, 20)

        x_2 = self.pool2(x_2) # (b, 20, 10, 1, 1)
        x_2 = x_2.view(x_2.size(0), -1) # (b, 200)
        x_2 = self.fc(x_2) # score: 1

        return x_2





class BVQA360(nn.Module):
    def __init__(self):
        super(BVQA360, self).__init__()
        self.spatial_module = SpatialFeatureModule(SphereAttentionBlock, [3, 4, 6])
        self.motion_module = MotionFeatureExtr()
        self.tempotalAgg_module = TemporalScoresAggModule()
    def forward(self, x):
        all_frames_feat = []
        # input: x shape (b, t, c, h, w ) (2, 10, 3, 720, 1280)
        for i in range(0, 30, 3):
            x = x[:, i:i+3, :, :, :]
            x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))
            x = self.spatial_module(x).view(x.size(0), x.size(1), x.size(2), x.size(3), x.size(4))
            x = x.permute(1, 0, 2, 3, 4).contiguous() # (t, b, c, w, h)
            x = self.motion_module(x) # (b, c, w, h)
            all_frames_feat.append(x)

        all_frames = torch.stack(all_frames_feat, dim = 0) # (t, b, c, w, h)
        all_frames = all_frames.permute(1, 2, 0, 3, 4) # (b, c, t, w, h)
        score = self.tempotalAgg_module(all_frames)

        return score





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
