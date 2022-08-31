
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

currentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(currentdir)

from utils.base_module import DoubleConv, Down, Up, OutConv
from utils.utils import init_weights

import ipdb


def make_layer2(input_feature, out_feature, up_scale=1, ksize=3, d=1, groups=1):
    p = int((ksize - 1) / 2)
    if up_scale == 1:
        return nn.Sequential(
        nn.InstanceNorm2d(input_feature),
        nn.ReLU(),
        nn.Conv2d(input_feature, out_feature, ksize, padding=p, dilation=d, groups=groups),
    )
    return nn.Sequential(
        nn.InstanceNorm2d(input_feature),
        nn.ReLU(),
        nn.Conv2d(input_feature, out_feature, ksize, padding=p),
        nn.UpsamplingBilinear2d(scale_factor=up_scale),
    )

class ResBlock2(nn.Module):
    def __init__(self, input_feature, planes, dilated=1, group=1):
        super(ResBlock2, self).__init__()
        self.conv1 = nn.Conv2d(input_feature, planes, kernel_size=1, bias=False, groups=group)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1 * dilated, bias=False, dilation=dilated, groups=group)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, input_feature, kernel_size=1, bias=False, groups=group)
        self.bn3 = nn.InstanceNorm2d(input_feature)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class FS2_One_Shot_RANet(nn.Module):
    def __init__(self, in_channels=1, base_dim=64, n_classes=1):
        super(FS2_One_Shot_RANet, self).__init__()
        self.feature = UNet(input_channels=in_channels, base_dim=base_dim)
        self.n_classes = n_classes
        self.cos_similarity_func = nn.CosineSimilarity()

        # segmentation head
        self.seg_layers = DoubleConv(193,  base_dim)
        self.prediction_layer = OutConv(base_dim, n_classes)
        
        # ranking layer
        # TODO: 4096 is trivial
        self.Ranking = nn.Sequential(make_layer2(4096, 128), ResBlock2(128, 32), make_layer2(128, 1))
        self.p_1 = make_layer2(128, 128)
        self.res_1 = ResBlock2(128, 128, 1)
        self.p_2 = make_layer2(128, 64)

    def corr_fun(self, support_feat, query_feat):
        size = support_feat.size()
        if size[0] == 1:
            support_feat = support_feat.view(size[1], size[2] * size[3]).transpose(0, 1)
            support_feat = support_feat.unsqueeze(2).unsqueeze(3)
            corr = F.conv2d(query_feat, support_feat.contiguous())
        else:
            CORR = []
            for i in range(size[0]):
                ker = query_feat[i:i+1]
                fea = support_feat[i:i+1]
                ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)
                ker = ker.unsqueeze(2).unsqueeze(3)
                co = F.conv2d(fea, ker.contiguous())
                CORR.append(co)
            corr = torch.cat(CORR, 0)
        return corr

    
    def correlation(self, support_feats, query_feats, support_mask, scale_factor=1):
        sup_feat_norm = F.interpolate(support_feats, scale_factor=scale_factor)
        que_feat_norm = F.interpolate(query_feats, scale_factor=scale_factor)

        h, w = sup_feat_norm.size()[-2::]
        c_size = w * h
        fore_mask = F.interpolate(support_mask.detach(), scale_factor=scale_factor, mode='nearest')
        back_mask = 1 - fore_mask

        corr = self.corr_fun(sup_feat_norm, que_feat_norm) # 1 x (H0xW0) x H x W
        corr_fore = corr * fore_mask.view(-1, c_size, 1, 1)
        corr_back = corr * back_mask.view(-1, c_size, 1, 1)

        # CNN layer in RAM
        # 1 x (H0xW0) x H x W --> 1 x (HxW) x H0 x W0
        T_corr_fore = corr_fore.permute(0, 2, 3, 1).view(-1, c_size, h, w)
        T_corr_back = corr_back.permute(0, 2, 3, 1).view(-1, c_size, h, w)
        # 1 x 1 x H0 x W0
        R_map_fore = (F.relu(self.Ranking(T_corr_fore)) * fore_mask).view(-1, 1, c_size)
        R_map_back = (F.relu(self.Ranking(T_corr_back)) * back_mask).view(-1, 1, c_size)

        # ranking 
        co_size = corr_fore.size()[2::] # H0 x W0
        max_only, indices = F.max_pool2d(corr_fore, co_size, return_indices=True)
        max_only = max_only.view(-1, 1, c_size) + R_map_fore
        m_sorted, m_sorted_idx = max_only.sort(descending=True, dim=2)
        corr_fore = torch.cat([co.index_select(0, m_sort[0, 0:128]).unsqueeze(0) for co, m_sort in zip(corr_fore, m_sorted_idx)])

        max_only_back, indices = F.max_pool2d(corr_back, co_size, return_indices=True)
        max_only_back = max_only_back.view(-1, 1, c_size) + R_map_back
        m_sorted, m_sorted_idx = max_only_back.sort(descending=True, dim=2)
        corr_back = torch.cat([co.index_select(0, m_sort[0, 0:128]).unsqueeze(0) for co, m_sort in zip(corr_back, m_sorted_idx)])

        # merge net
        fore_corr_feats = self.p_2(self.res_1(self.p_1(F.interpolate(corr_fore, scale_factor=1./scale_factor, mode='bilinear', align_corners=True))))
        back_corr_feats = self.p_2(self.res_1(self.p_1(F.interpolate(corr_back, scale_factor=1./scale_factor, mode='bilinear', align_corners=True))))
        return fore_corr_feats, back_corr_feats


    def forward(self, data_dict, skl=False):
        query_img, support_img, support_mask = data_dict['query image'], data_dict['support image'], data_dict['support mask']

        # support features
        support_feats_4, _, _, _ = self.feature(support_img)

        # prototypes
        support_prototypes_4 = torch.sum(torch.sum(support_feats_4*support_mask, dim=3), dim=2)/torch.sum(torch.sum(torch.sum(support_mask, dim=3), dim=2), dim=1, keepdim=True)
        support_prototypes_4 = support_prototypes_4.unsqueeze(dim=2).unsqueeze(dim=3)

        # query features
        query_feats_4, _, _, _ = self.feature(query_img)

        # prototype similarity 
        cosine_similarity_prototype_4 = self.cos_similarity_func(query_feats_4, support_prototypes_4)

        # ranet: input(support_feats3, query_feats_3, support mask)
        fore_sim, back_sim = self.correlation(support_feats_4, query_feats_4, support_mask)
        
        # fusion feats
        # 64 + 1 + 64 + 64
        exit_feat_in = torch.cat((query_feats_4, cosine_similarity_prototype_4.unsqueeze(dim=1), fore_sim, back_sim), dim=1)

        # segmentation layer
        logits = self.prediction_layer(self.seg_layers(exit_feat_in))

        if self.n_classes == 1:
            logits = torch.sigmoid(logits)

        return logits, _, _


class UNet(nn.Module):
    def __init__(self, input_channels=1, base_dim=64, bilinear=False):
        super(UNet, self).__init__()
        
        self.bilinear = bilinear
        self.fuse_mode = 'cat'

        # encoder
        self.inc = DoubleConv(input_channels, base_dim)
        self.down1 = Down(base_dim, base_dim*2)
        self.down2 = Down(base_dim*2, base_dim*4)
        self.down3 = Down(base_dim*4, base_dim*8)
        self.down4 = Down(base_dim*8, base_dim*16)

        # prototype branch
        self.up1 = Up(base_dim*16, base_dim*8, bilinear, fuse_mode=self.fuse_mode)
        self.up2 = Up(base_dim*8,  base_dim*4, bilinear, fuse_mode=self.fuse_mode)
        self.up3 = Up(base_dim*4,  base_dim*2, bilinear, fuse_mode=self.fuse_mode)
        self.up4 = Up(base_dim*2,  base_dim,   bilinear, fuse_mode=self.fuse_mode)
        
    def forward(self, x):

        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoders
        # prototype guidance branch
        o_m_4 = self.up1(x5, x4)
        o_m_3 = self.up2(o_m_4, x3)
        o_m_2 = self.up3(o_m_3, x2)
        o_m_1 = self.up4(o_m_2, x1)

        return o_m_1, o_m_2, o_m_3, o_m_4
