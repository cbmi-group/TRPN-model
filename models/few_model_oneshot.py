
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

currentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(currentdir)

from utils.base_module import DoubleConv, Down, Up, OutConv
from utils.utils import init_weights


class FS2_One_Shot_V1(nn.Module):
    def __init__(self, in_channels=1, base_dim=64, n_classes=1):
        super(FS2_One_Shot_V1, self).__init__()
        self.feature = UNet(input_channels=in_channels, base_dim=base_dim)
        self.n_classes = n_classes
        # segmentation head
        self.seg_layers = DoubleConv(base_dim+1,  base_dim)
        # self.seg_layers = DoubleConv(base_dim,  base_dim)
        self.prediction_layer = OutConv(base_dim, n_classes)
        # self.fusion_layer = OutConv(4, n_classes)
        self.cos_similarity_func = nn.CosineSimilarity()

    def forward(self, data_dict, skl=False):
        if skl:
            query_img, support_img, support_mask, support_skl = \
                data_dict['query image'], data_dict['support image'], data_dict['support mask'], data_dict['support skl']
        else:
            query_img, support_img, support_mask = \
                data_dict['query image'], data_dict['support image'], data_dict['support mask']

        support_feats_4, _, _, _ = self.feature(support_img)
        if skl:
            support_prototypes_4 = torch.sum(torch.sum(support_feats_4*support_skl, dim=3), dim=2)/torch.sum(torch.sum(torch.sum(support_skl, dim=3), dim=2), dim=1, keepdim=True)
        else:
            support_prototypes_4 = torch.sum(torch.sum(support_feats_4*support_mask, dim=3), dim=2)/torch.sum(torch.sum(torch.sum(support_mask, dim=3), dim=2), dim=1, keepdim=True)
            
        support_prototypes_4 = support_prototypes_4.unsqueeze(dim=2).unsqueeze(dim=3)

        query_feats_4, _, _, _ = self.feature(query_img)
        cosine_similarity_prototype_4 = self.cos_similarity_func(query_feats_4, support_prototypes_4)
        exit_feat_in = torch.cat((query_feats_4, cosine_similarity_prototype_4.unsqueeze(dim=1)), dim=1)

        logits = self.prediction_layer(self.seg_layers(exit_feat_in))

        if self.n_classes == 1:
            logits = torch.sigmoid(logits)

        return logits, support_feats_4, query_feats_4


class FS2_One_Shot_V2(nn.Module):
    def __init__(self, in_channels=1, base_dim=64, n_classes=1):
        super(FS2_One_Shot_V2, self).__init__()
        self.feature = UNet(input_channels=in_channels, base_dim=base_dim)
        self.n_classes = n_classes
        # segmentation head
        self.seg_layers = DoubleConv(base_dim,  base_dim)
        # self.seg_layers = DoubleConv(base_dim,  base_dim)
        self.prediction_layer = OutConv(base_dim, n_classes)
        # self.fusion_layer = OutConv(4, n_classes)
        self.cos_similarity_func = nn.CosineSimilarity()

    def forward(self, data_dict, skl=False):
        if skl:
            query_img, support_img, support_mask, support_skl = \
                data_dict['query image'], data_dict['support image'], data_dict['support mask'], data_dict['support skl']
        else:
            query_img, support_img, support_mask = \
                data_dict['query image'], data_dict['support image'], data_dict['support mask']

        support_feats_4, _, _, _ = self.feature(support_img)
        if skl:
            support_prototypes_4 = torch.sum(torch.sum(support_feats_4*support_skl, dim=3), dim=2)/torch.sum(torch.sum(torch.sum(support_skl, dim=3), dim=2), dim=1, keepdim=True)
        else:
            support_prototypes_4 = torch.sum(torch.sum(support_feats_4*support_mask, dim=3), dim=2)/torch.sum(torch.sum(torch.sum(support_mask, dim=3), dim=2), dim=1, keepdim=True)
            
        support_prototypes_4 = support_prototypes_4.unsqueeze(dim=2).unsqueeze(dim=3)

        query_feats_4, _, _, _ = self.feature(query_img)
        cosine_similarity_prototype_4 = self.cos_similarity_func(query_feats_4, support_prototypes_4)

        exit_feat_in = query_feats_4 * cosine_similarity_prototype_4.unsqueeze(dim=1)

        logits = self.prediction_layer(self.seg_layers(exit_feat_in))

        if self.n_classes == 1:
            logits = torch.sigmoid(logits)

        return logits, support_feats_4, query_feats_4


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