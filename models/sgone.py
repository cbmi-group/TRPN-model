
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models.vgg_sg as vgg


class OneModel(nn.Module):
    def __init__(self, n_classes=2, in_channels=1):
        super(OneModel, self).__init__()

        self.netB = vgg.vgg16(pretrained=True, use_decoder=True, in_channels=in_channels)

        self.classifier_6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, dilation=1,  padding=1),   #fc6
            nn.ReLU(inplace=True)
        )
        self.exit_layer = nn.Conv2d(128, n_classes, kernel_size=1, padding=1)

        # self.bce_logits_func = nn.BCEWithLogitsLoss()
        self.bce_logits_func = nn.CrossEntropyLoss()
        self.loss_func = nn.BCELoss()
        self.cos_similarity_func = nn.CosineSimilarity()
        self.triplelet_func = nn.TripletMarginLoss(margin=2.0)

    def forward(self, data_dict):
        # pos_img:    support image
        # anchor_img: query image
        anchor_img, pos_img, pos_mask = data_dict['query image'], data_dict['support image'], data_dict['support mask']

        # extract features from support image
        # outA_pos:  original vgg feature for prototype learning
        outA_pos, _ = self.netB(pos_img)

        _, _, mask_w, mask_h = pos_mask.size()

        # upsample 8 times larger to original image size
        outA_pos = F.interpolate(outA_pos, size=(mask_w, mask_h), mode='bilinear')

        # compute prototypes from support images 
        vec_pos = torch.sum(torch.sum(outA_pos*pos_mask, dim=3), dim=2)/torch.sum(pos_mask)

        # outB: protoype features of query image
        # outB_side: fused featured for segmentation of query image
        outB, outB_side= self.netB(anchor_img)

        # compute consine similarity between support and query image
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
        tmp_seg = self.cos_similarity_func(outB, vec_pos)

        # combine prototypes and fused vgg features of query image
        exit_feat_in = outB_side * tmp_seg.unsqueeze(dim=1)

        # predictions
        outB_side_6 = self.classifier_6(exit_feat_in)
        outB_side = self.exit_layer(outB_side_6)

        return outB_side

    def forward_5shot_avg(self, anchor_img, pos_img_list, pos_mask_list):
        vec_pos_sum = 0.
        for i in range(5):
            pos_img = pos_img_list[i]
            pos_mask = pos_mask_list[i]

            pos_img = self.warper_img(pos_img)
            pos_mask = self.warper_img(pos_mask)

            outA_pos, _ = self.netB(pos_img)

            _, _, mask_w, mask_h = pos_mask.size()
            outA_pos = F.upsample(outA_pos, size=(mask_w, mask_h), mode='bilinear')
            vec_pos = torch.sum(torch.sum(outA_pos*pos_mask, dim=3), dim=2)/torch.sum(pos_mask)
            vec_pos_sum += vec_pos

        vec_pos = vec_pos_sum/5.0

        outB, outB_side = self.netB(anchor_img)

        # tmp_seg = outB * vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
        tmp_seg = self.cos_similarity_func(outB, vec_pos)

        exit_feat_in = outB_side * tmp_seg.unsqueeze(dim=1)
        outB_side_6 = self.classifier_6(exit_feat_in)
        outB_side = self.exit_layer(outB_side_6)

        return outB, outA_pos, vec_pos, outB_side

    def warper_img(self, img):
        img_tensor = torch.Tensor(img).cuda()
        img_var = Variable(img_tensor)
        img_var = torch.unsqueeze(img_var, dim=0)
        return img_var

    def forward_5shot_max(self, anchor_img, pos_img_list, pos_mask_list):
        outB_side_list = []
        for i in range(5):
            pos_img = pos_img_list[i]
            pos_mask = pos_mask_list[i]

            pos_img = self.warper_img(pos_img)
            pos_mask = self.warper_img(pos_mask)

            outA_pos, _ = self.netB(pos_img)

            _, _, mask_w, mask_h = pos_mask.size()
            outA_pos = F.upsample(outA_pos, size=(mask_w, mask_h), mode='bilinear')
            vec_pos = torch.sum(torch.sum(outA_pos*pos_mask, dim=3), dim=2)/torch.sum(pos_mask)

            outB, outB_side = self.netB(anchor_img)

            # tmp_seg = outB * vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
            vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
            tmp_seg = self.cos_similarity_func(outB, vec_pos)

            exit_feat_in = outB_side * tmp_seg.unsqueeze(dim=1)
            outB_side_6 = self.classifier_6(exit_feat_in)
            outB_side = self.exit_layer(outB_side_6)

            outB_side_list.append(outB_side)

        return outB, outA_pos, vec_pos, outB_side_list

    def get_loss(self, outB_side, query_label):
        b, w, h = query_label.size()
        outB_side = F.interpolate(outB_side, size=(w, h), mode='bilinear')
        loss = self.bce_logits_func(outB_side, query_label.long())
        return loss

    def get_pred_5shot_max(self, logits, query_label):
        outB, outA_pos, vec_pos, outB_side_list = logits

        w, h = query_label.size()[-2:]
        res_pred = None
        for i in range(5):
            outB_side = outB_side_list[i]
            outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
            out_side = F.softmax(outB_side, dim=1).squeeze()
            values, pred = torch.max(out_side, dim=0)

            if res_pred is None:
                res_pred = pred
            else:
                res_pred = torch.max(pred, res_pred)

        return values, res_pred

    def get_pred(self, logits, query_image):
        outB, outA_pos, vec_pos, outB_side = logits

        w, h = query_image.size()[-2:]
        outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
        out_softmax = F.softmax(outB_side, dim=1).squeeze()
        values, pred = torch.max(out_softmax, dim=0)

        return out_softmax, pred


