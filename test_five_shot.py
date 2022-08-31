
import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image
import scipy.io as io
from collections import OrderedDict
from matplotlib import pyplot as plt
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from datasets.data_loader_fiveshot import ER_DataLoader_Pair
from models.sgone_unet import OneModel_DGNet
from models.few_model_fiveshot import FewModel_V1

GPU_ID = 2
os.environ['CUDA_VISIBLE_DEVICE'] = str(GPU_ID)
torch.cuda.set_device(GPU_ID)

def test(args):

    test_ckpt_epoch = args.test_ckpt_epoch
    network = args.network
    base_dim = args.base_dim
    support_list = os.path.join('./data/data_list', args.support_list)
    query_list = os.path.join('./data/data_list', args.query_list)

    data_type, data_name = os.path.split(args.query_list)
    data_type = data_type.split("_")[0]

    train_dir = args.train_dir

    ckpt_dir = os.path.join("./data/train_logs", train_dir, "checkpoints")
    # ckpt = os.path.join(ckpt_dir, "checkpoints_epoch_{}.pth".format(args.test_ckpt_epoch))
    ckpt = os.path.join(ckpt_dir, "checkpoints_best.pth")

    print("==> Create model.")
    model = FewModel_V1(base_dim=base_dim, n_classes=args.n_classes, in_channels=1)
    
    print("==> Load data.")
    dataset = ER_DataLoader_Pair(support_list=support_list, 
                                 query_list=query_list,
                                 equalize=args.equalize, 
                                 standardization=args.std, 
                                 split='val')
    test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)

    ckpt_file = os.path.splitext(os.path.split(ckpt)[-1])[0]
    result_dir = os.path.join('./data/results', data_type+"-"+data_name, args.train_dir, ckpt_file.split(".")[0])
    os.makedirs(result_dir, exist_ok=True)

    print("==> Load weights %s." % (ckpt_file))
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()

    # plt.figure()
    t1 = 0
    with torch.no_grad():
        for i, data_dict in enumerate(test_loader):
            t0 = time.time()
            data_dict["support image"] = data_dict["support image"].cuda()
            data_dict["support mask"] = data_dict["support mask"].cuda()
            data_dict['query image'] = data_dict['query image'].cuda()
            if args.val_skl:
                data_dict['support skl'] = data_dict['support skl'].cuda()
            mask_out, support_feat, query_feats = model(data_dict, skl=args.val_skl)
            for j in range(data_dict['query image'].size(0)):
                image_name = os.path.split(data_dict["Query_ID"][j])[-1]
                print(' --> working on image %s.' % (image_name))
                result = mask_out[j].squeeze().detach().cpu().numpy()
                np.save(os.path.join(result_dir, image_name.replace('tif', 'npy')), result) 
                print(' --> working on image %s.' % (image_name))

            t1 += (time.time()-t0)

        with open(query_list, 'r') as fid:
            lines = fid.readlines()

        print('Runtime: %.2f ms/image' % (t1*1000/len(lines)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='experiments/train_fiveshot_seg_few_model-nucleus-B-train_fiveshot_support_for_nucleus.txt-2021619_2139')
    parser.add_argument('--support_list', type=str, default='nucleus/test_fiveshot_support.txt')
    parser.add_argument('--query_list', type=str, default='nucleus/test.txt')
    parser.add_argument('--test_ckpt_epoch', type=int, default=30)

    parser.add_argument('--network', type=str, default='few_model')
    parser.add_argument('--base_dim', type=int, default=64)
    parser.add_argument('--equalize', action='store_true', default=True)
    parser.add_argument('--std', action='store_true', default=False)
    parser.add_argument('--in_img_size', type=int, default=256)
    parser.add_argument('--out_img_size', type=int, default=256)
    parser.add_argument('--val_skl', action='store_true', default=False) 
    parser.add_argument('--n_classes', type=int, default=1)

    args = parser.parse_args()

    test(args)


