
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

from datasets.data_loader import ER_DataLoader_Pair
from models import model_dict_segmentation

GPU_ID = 1
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
    ckpt = os.path.join(ckpt_dir, "checkpoints_epoch_{}.pth".format(args.test_ckpt_epoch))
    # ckpt = os.path.join(ckpt_dir, "checkpoints_best.pth")

    print("==> Create model.")
    if network in model_dict_segmentation.keys():
        model = model_dict_segmentation[network](base_dim=base_dim, n_classes=args.n_classes, in_channels=1)
    else:
        print("No support model type.")
        sys.exit(1)
    
    print("==> Load data.")
    dataset = ER_DataLoader_Pair(support_list=support_list,
                                    query_list=query_list,
                                    split='val',
                                    equalize=args.equalize,
                                    standardization=args.std,
                                    skl=False,
                                    img_size=args.img_size)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False, drop_last=False)

    ckpt_file = os.path.splitext(os.path.split(ckpt)[-1])[0]
    result_dir = os.path.join('./data/results', data_type+"-"+data_name, args.train_dir, ckpt_file.split(".")[0])
    os.makedirs(result_dir, exist_ok=True)

    print("==> Load weights %s." % (ckpt_file))
    checkpoint = torch.load(ckpt)
    print(checkpoint['epoch'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    t1 = 0
    with torch.no_grad():
        model.eval()
        for i, data_dict in enumerate(test_loader):
            t0 = time.time()
            data_dict["support image"] = data_dict["support image"].cuda()
            data_dict["support mask"] = data_dict["support mask"].cuda()
            data_dict['query image'] = data_dict['query image'].cuda()
            mask_out, _, _ = model(data_dict)
            for j in range(data_dict['query image'].size(0)):
                image_name = os.path.split(data_dict["Query_ID"][j])[-1]
                print(' --> working on image %s.' % (image_name))
                result = mask_out[j].squeeze().cpu().numpy()
                np.save(os.path.join(result_dir, image_name.replace('tif', 'npy')), result) 
                print(' --> working on image %s.' % (image_name))

            t1 += (time.time()-t0)

        with open(query_list, 'r') as fid:
            lines = fid.readlines()

        print('Runtime: %.2f ms/image' % (t1*1000/len(lines)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='experiments_20210906/train_ranet_model-mito-train_oneshot_support_for_er.txt-2022516_1213')
    parser.add_argument('--support_list', type=str, default='er_tubule_confocal_64/test_oneshot_support.txt')
    parser.add_argument('--query_list', type=str, default='er_tubule_confocal_64/test.txt')
    parser.add_argument('--test_ckpt_epoch', type=int, default=30)

    parser.add_argument('--network', type=str, default='ranet_model')
    parser.add_argument('--base_dim', type=int, default=64)
    parser.add_argument('--equalize', action='store_true', default=True)
    parser.add_argument('--std', action='store_true', default=False)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--n_classes', type=int, default=1)

    args = parser.parse_args()

    test(args)


