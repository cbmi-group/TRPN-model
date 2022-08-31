
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
from torch import nn
from torch.utils.data import DataLoader

from models import model_dict_segmentation
from datasets.data_loader import ER_DataLoader

GPU_ID = 2
os.environ['CUDA_VISIBLE_DEVICE'] = str(GPU_ID)
torch.cuda.set_device(GPU_ID)

def test(args):

    batch_size = args.batch_size
    split = args.split
    test_ckpt_epoch = args.test_ckpt_epoch
    network = args.network
    base_dim = args.base_dim

    data_type, txt_name = os.path.split(args.data_list)
    data_type = data_type.split("_")[0]
    data_list_dir = os.path.join("./data/data_list", args.data_list)
    ckpt_dir = os.path.join("./data/train_logs", args.train_dir, "checkpoints")

    print("==> Create model.")
    if network in model_dict_segmentation.keys():
        model = model_dict_segmentation[network](base_dim=base_dim, n_classes=args.n_classes, input_channels=1)
    else:
        print("No support model type.")
        sys.exit(1)
    
    print("==> Load data.")
    if not os.path.exists(data_list_dir):
        print("No file list found.")
        sys.exit(1)
    with open(data_list_dir, 'r') as fid:
        lines = fid.readlines()
    
    test_dataset = ER_DataLoader(img_list=data_list_dir, 
                                 split=split, 
                                 equalize=args.equalize, 
                                 standardization=args.std, 
                                 skl=args.skl, 
                                 img_size=args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, drop_last=False, shuffle=False)

    resume = os.path.join(ckpt_dir, "checkpoints_epoch_{}.pth".format(test_ckpt_epoch))
    ckpt_file = os.path.splitext(os.path.split(resume)[-1])[0]
    result_dir = os.path.join('./data/results', data_type+"-"+txt_name, args.train_dir, ckpt_file.split(".")[0])
    os.makedirs(result_dir, exist_ok=True)

    print("==> Load weights %s." % (ckpt_file))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model = nn.DataParallel(model, device_ids=[3])
    model.eval()

    # plt.figure()
    t1 = 0
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            
            images, IDs = inputs["image"].cuda(), inputs["ID"]
            t0 = time.time()
            mask_preds = model(images)
            t1 += (time.time()-t0)

            for j in range(images.shape[0]):
                _, image_name = os.path.split(IDs[j])
                print(' --> working on image  %s.' % (image_name))
                result = mask_preds[j].squeeze().detach().cpu().numpy()
                np.save(os.path.join(result_dir, image_name.replace('tif', 'npy')), result)     

        print('Runtime: %.2f ms/image' % (t1*1000/len(lines)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='experiments/train_seg_unet-nucleus-train_fiveshot.txt-2021619_2212')
    parser.add_argument('--data_list', type=str, default='nucleus/test.txt') 
    parser.add_argument('--test_ckpt_epoch', type=int, default=150)

    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--network', type=str, default='unet')
    parser.add_argument('--base_dim', type=int, default=64)
    parser.add_argument('--equalize', action='store_true', default=True)
    parser.add_argument('--std', action='store_true', default=False)
    parser.add_argument('--skl', action='store_true', default=False)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    test(args)


