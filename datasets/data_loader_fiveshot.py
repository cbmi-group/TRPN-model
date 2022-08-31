
import os
import sys
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from skimage import filters
from random import randint, random, randrange
from skimage.morphology import skeletonize

currentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(currentdir)

from datasets.pre_processing import rgb_standarization, \
                                    gray_standarization, \
                                    clahe_equalized, \
                                    adjust_gamma,\
                                    contrast_stretch

from datasets.data_augmentation import vertical_flip, horizontal_flip

augemtation_dict = {'0': 'vertical flip',
                    '1': 'horizontal flip',
                    '2': '90 degree rotation',
                    '3': '180 degree rotation',
                    '4': '270 degree rotation'}


class ER_DataLoader_Pair(data.Dataset):
    def __init__(self, 
                     support_list='', 
                     query_list='',
                     in_dim=1, 
                     split='train', 
                     equalize=False,
                     standardization=False,
                     transform=False, 
                     skl=False,
                     img_size=256,
                     semi=False,
                     fiveshot=True):

        super(ER_DataLoader_Pair, self).__init__()
        self.split = split
        self.in_dim = in_dim
        self.transform = transform
        self.equalize = equalize
        self.standardization=standardization
        self.support_list = support_list
        self.query_list = query_list
        self.img_size = img_size
        self.skl = skl
        self.semi = semi
        self.fiveshot = fiveshot
        
        with open(support_list, "r") as fid:
            lines = fid.readlines()
        support_paths = []
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(" ")
            if fiveshot:
                support_paths.append(words)
            else:
                support_paths.append(words[0])
        
        with open(query_list, "r") as fid:
            lines = fid.readlines()
        query_paths = []
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(" ")
            query_paths.append(words[0])
        
        self.support_paths = support_paths
        self.query_paths = query_paths

    def __len__(self):
        return len(self.query_paths)

    def _load_img_mask(self, img_path, mask_path):
        mask = cv2.imread(mask_path, 0)
        mask[mask>0] = 1
        mask = cv2.resize(mask, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        img = cv2.imread(img_path, -1)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        if self.equalize:
            img = clahe_equalized(img)
            
        if img.dtype == np.uint16:
            img = img / 65535.
        elif img.dtype == np.uint8:
            img = img / 255.

        if self.standardization:
            img = gray_standarization(img)

        img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        return img, mask

    def __getitem__(self, index):
        sample = dict()
        
        query_img_path = self.query_paths[index]
        query_mask_path = query_img_path.replace('images', 'masks')
        support_img_path = self.support_paths[index]
        
        if self.fiveshot:
            s, m = [], []
            for p in support_img_path:
                support_mask_path = p.replace('images', 'masks')
                support_img, support_mask = self._load_img_mask(p, support_mask_path)
                s.append(support_img)
                m.append(support_mask)
            support_img = np.array(s)
            support_mask = np.array(m)
            sample["support image"] = torch.from_numpy(support_img).float()
            sample["support mask"] = torch.from_numpy(support_mask).float() 

        else:
            support_mask_path = support_img_path.replace('images', 'masks')
            support_img, support_mask = self._load_img_mask(support_img_path, support_mask_path)
            sample["support image"] = torch.from_numpy(support_img[np.newaxis, :, :]).float()
            sample["support mask"] = torch.from_numpy(support_mask[np.newaxis, :, :]).float() 

        query_img, query_mask = self._load_img_mask(query_img_path, query_mask_path) 
        sample["query image"] = torch.from_numpy(query_img[np.newaxis, :, :]).float()
        sample["query mask"] = torch.from_numpy(query_mask[np.newaxis, :, :]).float()  
        
        sample["Support_ID"] = support_img_path
        sample["Query_ID"] = query_img_path
            
        return sample
        

if __name__ == "__main__":
    
    dataset = ER_DataLoader_Pair(support_list='./data/data_list/nucleus/test_fiveshot_support.txt', 
                                 query_list='./data/data_list/nucleus/test.txt',
                                 in_dim=1, 
                                 transform=False, 
                                 split="train",
                                 fiveshot=True)
                                 
    data_loader = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, drop_last=True)
    print(len(data_loader))

    for epoch in range(1):
        for i, inputs in enumerate(data_loader):
            print("Epoch: %d Batch %d \n %s %s." % (epoch, i, inputs['Support_ID'], inputs['Query_ID']))
            print(inputs['support image'].size())