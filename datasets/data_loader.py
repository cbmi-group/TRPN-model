
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


class ER_DataLoader(data.Dataset):
    def __init__(self, 
                     img_list='', 
                     in_dim=1, 
                     split='train', 
                     equalize=False,
                     standardization=False,
                     transform=False, 
                     skl=False,
                     img_size=256):
        super(ER_DataLoader, self).__init__()
        self.split = split
        self.in_dim = in_dim
        self.transform = transform
        self.equalize=equalize
        self.standardization=standardization
        self.filelist = img_list
        self.img_size = img_size
        self.skl = skl
        
        with open(img_list, "r") as fid:
            lines = fid.readlines()

        file_paths = []
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(" ")
            file_paths.append(words[0])
        
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        sample = dict()
        img_path = self.file_paths[index]
        if self.split == "train" or self.split == 'val':
            mask_path = img_path.replace('images', 'masks')
            mask = cv2.imread(mask_path, -1)
            mask[mask==255] = 1
            mask = cv2.resize(mask, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            if self.skl:
                mask_skl = skeletonize(mask, method='lee')
            
        # read image
        orig_img = cv2.imread(img_path, -1)
        if len(orig_img.shape) > 2:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        img_type = orig_img.dtype

        # normalization
        if self.equalize:
            orig_img = clahe_equalized(orig_img)

        if img_type == np.uint16:
            img = orig_img / 65535.
        elif img_type == np.uint8:
            img = orig_img / 255.
        
        # standardization
        if self.standardization:
            img = gray_standarization(img)

        img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        sample["ID"] = img_path
        img = img[np.newaxis, :, :]
        if self.split != 'test':
            mask = mask[np.newaxis, :, :]
            sample["mask"] = torch.from_numpy(mask).float() 
        sample["image"] = torch.from_numpy(img).float()
        orig_img = cv2.resize(orig_img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        sample["orig_image"] = torch.from_numpy(orig_img.astype(np.float)).float()   
        if self.skl and self.split != 'test':
            sample["skl"] = torch.from_numpy(mask_skl[np.newaxis,:,:]).float() 

        return sample


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
                     semi=False):

        super(ER_DataLoader_Pair, self).__init__()
        self.split = split
        self.in_dim = in_dim
        self.transform = transform
        self.equalize=equalize
        self.standardization=standardization
        self.support_list = support_list
        self.query_list = query_list
        self.img_size = img_size
        self.skl = skl
        self.semi = semi
        
        with open(support_list, "r") as fid:
            lines = fid.readlines()
        support_paths = []
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(" ")
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

    def __getitem__(self, index):
        sample = dict()
        
        query_img_path = self.query_paths[index]
        
        # if self.split == 'train':
        #     support_img_path = self.support_paths[index]
        # else:
        #     support_img_path = self.support_paths[0]
        
        support_img_path = self.support_paths[index]    

        support_mask_path, query_mask_path = support_img_path.replace('images', 'masks'), query_img_path.replace('images', 'masks')
        
        ##### prepare mask annotation
        support_mask = cv2.imread(support_mask_path, 0)
        query_mask = cv2.imread(query_mask_path, 0)
        support_mask[support_mask==255] = 1
        query_mask[query_mask==255] = 1
        support_mask = cv2.resize(support_mask, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        query_mask = cv2.resize(query_mask, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.skl:
            support_skl = skeletonize(support_mask, method='lee')
            query_skl = skeletonize(query_mask, method='lee')
        #####
        
        ##### read image
        support_img = cv2.imread(support_img_path, -1)
        if len(support_img.shape) > 2:
            support_img = cv2.cvtColor(support_img, cv2.COLOR_BGR2GRAY)
        query_img = cv2.imread(query_img_path, -1)
        if len(query_img.shape) > 2:
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)      
              
        # normalization
        if self.equalize:
            support_img = clahe_equalized(support_img)
            query_img = clahe_equalized(query_img)

        if support_img.dtype == np.uint16:
            support_img = support_img / 65535.
        elif support_img.dtype == np.uint8:
            support_img = support_img / 255.

        if query_img.dtype == np.uint16:
            query_img = query_img / 65535.
        elif query_img.dtype == np.uint8:
            query_img = query_img / 255.

        # standardization
        if self.standardization:
            support_img = gray_standarization(support_img)
            query_img = gray_standarization(query_img)

        support_img = cv2.resize(support_img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        query_img = cv2.resize(query_img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        #####

        sample["Support_ID"] = support_img_path
        sample["Query_ID"] = query_img_path
        sample["support image"] = torch.from_numpy(support_img[np.newaxis, :, :]).float()
        sample["support mask"] = torch.from_numpy(support_mask[np.newaxis, :, :]).float()  
        sample["query image"] = torch.from_numpy(query_img[np.newaxis, :, :]).float()
        sample["query mask"] = torch.from_numpy(query_mask[np.newaxis, :, :]).float()  
        if self.skl:
            sample["support skl"] = torch.from_numpy(support_skl[np.newaxis,:,:]).float()
            sample["query skl"] = torch.from_numpy(query_skl[np.newaxis,:,:]).float()
            
        return sample
        

if __name__ == "__main__":
    
    dataset = ER_DataLoader_Pair(support_list='./data/data_list/mito_widefield/train_oneshot_support_for_er.txt', 
                                 query_list='./data/data_list/mito_widefield/train_oneshot_support_for_er.txt',
                                 in_dim=1, 
                                 transform=False, 
                                 split="train")
    data_loader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    print(len(data_loader))

    for epoch in range(1):
        for i, inputs in enumerate(data_loader):
            print("Epoch: %d Batch %d \n %s %s." % (epoch, i, inputs['Support_ID'], inputs['Query_ID']))