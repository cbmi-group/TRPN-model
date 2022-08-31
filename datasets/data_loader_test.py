
import os
import sys
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
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

class ER_DataLoader_Test(data.Dataset):
    def __init__(self, img_list='', 
                     img_dir='', 
                     equalize=False,
                     standardization=False,
                     transform=False, 
                     in_size=256, out_size=256):
        super(ER_DataLoader_Test, self).__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.equalize = equalize
        self.standardization = standardization
        self.filelist = img_list
        self.in_size = in_size
        self.out_size = out_size
        self.data_type = os.path.split(img_list)[-1]
        
        with open(img_list, "r") as fid:
            lines = fid.readlines()

        file_paths = []
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(" ")
            file_paths.append((words[0], words[1]))
        
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        sample = dict()
        words = self.file_paths[index]
        support_image_path = os.path.join(self.img_dir, 'images', words[0])
        query_image_path = os.path.join(self.img_dir, 'images', words[1])
        support_mask_path = support_image_path.replace('images', 'masks')
        query_mask_path = query_image_path.replace('images', 'masks')
        
        # prepare mask annotation
        support_mask = cv2.imread(support_mask_path, 0)
        query_mask = cv2.imread(query_mask_path, 0)
        support_mask = cv2.resize(support_mask, dsize=(self.out_size, self.out_size), interpolation=cv2.INTER_NEAREST)
        query_mask = cv2.resize(query_mask, dsize=(self.out_size, self.out_size), interpolation=cv2.INTER_NEAREST)

        # read image
        orig_img_support = cv2.imread(support_image_path, -1)
        orig_img_query = cv2.imread(query_image_path, -1)

        # normalization
        if self.equalize:
            orig_img_support = clahe_equalized(orig_img_support)
            orig_img_query = clahe_equalized(orig_img_query)

        if orig_img_support.dtype == np.uint16:
            img_support = orig_img_support / 65535.
            img_query = orig_img_query / 65535.
        elif orig_img_support.dtype == np.uint8:
            img_support = orig_img_support / 255.
            img_query = orig_img_query / 255.
        
        # standardization
        if self.standardization:
            img_support = gray_standarization(img_support)
            img_query = gray_standarization(img_query)

        img_support = cv2.resize(img_support, dsize=(self.in_size, self.in_size), interpolation=cv2.INTER_LINEAR)
        img_query = cv2.resize(img_query, dsize=(self.in_size, self.in_size), interpolation=cv2.INTER_LINEAR)

        sample["Support ID"] = words[0]
        sample["Query ID"] = words[1]
        img_support = img_support[np.newaxis, :, :]
        support_mask = support_mask[np.newaxis, :, :]
        sample["support image"] = torch.from_numpy(img_support).float()  
        sample["support mask"] = torch.from_numpy(support_mask).float()
        img_query = img_query[np.newaxis, :, :]
        query_mask = query_mask[np.newaxis, :, :]
        sample["query image"] = torch.from_numpy(img_query).float()  
        sample["query mask"] = torch.from_numpy(query_mask).float()

        return sample

        
if __name__ == "__main__":
    
    dataset = ER_DataLoader_Test(img_list='./data/data_list/er_sheet_tubule_confocal/test_01.txt',
                                 img_dir='./data/fewshot_dataset/test/er_sheet_tubule_confocal')
    data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    print(len(data_loader))
    for epoch in range(1):
        for i, inputs in enumerate(data_loader):
            print("Epoch: %d Batch %d: \n %s \n %s." % (epoch, i, inputs['Support ID'], inputs['Query ID']))

            