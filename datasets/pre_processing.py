import os
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt


def rgb_standarization(img, axis=(0,1)):
    mean = np.mean(img, axis=axis, keepdims=True)
    std = np.sqrt(((img - mean)**2).mean(axis=axis, keepdims=True))
    out = (img - mean) / (std + 1e-10)
    return out
    

def gray_standarization(img):
    return (img-np.mean(img)) / (np.std(img)+1e-10)


def contrast_stretch(img):
    I_strech = gray_standarization(img)
    I_strech = (I_strech-np.amin(I_strech))/(np.amax(I_strech)-np.amin(I_strech)+1e-10)
    if img.dtype == np.uint8:
        return (I_strech*255).astype(np.uint8)
    elif img.dtype == np.uint16 or img.dtype == np.float32:
        return (I_strech*65535).astype(np.uint16)


def hist_equalization(img):
    colors = len(img.shape)
    if colors == 2:
        img_equalized = cv2.equalizeHist(img)
    else:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img_yuv[...,0] = cv2.equalizeHist(img_yuv[...,0])
        img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YCR_CB2BGR)
    return img_equalized


def clahe_equalized(img, clipLimit=1, grid_size=4):
    assert img.dtype == np.uint16 or img.dtype == np.uint8
    colors = len(img.shape)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid_size,grid_size))
    if colors == 2:
        img_equalized = clahe.apply(img)
    else:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img_yuv[...,0] = clahe.apply(img_yuv[...,0])
        img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YCR_CB2BGR)
    return img_equalized


def adjust_gamma(img, gamma=1.2, mode='uint8'):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    new_img = cv2.LUT(img, table)
    return new_img


def gaussian_blur(img, kz=5, sigma=1):
    return cv2.GaussianBlur(img, ksize=(kz,kz), sigmaX=sigma)



if __name__ == "__main__":

    img_dir = "data/er-sheet-dataset/train/tubule_images_aug_v2"
    img_list = glob.glob(os.path.join(img_dir, "*.tif"))

    plt.figure(figsize=(8,4))
    for img in img_list:
        I = cv2.imread(img, -1)

        I1 = gaussian_blur(I)

        I2 = clahe_equalized(I1)

        plt.subplot(1,3,1)
        plt.imshow(I, cmap='gray')

        plt.subplot(1,3,2)
        plt.imshow(I1, cmap='gray')
       
        plt.subplot(1,3,3)
        plt.imshow(I2, cmap='gray')

        plt.draw()
        plt.pause(1)

    

