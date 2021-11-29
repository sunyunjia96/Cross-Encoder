import torch.utils.data as data
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re
import glob
import random
import cv2
import torch.nn as nn
import math
import random
import scipy.io as sio
import csv

def get_filelist(path,images_same_id):
    dir_or_files = os.listdir(path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(path, dir_file)
        if os.path.isdir(dir_file_path):
            get_filelist(dir_file_path,images_same_id)
        else:
            if dir_file_path[-5:]=='l.png':
                img_path_l = dir_file_path
                img_path_r = dir_file_path[:-5]+'r.png'
                tmp_load_l, tmp_load_r = cv2.imread(img_path_l), cv2.imread(img_path_r)
                if tmp_load_l.shape[0] >= 20 and tmp_load_r.shape[0] >= 20:
                    images_same_id.append([img_path_r,img_path_l])

def make_dataset(data_root):
    images = []
    images_same_id = []

    persons = os.listdir(data_root)
    for p in persons:
        #get file name
        get_filelist(os.path.join(data_root,p),images_same_id)

        images.append(images_same_id)
        images_same_id = []
    
    return images

def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except OSError:
        print(path)
        return Image.new("RGB", (512, 512), "white")

def histeq(im,nbr_bins = 256):
    imhist,bins = np.histogram(im.flatten(),nbr_bins,density= True)
    cdf = imhist.cumsum()
    cdf = 255.0 * cdf / cdf[-1]
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf

def eye_loader(path):
    try:
        im = np.array(Image.open(path).convert('L'))
        im2,cdf = histeq(im)
        im2 = Image.fromarray(np.uint8(im2))
        return im2
    except OSError:
        print(path)
        return Image.new("RGB", (512, 512), "white")

class ImagerLoader(data.Dataset):
    def __init__(self, data_root,transform_E=None, loader=default_loader):

        imgs_ids = make_dataset(data_root)
        self.imgs_ids = imgs_ids

        imgs = []
        for same_id in imgs_ids:
            random.shuffle(same_id)
            for i in range(len(same_id)-1):
                imgs.append([same_id[i],same_id[i+1]])

        self.data_root = data_root

        self.imgs = imgs
        self.transform_E = transform_E
        self.loader = loader
        self.eye_loader = eye_loader

    def resample(self):
        imgs = []
        for same_id in self.imgs_ids:
            random.shuffle(same_id)
            for i in range(len(same_id)-1):
                imgs.append([same_id[i],same_id[i+1]])

        self.imgs = imgs

    def __getitem__(self, index):
        path_source_1_r,path_source_1_l = self.imgs[index][0]
        path_source_2_r,path_source_2_l = self.imgs[index][1]

        source_frame_1_r = torch.FloatTensor(1,32,64)
        source_frame_1_r = self.transform_E(self.eye_loader(path_source_1_r))
        source_frame_1_l = torch.FloatTensor(1,32,64)
        source_frame_1_l = self.transform_E(self.eye_loader(path_source_1_l))
        source_frame_2_r = torch.FloatTensor(1,32,64)
        source_frame_2_r = self.transform_E(self.eye_loader(path_source_2_r))
        source_frame_2_l = torch.FloatTensor(1,32,64)
        source_frame_2_l = self.transform_E(self.eye_loader(path_source_2_l))

        sample = {'img_1_r':source_frame_1_r,'img_1_l':source_frame_1_l,
        'img_2_r':source_frame_2_r,'img_2_l':source_frame_2_l,}

        return sample

    def __len__(self):
        return len(self.imgs)
