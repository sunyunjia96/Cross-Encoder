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

def make_dataset(data_root,group_ids,test=False):
    images = []
    images_same_id = []

    for p in group_ids:
        #get file name
        data_path = os.path.join(data_root,'p'+str(p).zfill(2))

        #get data
        if test:
            eval_file = os.path.join(data_root,'..','..','Evaluation Subset','sample list for eye image','p'+str(p).zfill(2)+'.txt')
            with open(eval_file,'r') as eval_set:
                while True:
                    line = eval_set.readline()
                    if not line:
                        break
                    eval_item = line.split()
                    selected = eval_item[0].split('/')
                    day = selected[0]
                    pic = int(selected[1][:-4])-1
                    LorR = eval_item[1]

                    data = sio.loadmat(os.path.join(data_path,day+'.mat'))
                    gaze_l = data['data']['left'][0,0]['gaze'][0,0][pic]
                    head_l = data['data']['left'][0,0]['pose'][0,0][pic]
                    gaze_r = data['data']['right'][0,0]['gaze'][0,0][pic]
                    head_r = data['data']['right'][0,0]['pose'][0,0][pic]
                    images_same_id.append([os.path.join(data_path,day+'.mat'),pic,gaze_r,gaze_l,head_r,head_l,p])

        else:
            matlist = os.listdir(data_path)
            for mat in matlist:
                data = sio.loadmat(os.path.join(data_path,mat))
                data_num = len(data['filenames'])
                for i in range(data_num):
                    gaze_l = data['data']['left'][0,0]['gaze'][0,0][i]
                    head_l = data['data']['left'][0,0]['pose'][0,0][i]
                    gaze_r = data['data']['right'][0,0]['gaze'][0,0][i]
                    head_r = data['data']['right'][0,0]['pose'][0,0][i]
                    images_same_id.append([os.path.join(data_path,mat),i,gaze_r,gaze_l,head_r,head_l,p])

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

def eye_loader(im):
    try:
        im2,cdf = histeq(im)
        im2 = Image.fromarray(np.uint8(im2))
        return im2
    except OSError:
        print(path)
        return Image.new("RGB", (512, 512), "white")

class ImagerLoader(data.Dataset):
    def __init__(self, data_root,group_ids,test=False,
                transform_E=None, loader=default_loader,
                small=-1,rdnseed=1):

        imgs_ids = make_dataset(data_root,group_ids,test)
        self.imgs_ids = imgs_ids

        random.seed(rdnseed)

        imgs = []
        for same_id in imgs_ids:
            random.shuffle(same_id)
            for i in range(-1,len(same_id)-1):
                imgs.append([same_id[i],same_id[i+1]])

        if small > 0:
            imgs = random.sample(imgs,int(small/2))
        random.seed()

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
        path_source_1,index_1,gaze_1_r,gaze_1_l,head_1_r,head_1_l,person = self.imgs[index][0]
        path_source_2,index_2,gaze_2_r,gaze_2_l,head_2_r,head_2_l,person = self.imgs[index][1]

        gaze_float_1_l = torch.Tensor(gaze_1_l)
        gaze_float_1_l = torch.FloatTensor(gaze_float_1_l)
        gaze_float_1_r = torch.Tensor(gaze_1_r)
        gaze_float_1_r = torch.FloatTensor(gaze_float_1_r)
        gaze_float_2_l = torch.Tensor(gaze_2_l)
        gaze_float_2_l = torch.FloatTensor(gaze_float_2_l)
        gaze_float_2_r = torch.Tensor(gaze_2_r)
        gaze_float_2_r = torch.FloatTensor(gaze_float_2_r)

        head_float_1_l = torch.Tensor(head_1_l)
        head_float_1_l = torch.FloatTensor(head_float_1_l)
        head_float_1_r = torch.Tensor(head_1_r)
        head_float_1_r = torch.FloatTensor(head_float_1_r)
        head_float_2_l = torch.Tensor(head_2_l)
        head_float_2_l = torch.FloatTensor(head_float_2_l)
        head_float_2_r = torch.Tensor(head_2_r)
        head_float_2_r = torch.FloatTensor(head_float_2_r)
       
        mat_1 = sio.loadmat(path_source_1)
        image_1_r = mat_1['data']['right'][0,0]['image'][0,0][index_1]
        image_1_l = mat_1['data']['left'][0,0]['image'][0,0][index_1]
        source_frame_1_r = torch.FloatTensor(1,32,64)        
        source_frame_1_r = self.transform_E(self.eye_loader(image_1_r))
        source_frame_1_l = torch.FloatTensor(1,32,64)
        source_frame_1_l = self.transform_E(self.eye_loader(image_1_l))

        mat_2 = sio.loadmat(path_source_2)
        image_2_r = mat_2['data']['right'][0,0]['image'][0,0][index_2]
        image_2_l = mat_2['data']['left'][0,0]['image'][0,0][index_2]
        source_frame_2_r = torch.FloatTensor(1,32,64)
        source_frame_2_r = self.transform_E(self.eye_loader(image_2_r))
        source_frame_2_l = torch.FloatTensor(1,32,64)
        source_frame_2_l = self.transform_E(self.eye_loader(image_2_l))

        sample = {'img_1_r':source_frame_1_r,'img_1_l':source_frame_1_l,
        'img_2_r':source_frame_2_r,'img_2_l':source_frame_2_l,
        'gaze_1_r':gaze_float_1_r,'gaze_1_l':gaze_float_1_l,
        'gaze_2_r':gaze_float_2_r,'gaze_2_l':gaze_float_2_l,
        'head_1_r':head_float_1_r,'head_1_l':head_float_1_l,
        'head_2_r':head_float_2_r,'head_2_l':head_float_2_l,
        'id':person,}

        return sample

    def __len__(self):
        return len(self.imgs)
