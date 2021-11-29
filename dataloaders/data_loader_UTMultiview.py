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

def make_dataset(data_root,group_ids,session):
    images = []
    images_same_id = []
    if session == 'synth':
        cam_num = 144
    else:
        cam_num = 8

    for p in group_ids:
        #get file name
        data_path = os.path.join(data_root,'s'+str(p).zfill(2),session)

        #get data
        for i in range(160):
            data_folder_l = os.path.join(data_path,str(i).zfill(3)+'_left')
            label_file_l = list(csv.reader(open(data_folder_l+'.csv','r')))
            data_folder_r = os.path.join(data_path,str(i).zfill(3)+'_right')
            label_file_r = list(csv.reader(open(data_folder_r+'.csv','r')))
            for j in range(cam_num):
                img_path_l = os.path.join(data_folder_l,str(j).zfill(8)+'.bmp')
                gaze_l = [float(d) for d in label_file_l[j][:3]]
                head_l = [float(d) for d in label_file_l[j][3:9]]
                img_path_r = os.path.join(data_folder_r,str(j).zfill(8)+'.bmp')
                gaze_r = [float(d) for d in label_file_r[j][:3]]
                head_r = [float(d) for d in label_file_r[j][3:9]]
                images_same_id.append([img_path_r,img_path_l,gaze_r,gaze_l,head_r,head_l,p])
        
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
    def __init__(self, data_root,group_ids,session='test',
                transform_E=None, loader=default_loader, small=-1,
                single=False,rdnseed=1):

        imgs_ids = make_dataset(data_root,group_ids,session)
        self.imgs_ids = imgs_ids

        random.seed(rdnseed)
        imgs = []
        if single:
            for same_id in imgs_ids:
                random.shuffle(same_id)
                for i in range(int(len(same_id)/2)):
                    imgs.append([same_id[2*i],same_id[2*i+1]])
        else:
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
            for i in range(-1,len(same_id)-1):
                imgs.append([same_id[i],same_id[i+1]])

        self.imgs = imgs

    def __getitem__(self, index):
        path_source_1_r,path_source_1_l,gaze_1_r,gaze_1_l,head_1_r,head_1_l,person = self.imgs[index][0]
        path_source_2_r,path_source_2_l,gaze_2_r,gaze_2_l,head_2_r,head_2_l,person = self.imgs[index][1]

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
        
        source_frame_1_r = torch.FloatTensor(1,32,64)
        source_frame_1_r = self.transform_E(self.eye_loader(path_source_1_r))
        source_frame_1_l = torch.FloatTensor(1,32,64)
        source_frame_1_l = self.transform_E(self.eye_loader(path_source_1_l))
        source_frame_2_r = torch.FloatTensor(1,32,64)
        source_frame_2_r = self.transform_E(self.eye_loader(path_source_2_r))
        source_frame_2_l = torch.FloatTensor(1,32,64)
        source_frame_2_l = self.transform_E(self.eye_loader(path_source_2_l))

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
