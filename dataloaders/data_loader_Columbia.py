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

def make_dataset(data_root,group_ids):
    images = []
    images_same_id = []
    for p in group_ids:

        #get file name
        data_path = os.path.join(data_root,str(p).zfill(4),'file.txt')

        #get data
        with open(data_path,'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                img_path = os.path.join(data_root,str(p).zfill(4),line)
                img_path_r = img_path[:-1]+'_r.bmp'
                img_path_l = img_path[:-1]+'_l.bmp'
                label = line.split('_')
                head = float(label[-3][:-1])
                gaze = [float(label[-2][:-1]),float(label[-1][:-2])]
                images_same_id.append([img_path_r,img_path_l,gaze,head,p])
        
        #random pair
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

def spherical2cartesial(x):
    #angle to radian
    x = x*math.pi/180

    output = torch.zeros(3)
    output[2] = torch.cos(x[0])*torch.cos(x[1])
    output[0] = torch.cos(x[0])*torch.sin(x[1])
    output[1] = torch.sin(x[0])

    return output

def H2vector(x):
    x = x*math.pi/180

    rot_mat = np.zeros((3,3))
    rot_mat[0] = [1,0,0]
    rot_mat[1] = [0,math.cos(x),-math.sin(x)]
    rot_mat[2] = [0,math.sin(x),math.cos(x)]

    return cv2.Rodrigues(rot_mat)[0]

class ImagerLoader(data.Dataset):
    def __init__(self, data_root,group_ids,
                transform_E=None, loader=default_loader, small=-1,
                single=False,rdnseed=1):

        imgs_ids = make_dataset(data_root,group_ids)
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
            imgs = random.sample(imgs,small)
        
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
        path_source_1_r,path_source_1_l,gaze_1,head_1,person = self.imgs[index][0]
        path_source_2_r,path_source_2_l,gaze_2,head_2,person = self.imgs[index][1]

        gaze_float_1 = torch.Tensor(gaze_1)
        gaze_float_1 = torch.FloatTensor(gaze_float_1)
        normalized_gaze_1 = gaze_float_1
        #normalized_gaze_1 = spherical2cartesial(gaze_float_1)
        gaze_float_2 = torch.Tensor(gaze_2)
        gaze_float_2 = torch.FloatTensor(gaze_float_2)
        normalized_gaze_2 = gaze_float_2
        #normalized_gaze_2 = spherical2cartesial(gaze_float_1)

        head_float_1 = torch.Tensor([head_1])
        head_float_1 = torch.FloatTensor(head_float_1)
        head_float_2 = torch.Tensor([head_2])
        head_float_2 = torch.FloatTensor(head_float_2)

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
        'gaze_1':normalized_gaze_1, 'gaze_2':normalized_gaze_2,
        'head_1':head_float_1, 'head_2':head_float_2,
        'id':person,}

        return sample

    def __len__(self):
        return len(self.imgs)
