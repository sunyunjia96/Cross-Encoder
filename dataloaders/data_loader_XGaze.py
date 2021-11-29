import torch.utils.data as data
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
from numpy import *
from pylab import *
import re
import glob
import random
import cv2
import torch.nn as nn
import math
import random
import scipy.io as sio
import h5py
import json

def make_dataset(ldmk_root,data_root,group_ids,cam=-1):
    #80 in total

    images = []
    images_same_id = []
    for p in group_ids:
        #get h5 name
        data_file = os.path.join(data_root,'subject'+str(p).zfill(4)+'.h5')
        ldmk_file = os.path.join(ldmk_root,'subject'+str(p).zfill(4)+'.h5')

        dataf = h5py.File(data_file,'r')
        ldmkf = h5py.File(ldmk_file,'r')
        #get images amount
        ndata = dataf['face_patch'].shape[0]

        #get file path
        for i in range(ndata):
            if cam>0:
                if dataf['cam_index'][i][0] != cam:
                    continue
            gaze = dataf['face_gaze'][i]
            head = dataf['face_head_pose'][i]
            points = ldmkf['landmarks'][i]
            images_same_id.append([data_file,i,gaze,head,points])

        dataf.close()
        ldmkf.close()
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
    imhist,bins = histogram(im.flatten(),nbr_bins,density= True)
    cdf = imhist.cumsum()
    cdf = 255.0 * cdf / cdf[-1]
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf

def eye_loader(im):
    try:
        if im.shape[0]<=0 or im.shape[1]<=0:
            return Image.new("L", (32, 64), "white")
        im = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        im = array(im.convert('L'))
        im2,cdf = histeq(im)
        im2 = Image.fromarray(uint8(im2))
        return im2
    except OSError:
        return Image.new("L", (32, 64), "white")

def get_rect(points):
    x = []
    y = []
    for p in points:
        x.append(p[0])
        y.append(p[1])

    x_expand = 0.1*(max(x)-min(x))
    y_expand = 0.1*(max(y)-min(y))

    x_max, x_min = max(x)+x_expand, min(x)-x_expand
    y_max, y_min = max(y)+y_expand, min(y)-y_expand

    #h:w=1:2
    if (y_max-y_min)*2 < (x_max-x_min):
        h = (x_max-x_min)/2
        pad = (h-(y_max-y_min))/2
        y_max += pad
        y_min -= pad
    else:
        h = (y_max-y_min)
        pad = (h*2-(x_max-x_min))/2
        x_max += pad
        x_min -= pad

    return int(x_max),int(x_min),int(y_max),int(y_min)

class ImagerLoader(data.Dataset):
    def __init__(self,ldmk_root, data_root,group_ids,
                transform_E=None, loader=default_loader,
                small=-1,cam=-1,everyone=False,):

        imgs_ids = make_dataset(ldmk_root,data_root,group_ids,cam)
        self.imgs_ids = imgs_ids

        self.data_root = data_root

        if everyone and small>0:
            #number of people can be sampled
            p_num = min(len(imgs_ids), small)
            left = small
            imgs = []
            while p_num > 0:
                random.seed(1)
                p_id = random.sample([i for i in range(len(imgs_ids))],p_num)
                for same_id in p_id:
                    imgs.append(random.sample(imgs_ids[same_id],2))
                    random.seed()
                left = left-p_num
                p_num = min(left, len(imgs_ids))
        else:
            imgs = []
            for same_id in imgs_ids:
                random.seed(1)
                random.shuffle(same_id)
                for i in range(-1,len(same_id)-1):
                    imgs.append([same_id[i],same_id[i+1]])

            if small > 0:
                imgs = random.sample(imgs,small)
                random.seed()

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
        data_file,data_id,gaze,head,points = self.imgs[index][0]
        dataf = h5py.File(data_file,'r')
        face_image = dataf['face_patch'][data_id]
        dataf.close()

        left_x_max, left_x_min, left_y_max, left_y_min = get_rect(points[36:42])
        right_x_max, right_x_min, right_y_max, right_y_min = get_rect(points[42:48])
        eye_l = face_image[left_y_min:left_y_max,left_x_min:left_x_max]
        eye_r = face_image[right_y_min:right_y_max,right_x_min:right_x_max]

        eye_l = self.eye_loader(eye_l)
        eye_r = self.eye_loader(eye_r)

        source_frame_1_l = torch.FloatTensor(1,32,64)
        source_frame_1_l = self.transform_E(eye_l)
        source_frame_1_r = torch.FloatTensor(1,32,64)
        source_frame_1_r = self.transform_E(eye_r)

        gaze_float_1 = torch.Tensor(gaze)
        gaze_float_1 = torch.FloatTensor(gaze_float_1)
        head_float_1 = torch.Tensor(head)
        head_float_1 = torch.FloatTensor(head_float_1)

        data_file,data_id,gaze,head,points = self.imgs[index][1]
        dataf = h5py.File(data_file,'r')
        face_image = dataf['face_patch'][data_id]
        dataf.close()

        left_x_max, left_x_min, left_y_max, left_y_min = get_rect(points[36:42])
        right_x_max, right_x_min, right_y_max, right_y_min = get_rect(points[42:48])
        eye_l = face_image[left_y_min:left_y_max,left_x_min:left_x_max]
        eye_r = face_image[right_y_min:right_y_max,right_x_min:right_x_max]

        eye_l = self.eye_loader(eye_l)
        eye_r = self.eye_loader(eye_r)

        source_frame_2_l = torch.FloatTensor(1,32,64)
        source_frame_2_l = self.transform_E(eye_l)
        source_frame_2_r = torch.FloatTensor(1,32,64)
        source_frame_2_r = self.transform_E(eye_r)

        gaze_float_2 = torch.Tensor(gaze)
        gaze_float_2 = torch.FloatTensor(gaze_float_2)
        head_float_2 = torch.Tensor(head)
        head_float_2 = torch.FloatTensor(head_float_2)

        sample = {'img_1_r':source_frame_1_r,'img_1_l':source_frame_1_l,
        'img_2_r':source_frame_2_r,'img_2_l':source_frame_2_l,
        'gaze_1':gaze_float_1, 'gaze_2':gaze_float_2,
        'head_1':head_float_1, 'head_2':head_float_2,
        }

        return sample

    def __len__(self):
        return len(self.imgs)
