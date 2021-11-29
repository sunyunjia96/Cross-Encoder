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
import h5py

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



def make_dataset_F(data_root):
    images = []
    images_same_id = []

    persons = os.listdir(data_root)
    for p in persons:
        #get file name
        get_filelist(os.path.join(data_root,p),images_same_id)

        images.append(images_same_id)
        images_same_id = []
    
    return images

def make_dataset_T(data_root):
    images = []
    images_same_id = []

    for p in range(1,52):
        for s1 in range(1,6):
            for s2 in range(1,5):
                #get file name
                data_path = os.path.join(data_root,str(p),str(p)+'_'+str(s1)+'_'+str(s2))
                if not os.path.exists(data_path):
                    continue
                files = os.listdir(data_path)
                num_frame = int(len(files)/2)
                #get data
                for i in range(num_frame):
                    img_path_l = os.path.join(data_path,str(i*15)+'_l.bmp')
                    img_path_r = os.path.join(data_path,str(i*15)+'_r.bmp')
                    if os.path.exists(img_path_l) and os.path.exists(img_path_r):
                        images_same_id.append([img_path_r,img_path_l])

        images.append(images_same_id)
        images_same_id = []

    return images

def make_dataset_X(ldmk_root,data_root,group_ids=[i for i in range(80)]):
    #80 in total
    subject_filenames = os.listdir(data_root)

    images = []
    images_same_id = []
    for p in group_ids:
        #get h5 name
        data_file = os.path.join(data_root,subject_filenames[p])
        ldmk_file = os.path.join(ldmk_root,subject_filenames[p])

        ldmkf = h5py.File(ldmk_file,'r')
        #get images amount
        ndata = ldmkf['landmarks'].shape[0]

        #get file path
        for i in range(ndata):
            points = ldmkf['landmarks'][i]
            images_same_id.append([data_file,i,points])
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
    imhist,bins = np.histogram(im.flatten(),nbr_bins,density= True)
    cdf = imhist.cumsum()
    cdf = 255.0 * cdf / cdf[-1]
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf

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

def eye_loader(path):
    try:
        im = np.array(Image.open(path).convert('L'))
        im2,cdf = histeq(im)
        im2 = Image.fromarray(np.uint8(im2))
        return im2
    except OSError:
        print(path)
        return Image.new("RGB", (512, 512), "white")

def eye_loader_X(im):
    try:
        if im.shape[0]<=0 or im.shape[1]<=0:
            return Image.new("L", (32, 64), "white")
        im = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        im = np.array(im.convert('L'))
        im2,cdf = histeq(im)
        im2 = Image.fromarray(np.uint8(im2))
        return im2
    except OSError:
        return Image.new("L", (32, 64), "white")

class ImagerLoader(data.Dataset):
    def __init__(self, path_F, path_T, path_X, ldmk_path_X, transform_E=None, loader=default_loader):

        imgs_ids = make_dataset_F(path_F)
        imgs_ids += make_dataset_T(path_T)
        imgs_ids += make_dataset_X(ldmk_path_X,path_X)
        self.imgs_ids = imgs_ids

        imgs = []
        for same_id in imgs_ids:
            random.shuffle(same_id)
            for i in range(len(same_id)-1):
                imgs.append([same_id[i],same_id[i+1]])

        self.imgs = imgs
        self.transform_E = transform_E
        self.loader = loader
        self.eye_loader = eye_loader
        self.eye_loader_X = eye_loader_X

    def resample(self):
        imgs = []
        for same_id in self.imgs_ids:
            random.shuffle(same_id)
            for i in range(len(same_id)-1):
                imgs.append([same_id[i],same_id[i+1]])

        self.imgs = imgs

    def __getitem__(self, index):
        if len(self.imgs[index][0])==2:
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
        
        else:
            data_file,data_id,points = self.imgs[index][0]
            dataf = h5py.File(data_file,'r')
            face_image = dataf['face_patch'][data_id]
            dataf.close()

            left_x_max, left_x_min, left_y_max, left_y_min = get_rect(points[36:42])
            right_x_max, right_x_min, right_y_max, right_y_min = get_rect(points[42:48])
            eye_l = face_image[left_y_min:left_y_max,left_x_min:left_x_max]
            eye_r = face_image[right_y_min:right_y_max,right_x_min:right_x_max]

            eye_l = self.eye_loader_X(eye_l)
            eye_r = self.eye_loader_X(eye_r)

            source_frame_1_l = torch.FloatTensor(1,32,64)
            source_frame_1_l = self.transform_E(eye_l)
            source_frame_1_r = torch.FloatTensor(1,32,64)
            source_frame_1_r = self.transform_E(eye_r)

            data_file,data_id,points = self.imgs[index][1]
            dataf = h5py.File(data_file,'r')
            face_image = dataf['face_patch'][data_id]
            dataf.close()

            left_x_max, left_x_min, left_y_max, left_y_min = get_rect(points[36:42])
            right_x_max, right_x_min, right_y_max, right_y_min = get_rect(points[42:48])
            eye_l = face_image[left_y_min:left_y_max,left_x_min:left_x_max]
            eye_r = face_image[right_y_min:right_y_max,right_x_min:right_x_max]

            eye_l = self.eye_loader_X(eye_l)
            eye_r = self.eye_loader_X(eye_r)

            source_frame_2_l = torch.FloatTensor(1,32,64)
            source_frame_2_l = self.transform_E(eye_l)
            source_frame_2_r = torch.FloatTensor(1,32,64)
            source_frame_2_r = self.transform_E(eye_r)

        sample = {'img_1_r':source_frame_1_r,'img_1_l':source_frame_1_l,
        'img_2_r':source_frame_2_r,'img_2_l':source_frame_2_l,}

        return sample

    def __len__(self):
        return len(self.imgs)
