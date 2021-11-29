import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import cv2
import gc
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
import numpy as np
from models import DTED
from models.regressor import regressor
from PIL import Image
import matplotlib.pyplot as plt
import os
from data_loader_MPIIGaze import ImagerLoader
import scipy.io as sio

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

data_root = '../../datas/MPIIGaze/MPIIGaze/Data/Normalized/'
data_path = os.path.join(data_root,'p'+str(0).zfill(2))
matlist = os.listdir(data_path)
for mat in matlist:
    mat_path = os.path.join(data_path,mat)
    if not os.path.exists(os.path.join('temp_pick',mat[:-4])):
        os.makedirs(os.path.join('temp_pick',mat[:-4]))

    data = sio.loadmat(mat_path)
    data_num = len(data['filenames'])
    for i in range(data_num):
        image_r = data['data']['right'][0,0]['image'][0,0][i]
        image_l = data['data']['left'][0,0]['image'][0,0][i]
        print(image_r.shape)
        cv2.imwrite(os.path.join('temp_pick',mat[:-4],str(i)+'_r_org.jpg'),image_r)
        cv2.imwrite(os.path.join('temp_pick',mat[:-4],str(i)+'_l_org.jpg'),image_l)

        #image_r,_ = histeq(image_r)
        #image_l,_ = histeq(image_l)
        #cv2.imwrite(os.path.join('temp_pick',mat[:-4],str(i)+'_r_hst.jpg'),image_r)
        #cv2.imwrite(os.path.join('temp_pick',mat[:-4],str(i)+'_l_hst.jpg'),image_l)
        break
    break
