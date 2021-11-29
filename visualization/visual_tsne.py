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
from data_loader_UTMultiview import ImagerLoader

def histeq(im,nbr_bins = 256):
    imhist,bins = np.histogram(im.flatten(),nbr_bins,density= True)
    cdf = imhist.cumsum()
    cdf = 255.0 * cdf / cdf[-1]
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf

growth_rate=32
z_dim_app=32
z_dim_gaze=3
decoder_input_c=32

network = DTED(
    growth_rate=growth_rate,
    z_dim_app=z_dim_app,
    z_dim_gaze=z_dim_gaze,
    decoder_input_c=decoder_input_c,
)
pretrain_dict = torch.load('saved_models/u0.pth.tar')
network.load_state_dict(pretrain_dict)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def send_data_dict_to_gpu(data):
    for k in data:
        v = data[k]
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    return data

def eye_loader(path):
    try:
        im = np.array(Image.open(path).convert('L'))
        im2,cdf = histeq(im)
        im2 = Image.fromarray(np.uint8(im2))
        return im2
    except OSError:
        print(path)
        return Image.new("RGB", (512, 512), "white")

network = network.to(device)
network.eval()

data_root = '../../datas/UTMultiview'
batch_size=64
small_trainset = ImagerLoader(data_root,[i for i in range(34,50)]+[i for i in range(50,50)],'test',transforms.Compose([
                    transforms.Resize((32,64)),transforms.ToTensor(),#image_normalize,
                    ]),transforms.Compose([
                    transforms.Resize((224,224)),transforms.ToTensor(),#image_normalize,
                    ]),single=True)
small_train = torch.utils.data.DataLoader(
        small_trainset,
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

greps = []
ereps = []
labels = []
ids = []
for input_dict in small_train:
    input_dict = send_data_dict_to_gpu(input_dict)
    output_dict = network(input_dict)

    greps += output_dict['z_gaze_enc_1_l'].view(-1,3*z_dim_gaze).tolist()
    greps += output_dict['z_gaze_enc_1_r'].view(-1,3*z_dim_gaze).tolist()
    greps += output_dict['z_gaze_enc_2_l'].view(-1,3*z_dim_gaze).tolist()
    greps += output_dict['z_gaze_enc_2_r'].view(-1,3*z_dim_gaze).tolist()

    ereps += output_dict['z_app_1_l'].tolist()
    ereps += output_dict['z_app_1_r'].tolist()
    ereps += output_dict['z_app_2_l'].tolist()
    ereps += output_dict['z_app_2_r'].tolist()

    labels += input_dict['gaze_1_l'].tolist()
    labels += input_dict['gaze_1_r'].tolist()
    labels += input_dict['gaze_2_l'].tolist()
    labels += input_dict['gaze_2_r'].tolist()

    ids += (2*input_dict['id']).tolist()
    ids += (2*input_dict['id']+1).tolist()
    ids += (2*input_dict['id']).tolist()
    ids += (2*input_dict['id']+1).tolist()

import numpy as np
from sklearn.manifold import TSNE
import random

def spherical2cartesial(x):
    #angle to radian
    x = x*math.pi/180

    output = np.zeros((x.shape[0],3))
    output[:,2] = -np.cos(x[:,0])*np.cos(x[:,1])
    output[:,0] = np.cos(x[:,0])*np.sin(x[:,1])
    output[:,1] = np.sin(x[:,0])

    return output

def set_colors_vec(x):
    #x = spherical2cartesial(x)
    x = (x+1)/2
    return x 

def set_colors_number(x):
    color = [[(int(i/8)%4)/3,(int(i/2)%4)/3,(i%2)/2] for i in range(16*2)]
    return np.array([color[i] for i in x])
        
def rearrange_id(x,a,b,tr):
    if tr:
        for i in range(len(x)):
            if int(x[i]/2)>=b:
                x[i] = x[i]-(b-a)*2
    else:
        for i in range(len(x)):
            x[i] = x[i]-2*a
    return x

labels = np.array(labels)
greps = np.array(greps)
ereps = np.array(ereps)
ids = rearrange_id(ids,34,50,False)
ids = np.array(ids)

labels = set_colors_vec(labels)
ids = set_colors_number(ids)

tsne = TSNE()
grep2 = tsne.fit_transform(greps)
erep2 = tsne.fit_transform(ereps)

plt.scatter(grep2[:,0],grep2[:,1],c=labels,s=10)
plt.savefig(os.path.join('visualization','U','grep_g_t.jpg'))
plt.clf()
plt.scatter(grep2[:,0],grep2[:,1],c=ids,s=10)
plt.savefig(os.path.join('visualization','U','grep_p_t.jpg'))
plt.clf()
plt.scatter(erep2[:,0],erep2[:,1],c=labels,s=10)
plt.savefig(os.path.join('visualization','U','erep_g_t.jpg'))
plt.scatter(erep2[:,0],erep2[:,1],c=ids,s=10)
plt.savefig(os.path.join('visualization','U','erep_p_t.jpg'))
