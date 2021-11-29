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
small_trainset = ImagerLoader(data_root,[i for i in range(34,50)]+[i for i in range(50,50)],'test',
                    transforms.Compose([
                    transforms.Resize((32,64)),transforms.ToTensor(),#image_normalize,
                    ]),transforms.Compose([
                    transforms.Resize((224,224)),transforms.ToTensor(),#image_normalize,
                    ]),single=True)
small_train = torch.utils.data.DataLoader(
        small_trainset,
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

gsim_g = 0
esim_e = 0
gsim_e = 0
esim_g = 0
asim_g = 0
asim_e = 0
for input_dict in small_train:
 
    input_dict = send_data_dict_to_gpu(input_dict)
    output_dict = network(input_dict)

    grep_1_l = output_dict['z_gaze_enc_1_l'].view(-1,3*z_dim_gaze)
    grep_1_r = output_dict['z_gaze_enc_1_r'].view(-1,3*z_dim_gaze)
    grep_2_l = output_dict['z_gaze_enc_2_l'].view(-1,3*z_dim_gaze)
    grep_2_r = output_dict['z_gaze_enc_2_r'].view(-1,3*z_dim_gaze)

    erep_1_l = output_dict['z_app_1_l']
    erep_1_r = output_dict['z_app_1_r']
    erep_2_l = output_dict['z_app_2_l']
    erep_2_r = output_dict['z_app_2_r']

    arep_1_l = torch.cat((grep_1_l,erep_1_l),dim=1)
    arep_1_r = torch.cat((grep_1_r,erep_1_r),dim=1)
    arep_2_l = torch.cat((grep_2_l,erep_2_l),dim=1)
    arep_2_r = torch.cat((grep_2_r,erep_2_r),dim=1)

    gsim_g += nn.CosineSimilarity()(grep_1_l,grep_1_r).sum().item()+nn.CosineSimilarity()(grep_2_l,grep_2_r).sum().item()
    esim_g += nn.CosineSimilarity()(erep_1_l,erep_1_r).sum().item()+nn.CosineSimilarity()(erep_2_l,erep_2_r).sum().item()
    esim_e += nn.CosineSimilarity()(erep_1_l,erep_2_l).sum().item()+nn.CosineSimilarity()(erep_1_r,erep_2_r).sum().item()
    gsim_e += nn.CosineSimilarity()(grep_1_l,grep_2_l).sum().item()+nn.CosineSimilarity()(grep_1_r,grep_2_r).sum().item()

    asim_g += nn.CosineSimilarity()(arep_1_l,arep_1_r).sum().item()+nn.CosineSimilarity()(arep_2_l,arep_2_r).sum().item()
    asim_e += nn.CosineSimilarity()(arep_1_l,arep_2_l).sum().item()+nn.CosineSimilarity()(arep_1_r,arep_2_r).sum().item()

#print('gsim_g: {0:.4f}, esim_g: {1:.4f}, esim_e: {2:.4f}, gsim_e: {3:.4f}'.format(gsim_g/len(small_trainset)/2,
#esim_g/len(small_trainset)/2,esim_e/len(small_trainset)/2,gsim_e/len(small_trainset)/2))

print('sim_g: {0:.4f}, sim_e: {1:.4f}'.format(asim_g/len(small_trainset)/2,asim_e/len(small_trainset)/2))

