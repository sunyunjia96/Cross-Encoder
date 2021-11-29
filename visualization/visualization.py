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

def recover_in_images(x):
    # Every specified iterations save sample images
    # Note: We're doing this separate to Tensorboard to control which input
    #       samples we visualize, and also because Tensorboard is an inefficient
    #       way to store such images.
    x = x.cpu().detach().numpy()
    x = x * 255.0
    x = np.clip(x, 0, 255)  # Avoid artifacts due to slight under/overflow
    x = np.transpose(x, [0, 2, 3, 1])  # CHW to HWC
    x = x.astype(np.uint8)
    x = x[:, :, :, ::-1]  # RGB to BGR for OpenCV
    return x

def histeq(im,nbr_bins = 256):
    imhist,bins = np.histogram(im.flatten(),nbr_bins,density= True)
    cdf = imhist.cumsum()
    cdf = 255.0 * cdf / cdf[-1]
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf

growth_rate=32
z_dim_app=32
z_dim_gaze=4
decoder_input_c=32

network = DTED(
    growth_rate=growth_rate,
    z_dim_app=z_dim_app,
    z_dim_gaze=z_dim_gaze,
    decoder_input_c=decoder_input_c,
)
pretrain_dict = torch.load('saved_models/12/c1.pth.tar')
network.load_state_dict(pretrain_dict)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def send_data_dict_to_gpu(data):
    for k in data:
        v = data[k]
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True).expand(1,v.shape[0],v.shape[1],v.shape[2])
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

dataset = 'Columbia_test'

for img_index in [1,3,5,7]:

    pair = (str(img_index),str(img_index+1))

    img_name_1_r, img_name_1_l = pair[0]+'_r.bmp',pair[0]+'_l.bmp'
    img_name_2_r, img_name_2_l = pair[1]+'_r.bmp',pair[1]+'_l.bmp'

    path_source_1_r,path_source_1_l = os.path.join('egfig',dataset,img_name_1_r),os.path.join('egfig',dataset,img_name_1_l)
    path_source_2_r,path_source_2_l = os.path.join('egfig',dataset,img_name_2_r),os.path.join('egfig',dataset,img_name_2_l)

    transform_E = transforms.Compose([transforms.Resize((32,64)),transforms.ToTensor()])

    source_frame_1_r = torch.FloatTensor(1,32,64)
    source_frame_1_r = transform_E(eye_loader(path_source_1_r))
    source_frame_1_l = torch.FloatTensor(1,32,64)
    source_frame_1_l = transform_E(eye_loader(path_source_1_l))
    source_frame_2_r = torch.FloatTensor(1,32,64)
    source_frame_2_r = transform_E(eye_loader(path_source_2_r))
    source_frame_2_l = torch.FloatTensor(1,32,64)
    source_frame_2_l = transform_E(eye_loader(path_source_2_l))

    input_dict = {'img_1_r':source_frame_1_r,'img_1_l':source_frame_1_l,
    'img_2_r':source_frame_2_r,'img_2_l':source_frame_2_l,}

    input_dict = send_data_dict_to_gpu(input_dict)
    output_dict = network(input_dict)

    img_1_l = output_dict['image_hat_1_l']
    img_1_r = output_dict['image_hat_1_r']
    img_2_l = output_dict['image_hat_2_l']
    img_2_r = output_dict['image_hat_2_r']

    img_1_l = recover_in_images(img_1_l)
    img_1_r = recover_in_images(img_1_r)
    img_2_l = recover_in_images(img_2_l)
    img_2_r = recover_in_images(img_2_r)

    cv2.imwrite(os.path.join('visualization','recon','C_t',img_name_1_l[:-4]+'.jpg'),img_1_l[0])
    cv2.imwrite(os.path.join('visualization','recon','C_t',img_name_1_r[:-4]+'.jpg'),img_1_r[0])
    cv2.imwrite(os.path.join('visualization','recon','C_t',img_name_2_l[:-4]+'.jpg'),img_2_l[0])
    cv2.imwrite(os.path.join('visualization','recon','C_t',img_name_2_r[:-4]+'.jpg'),img_2_r[0])

    '''
    r_1_l = output_dict['z_gaze_enc_1_l'].view(-1,3*z_dim_gaze).tolist()[0]
    r_1_r = output_dict['z_gaze_enc_1_r'].view(-1,3*z_dim_gaze).tolist()[0]
    r_2_l = output_dict['z_gaze_enc_2_l'].view(-1,3*z_dim_gaze).tolist()[0]
    r_2_r = output_dict['z_gaze_enc_2_r'].view(-1,3*z_dim_gaze).tolist()[0]
    z_1_l = output_dict['z_app_1_l'].tolist()[0]
    z_1_r = output_dict['z_app_1_r'].tolist()[0]
    z_2_l = output_dict['z_app_2_l'].tolist()[0]
    z_2_r = output_dict['z_app_2_r'].tolist()[0]


    #f_1_l = torch.cat((r_1_l,z_1_l),dim=1).tolist()[0]
    #f_1_r = torch.cat((r_1_r,z_1_r),dim=1).tolist()[0]
    #f_2_l = torch.cat((r_2_l,z_2_l),dim=1).tolist()[0]
    #f_2_r = torch.cat((r_2_r,z_2_r),dim=1).tolist()[0]

    dim_index_r = [i for i in range(len(r_1_l))]
    dim_index_z = [i for i in range(len(z_1_l))]

    def draw_plot(vec,name):
        fig,ax=plt.subplots()
        ax.bar([i+1 for i in range(len(vec))],vec,width=0.5)
        ax.set_xlabel('dimension index')
        ax.set_ylabel('value')
        ax.set_title(name[0])
        plt.xticks([i+1 for i in range(len(vec))],[i+1 for i in range(len(vec))])
        plt.savefig(os.path.join('visualization','equal',dataset,name))
        plt.close()

    draw_plot(r_1_l,'r_'+img_name_1_l[:-4]+'.jpg')
    draw_plot(r_1_r,'r_'+img_name_1_r[:-4]+'.jpg')
    draw_plot(r_2_l,'r_'+img_name_2_l[:-4]+'.jpg')
    draw_plot(r_2_r,'r_'+img_name_2_r[:-4]+'.jpg')
    draw_plot(z_1_l,'z_'+img_name_1_l[:-4]+'.jpg')
    draw_plot(z_1_r,'z_'+img_name_1_r[:-4]+'.jpg')
    draw_plot(z_2_l,'z_'+img_name_2_l[:-4]+'.jpg')
    draw_plot(z_2_r,'z_'+img_name_2_r[:-4]+'.jpg')
    '''
