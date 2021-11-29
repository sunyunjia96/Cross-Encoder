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
from data_loader_Columbia import ImagerLoader
from models import DTED
from models.regressor import regressor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE()

growth_rate=32
z_dim_app=32
z_dim_gaze=4
decoder_input_c=32

fold = [(46,57),(35,46),(24,35),(13,24),(1,13)]
mi = 0
small = -1

network = DTED(
    growth_rate=growth_rate,
    z_dim_app=z_dim_app,
    z_dim_gaze=z_dim_gaze,
    decoder_input_c=decoder_input_c,
)
#model_dict = network.state_dict()
pretrain_dict = torch.load('saved_models/c'+str(mi+1)+'.pth.tar')
#pretrain_dict = torch.load('saved_models/T.pth.tar')
#model_dict.update(pretrain_dict)
network.load_state_dict(pretrain_dict)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def send_data_dict_to_gpu(data):
    for k in data:
        v = data[k]
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    return data

network = network.to(device)
network.eval()

criterion = nn.L1Loss()
#criterion = nn.CosineSimilarity(dim=1)

#data loaders
data_root = '../data/eyes/Columbia'

batch_size=8

small_trainset = ImagerLoader(data_root,[i for i in range(1,fold[mi][0])]+[i for i in range(fold[mi][1],57)],transforms.Compose([
                    transforms.Resize((32,64)),transforms.ToTensor(),#image_normalize,
                    ]),transforms.Compose([
                    transforms.Resize((224,224)),transforms.ToTensor(),#image_normalize,
                    ]),small=small,single=True)
small_train = torch.utils.data.DataLoader(
        small_trainset,
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

small_valset = ImagerLoader(data_root,[i for i in range(fold[mi][0],fold[mi][1])],transforms.Compose([
                    transforms.Resize((32,64)),transforms.ToTensor(),#image_normalize,
                    ]),transforms.Compose([
                    transforms.Resize((224,224)),transforms.ToTensor(),#image_normalize,
                    ]),single=True)
small_val = torch.utils.data.DataLoader(
        small_valset,
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

r = []
z = []
with torch.no_grad():
    for input_dict in small_train:
        input_dict = send_data_dict_to_gpu(input_dict)
        output_dict = network(input_dict)

        r_1_l = output_dict['z_gaze_enc_1_l'].view(-1,3*z_dim_gaze).tolist()
        r.append(r_1_l)
        r_1_r = output_dict['z_gaze_enc_1_r'].view(-1,3*z_dim_gaze).tolist()
        r.append(r_1_r)
        z_1_l = output_dict['z_app_1_l'].tolist()
        z.append(z_1_l)
        z_1_r = output_dict['z_app_1_r'].tolist()
        z.append(z_1_r)
        r_2_l = output_dict['z_gaze_enc_2_l'].view(-1,3*z_dim_gaze).tolist()
        r.append(r_2_l)
        r_2_r = output_dict['z_gaze_enc_2_r'].view(-1,3*z_dim_gaze).tolist()
        r.append(r_2_r)
        z_2_l = output_dict['z_app_2_l'].tolist()
        z.append(z_2_l)
        z_2_r = output_dict['z_app_2_r'].tolist()
        z.append(z_2_r)
        break

r = np.array(r)
z = np.array(z)
r = tsne.fit_transform(r)
z = tsne.fit_transform(z)
plt.plot(r[0], r[1])
plt.savefig('vis.png')
