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
from dataloaders import Columbia, UTMultiview, MPIIGaze, XGaze, Unite
from models import CrossEncoder, regressor

growth_rate=32
z_dim_app=32
z_dim_gaze=4
decoder_input_c=32

mi = 0
small = 100

network = DTED(
    growth_rate=growth_rate,
    z_dim_app=z_dim_app,
    z_dim_gaze=z_dim_gaze,
    decoder_input_c=decoder_input_c,
)
model_dict = network.state_dict()
network.load_state_dict(torch.load('m'+str(mi)+'.pth.tar'))

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

def spherical2cartesial(x):
    #angle to radian for Columbia
    #x = x*math.pi/180

    output = torch.zeros(x.size(0),3).cuda()
    output[:,2] = torch.cos(x[:,0])*torch.cos(x[:,1])
    output[:,0] = torch.cos(x[:,0])*torch.sin(x[:,1])
    output[:,1] = torch.sin(x[:,0])

    return output

def cartesial2spherical(x):

    spherical_vector = torch.FloatTensor(x.size(0),2).cuda()
    spherical_vector[:,1] = torch.atan2(x[:,0],x[:,2])
    spherical_vector[:,0] = torch.asin(x[:,1])

    return spherical_vector

def compute_angular_error(input,target):
    input = spherical2cartesial(input)
    target = spherical2cartesial(target)

    input = input.view(-1,3,1)
    #input = F.normalize(input)
    target = target.view(-1,1,3)
    output_dot = torch.bmm(target,input)
    output_dot = output_dot.view(-1)
    output_dot = output_dot.clamp(-1.,1.)
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    output_dot = 180*torch.sum(output_dot)/math.pi
    return output_dot

#data loader
data_root = '../../datas/MPIIGaze/MPIIGaze/Data/Normalized/'

reg = regressor(3*z_dim_gaze)
reg = reg.to(device)
optimizer = optim.Adam(reg.parameters(),lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5, last_epoch=-1)

batch_size=4

small_trainset = MPIIGaze(data_root,[i for i in range(0,14-mi)]+[i for i in range(15-mi,15)],False,
                transforms.Compose([transforms.Resize((32,64)),transforms.ToTensor()]),small=small)
small_train = torch.utils.data.DataLoader(
        small_trainset,
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

small_valset = MPIIGaze(data_root,[i for i in range(14-mi,15-mi)],True,
                transforms.Compose([transforms.Resize((32,64)),transforms.ToTensor()]))
small_val = torch.utils.data.DataLoader(
        small_valset,
        batch_size=batch_size*16, shuffle=False,
        num_workers=4, pin_memory=True)

#train regressor
epochs = 30
best = 100
for epoch in range(epochs):

    reg.train()
    running_loss = 0.
    running_angle = 0.
    for i, input_dict in enumerate(small_train):
        input_dict = send_data_dict_to_gpu(input_dict)
        output_dict = network(input_dict,inference=True)

        grep_1_l = output_dict['z_gaze_enc_1_l'].view(-1,3*z_dim_gaze)
        grep_1_r = output_dict['z_gaze_enc_1_r'].view(-1,3*z_dim_gaze)

        grep = torch.cat((grep_1_l,grep_1_r),dim=0)

        head = torch.cat((input_dict['head_1_l'],input_dict['head_1_r']),dim=0)

        optimizer.zero_grad()
        gaze_hat = reg(grep,head)
        gaze = torch.cat((input_dict['gaze_1_l'],input_dict['gaze_1_r']),dim=0)
        loss = criterion(gaze_hat, gaze)
        angular_error = compute_angular_error(gaze_hat,gaze)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_angle += angular_error

    scheduler.step()

    network.eval()
    reg.eval()
    error = 0
    with torch.no_grad():
        for input_dict in small_val:
            input_dict = send_data_dict_to_gpu(input_dict)
            output_dict = network(input_dict,inference=True)

            grep_1_l = output_dict['z_gaze_enc_1_l'].view(-1,3*z_dim_gaze)
            grep_1_r = output_dict['z_gaze_enc_1_r'].view(-1,3*z_dim_gaze)

            grep = torch.cat((grep_1_l,grep_1_r),dim=1)

            head = torch.cat((input_dict['head_1_l'],input_dict['head_1_r']),dim=0)

            gaze_hat = reg(grep,head)
            gaze = torch.cat((input_dict['gaze_1_l'],input_dict['gaze_1_r']),dim=0)

            angular_error = compute_angular_error(gaze_hat,gaze)
            error += angular_error

    #Each sample contains two eyes
    error /= len(small_valset)*2
    logging.info('Validation: {0}\t'
                    'Angular Error: {angular_error:.2f}'.format(
                    epoch,angular_error = error))
    if error < best:
        torch.save(reg.state_dict(),'regressor.pth.tar')
        best = error

#clean up a bit
optimizer.zero_grad()
del (small_trainset, small_train, small_valset, small_val, optimizer, reg)
torch.cuda.empty_cache()
