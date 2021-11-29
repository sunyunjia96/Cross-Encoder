#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Train Cross-Encoder')

# architecture
parser.add_argument('--densenet-growthrate', type=int, default=32,
                    help='growth rate of encoder/decoder base densenet archi. (default: 32)')
parser.add_argument('--z-dim-app', type=int, default=32,
                    help='size of 1D latent code for appearance (default: 32)')
parser.add_argument('--z-dim-gaze', type=int, default=4,
                    help='size of 2nd dim. of 3D latent code for each gaze direction (default: 4)')
parser.add_argument('--decoder-input-c', type=int, default=32,
                    help='size of feature map stack as input to decoder (default: 32)')

# training
parser.add_argument('--base-lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (to be multiplied with batch size) (default: 0.00005)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='training batch size (default: 128)')
parser.add_argument('--num-training-epochs', type=float, default=20, metavar='N',
                    help='number of steps to train (default: 20)')
parser.add_argument('--print-freq-train', type=int, default=20, metavar='N',
                    help='print training statistics after every N iterations (default: 20)')
# data
parser.add_argument('--num-data-loaders', type=int, default=0, metavar='N',
                    help='number of data loading workers (default: 0)')
# logging
parser.add_argument('--use-tensorboard', action='store_true', default=False,
                    help='create tensorboard logs (stored in the args.save_path directory)')
parser.add_argument('--save-path', type=str, default='.',
                    help='path to save network parameters (default: .)')
# image saving
parser.add_argument('--save-image-samples', type=str, default=1,
                    help='save image samples or not, 0 for not saving, 1 for saving')
parser.add_argument('--image-path', type=str, default='images',
                    help='path to save image samples')
parser.add_argument('--save-freq-images', type=int, default=10000,
                    help='save sample images after every N iterations (default: 1000)')

args = parser.parse_args()

import numpy as np
from collections import OrderedDict
import gc
import time
import os

#import moviepy.editor as mpy
import cv2

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from dataloaders import Columbia, UTMultiview, MPIIGaze, XGaze, Unite
import torchvision.transforms as transforms

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from models import CrossEncoder
network = CrossEncoder(
    growth_rate=args.densenet_growthrate,
    z_dim_app=args.z_dim_app,
    z_dim_gaze=args.z_dim_gaze,
    decoder_input_c=args.decoder_input_c,
)

network = network.to(device)

optimizer = optim.Adam(network.parameters(),lr=args.base_lr)

if torch.cuda.device_count() > 1:
    logging.info('Using %d GPUs!' % torch.cuda.device_count())
    network = nn.DataParallel(network)

from losses import ReconstructionL1Loss

loss_functions = OrderedDict()
loss_functions['recon_l1'] = ReconstructionL1Loss()

################################################
# Create the datasets.

#The path of the dataset.
data_root = '../../datas/MPIIGaze/MPIIGaze/Data/Normalized/'
#The training group.
#The validation group need to be excluded.
data_group = [i for i in range(0,13)]+[i for i in range(14,15)]

data_set = MPIIGaze(data_root,data_group,
        transform_E=transforms.Compose([
            transforms.Resize((32,64)),transforms.ToTensor(),
        ]))
data_loader = torch.utils.data.DataLoader(data_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_data_loaders, pin_memory=True)

def send_data_dict_to_gpu(data):
    for k in data:
        v = data[k]
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    return data


def recover_images(x):
    x = x.cpu().detach().numpy()
    x = x * 255.0
    x = np.clip(x, 0, 255)  
    x = np.transpose(x, [0, 2, 3, 1])  # CHW to HWC
    x = x.astype(np.uint8)
    x = x[:, :, :, ::-1]  # RGB to BGR for OpenCV
    return x

############################
# Load weights if available

from utils import CheckpointsManager
saver = CheckpointsManager(network, args.save_path)
initial_step = saver.load_last_checkpoint()

######################
# Training step update

class RunningStatistics(object):
    def __init__(self):
        self.losses = OrderedDict()

    def add(self, key, value):
        if key not in self.losses:
            self.losses[key] = []
        self.losses[key].append(value)

    def means(self):
        return OrderedDict([
            (k, np.mean(v)) for k, v in self.losses.items() if len(v) > 0
        ])

    def reset(self):
        for key in self.losses.keys():
            self.losses[key] = []


time_epoch_start = None
num_elapsed_epochs = 0

def execute_training_step(current_step):
    global data_iterator, time_epoch_start, num_elapsed_epochs
    time_iteration_start = time.time()

    # Get data
    try:
        if time_epoch_start is None:
            time_epoch_start = time.time()
        time_batch_fetch_start = time.time()
        input_dict = next(data_iterator)
    except StopIteration:
        # Epoch counter and timer
        num_elapsed_epochs += 1
        time_epoch_end = time.time()
        time_epoch_diff = time_epoch_end - time_epoch_start
        if args.use_tensorboard:
            tensorboard.add_scalar('timing/epoch', time_epoch_diff, num_elapsed_epochs)

        # Done with an epoch now...!
        if num_elapsed_epochs % 5 == 0:
            saver.save_checkpoint(current_step)

        np.random.seed()  # Ensure randomness

        # Some cleanup
        data_iterator = None
        torch.cuda.empty_cache()
        gc.collect()

        # Restart!
        time_epoch_start = time.time()
        global data_loader, data_set
        data_set.resample()
        data_loader = torch.utils.data.DataLoader(data_set,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_data_loaders, pin_memory=True)
        data_iterator = iter(data_loader)
        time_batch_fetch_start = time.time()
        input_dict = next(data_iterator)

    # get the inputs
    input_dict = send_data_dict_to_gpu(input_dict)
    running_timings.add('batch_fetch', time.time() - time_batch_fetch_start)

    # zero the parameter gradient
    network.train()
    optimizer.zero_grad()

    # forward + backward + optimize
    time_forward_start = time.time()
    output_dict, loss_dict = network(input_dict, loss_functions=loss_functions)

    # If doing multi-GPU training, just take an average
    for key, value in loss_dict.items():
        if value.dim() > 0:
            value = torch.mean(value)
            loss_dict[key] = value

    # Construct main loss
    loss = loss_dict['recon_l1']

    # Optimize main objective
    loss.backward()
    optimizer.step()

    # Register timing
    time_backward_end = time.time()
    running_timings.add('forward_and_backward', time_backward_end - time_forward_start)

    # Store values for logging later
    for key, value in loss_dict.items():
        loss_dict[key] = value.detach().cpu()
    for key, value in loss_dict.items():
        running_losses.add(key, value.numpy())

    running_timings.add('iteration', time.time() - time_iteration_start)

    return input_dict, output_dict


############
# Main

num_training_steps = int(args.num_training_epochs * len(data_loader))
logging.info('Training')
last_training_step = num_training_steps - 1
if args.use_tensorboard:
    from tensorboardX import SummaryWriter
    tensorboard = SummaryWriter(log_dir=args.save_path)

data_iterator = iter(data_loader)
running_losses = RunningStatistics()
running_timings = RunningStatistics()

for current_step in range(initial_step, num_training_steps):

    ################
    # Training
    input_dict, output_dict = execute_training_step(current_step)

    if current_step % args.print_freq_train == args.print_freq_train - 1:
        conv1_wt_lr = optimizer.param_groups[0]['lr']
        running_loss_means = running_losses.means()
        logging.info('Losses at [%7d]: %s' %
                     (current_step + 1,
                      ', '.join(['%s: %.5f' % v
                                 for v in running_loss_means.items()])))
        if args.use_tensorboard:
            tensorboard.add_scalar('train_lr', conv1_wt_lr, current_step + 1)
            for k, v in running_loss_means.items():
                tensorboard.add_scalar('train/' + k, v, current_step + 1)
        running_losses.reset()

    # Print some timing statistics
    if current_step % 100 == 99:
        if args.use_tensorboard:
            for k, v in running_timings.means().items():
                tensorboard.add_scalar('timing/' + k, v, current_step + 1)
        running_timings.reset()

    # Print some memory statistics
    if current_step % 5000 == 0:
        for i in range(torch.cuda.device_count()):
            bytes = (torch.cuda.memory_allocated(device=i)
                     + torch.cuda.memory_cached(device=i))
            logging.info('GPU %d: probably allocated approximately %.2f GB' % (i, bytes / 1e9))

    #####################
    # Visualization

    # Save image samples
    if (args.save_image_samples > 0
        and (current_step % args.save_freq_images
             == (args.save_freq_images - 1)
             or current_step == last_training_step)):
        network.eval()
        torch.cuda.empty_cache()
        output_images = recover_images(torch.cat((output_dict['image_hat_1_l'],output_dict['image_hat_1_r']),dim=3))
        input_images = recover_images(torch.cat((input_dict['img_1_l'],input_dict['img_1_r']),dim=3))
        cv2.imwrite(os.path.join(args.image_path,'decimg_'+str(current_step)+'_1.jpg'),output_images[0])
        cv2.imwrite(os.path.join(args.image_path,'orgimg_'+str(current_step)+'_1.jpg'),input_images[0])
        output_images = recover_images(torch.cat((output_dict['image_hat_2_l'],output_dict['image_hat_2_r']),dim=3))
        input_images = recover_images(torch.cat((input_dict['img_2_l'],input_dict['img_2_r']),dim=3))
        cv2.imwrite(os.path.join(args.image_path,'decimg_'+str(current_step)+'_2.jpg'),output_images[0])
        cv2.imwrite(os.path.join(args.image_path,'orgimg_'+str(current_step)+'_2.jpg'),input_images[0])

        torch.cuda.empty_cache()

logging.info('Finished Training')

# Save model parameters
saver.save_checkpoint(current_step)

if args.use_tensorboard:
    tensorboard.close()
    del tensorboard

# Clean up a bit
optimizer.zero_grad()
del (data_loader, optimizer)

logging.info('Done')
