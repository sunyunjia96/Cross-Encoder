#!/usr/bin/env python3

import torch.nn as nn
import torch

class ReconstructionL1Loss(object):

    def __init__(self):
        self.loss_fn = nn.L1Loss(reduction='mean')

    def __call__(self, input_dict, output_dict):

        x_1_l = input_dict['img_1_l'].detach()
        half = int(x_1_l.size(0)/2)
        x_hat_1_l = output_dict['image_hat_1_l']
        x_2_l =  input_dict['img_2_l'].detach()
        x_hat_2_l = output_dict['image_hat_2_l']
        recon_loss_l_id = self.loss_fn(torch.cat((x_1_l[:half],x_2_l[:half]),dim=0), torch.cat((x_hat_1_l[:half],x_hat_2_l[:half]),dim=0))
        recon_loss_l_g = self.loss_fn(torch.cat((x_1_l[half:],x_2_l[half:]),dim=0), torch.cat((x_hat_1_l[half:],x_hat_2_l[half:]),dim=0))
        res_loss_l = self.loss_fn(x_1_l[:half]-x_2_l[:half],x_hat_1_l[:half]-x_hat_2_l[:half])

        x_1_r = input_dict['img_1_r'].detach()
        x_hat_1_r = output_dict['image_hat_1_r']
        x_2_r =  input_dict['img_2_r'].detach()
        x_hat_2_r = output_dict['image_hat_2_r']
        recon_loss_r_id = self.loss_fn(torch.cat((x_1_r[:half],x_2_r[:half]),dim=0), torch.cat((x_hat_1_r[:half],x_hat_2_r[:half]),dim=0))
        recon_loss_r_g = self.loss_fn(torch.cat((x_1_r[half:],x_2_r[half:]),dim=0), torch.cat((x_hat_1_r[half:],x_hat_2_r[half:]),dim=0))
        res_loss_r = self.loss_fn(x_1_r[:half]-x_2_r[:half],x_hat_1_r[:half]-x_hat_2_r[:half])

        res_loss_1 = self.loss_fn(x_1_l[half:]-x_1_r[half:],x_hat_1_l[half:]-x_hat_1_r[half:])
        res_loss_2 = self.loss_fn(x_2_l[half:]-x_2_r[half:],x_hat_2_l[half:]-x_hat_2_r[half:])

        return recon_loss_l_id+recon_loss_l_g+recon_loss_r_id+recon_loss_r_g+res_loss_l+res_loss_r+0.5*res_loss_1+0.5*res_loss_2
