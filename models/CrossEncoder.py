#!/usr/bin/env python3

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.regressor import regressor
from torchvision import models

from .densenet import (
    DenseNetInitialLayers,
    DenseNetBlock,
    DenseNetTransitionDown,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CrossEncoder(nn.Module):

    def __init__(self, z_dim_app, z_dim_gaze,
                 growth_rate=32, activation_fn=nn.LeakyReLU,
                 normalization_fn=nn.InstanceNorm2d,
                 decoder_input_c=16,
                 use_triplet=False,
                 gaze_hidden_layer_neurons=64,
                 backprop_gaze_to_encoder=False,
                 labeled=False,
                 ):
        super(CrossEncoder, self).__init__()

        # Cache some specific configurations
        self.use_triplet = use_triplet
        self.gaze_hidden_layer_neurons = gaze_hidden_layer_neurons
        self.backprop_gaze_to_encoder = backprop_gaze_to_encoder

        # Define feature map dimensions at bottleneck
        bottleneck_shape = (2, 4)
        self.bottleneck_shape = bottleneck_shape

        self.encoder = models.resnet18(pretrained=True)
        self.decoder_input_c = decoder_input_c
        enc_num_all = np.prod(bottleneck_shape) * self.decoder_input_c
        self.decoder = DenseNetDecoder(
            self.decoder_input_c,
            num_blocks=4,
            growth_rate=growth_rate,
            activation_fn=activation_fn,
            normalization_fn=normalization_fn,
            compression_factor=1.0,
        )

        # The latent code parts
        self.z_dim_app = z_dim_app
        self.z_dim_gaze = z_dim_gaze
        self.head_size = 9
        z_num_all = 3 * (z_dim_gaze) + z_dim_app

        self.fc_enc = self.linear(1000, z_num_all)
        self.fc_dec = self.linear(z_num_all, enc_num_all)

        self.labeled = labeled
        if labeled:
            self.regressor = regressor(3*z_dim_gaze*2,2)

    def linear(self, f_in, f_out):
        fc = nn.Linear(f_in, f_out)
        nn.init.kaiming_normal_(fc.weight.data)
        nn.init.constant_(fc.bias.data, val=0)
        return fc

    def encode_to_z(self, data):
        x = self.encoder(data)
        enc_output_shape = x.shape

        # Create latent codes
        z_all = self.fc_enc(x)
        z_app = z_all[:, :self.z_dim_app]
        z_all = z_all[:, self.z_dim_app:]
        z_all = z_all.view(self.batch_size, -1, 3)
        z_gaze_enc = z_all[:, :self.z_dim_gaze, :]

        return [z_app, z_gaze_enc, x, enc_output_shape]

    def decode_to_image(self, codes):
        z_all = torch.cat([code.view(self.batch_size, -1) for code in codes], dim=1)
        x = self.fc_dec(z_all)
        x = x.view(self.batch_size, self.decoder_input_c, *self.bottleneck_shape)
        x = self.decoder(x)
        return x

    def maybe_do_norm(self, code):
        norm_axis = 3
        assert code.dim() == 3
        assert code.shape[-1] == 3
        b, f, _ = code.shape
        code = code.view(b, -1)
        normalized_code = F.normalize(code, dim=-1)
        return normalized_code.view(b, f, -1)

    def forward(self, data, loss_functions=None, inference=False):
        self.batch_size = data['img_1_r'].shape[0]*4

        # Encode input from a
        input_img_l = torch.cat((data['img_1_l'],data['img_2_l']),dim=0)
        input_img_r = torch.cat((data['img_1_r'],data['img_2_r']),dim=0)
        input_img = torch.cat((input_img_l,input_img_r),dim=0)
        input_img = input_img.expand(input_img.size(0),input_img.size(1)*3,input_img.size(2),input_img.size(3))
        (z_a, ze1_g, ze1_before_z, _) = self.encode_to_z(input_img)
        cut_size = int(self.batch_size/4)
        z_a_1_l, ze1_g_1_l = z_a[:cut_size], ze1_g[:cut_size]
        z_a_2_l, ze1_g_2_l = z_a[cut_size:cut_size*2], ze1_g[cut_size:cut_size*2]
        z_a_1_r, ze1_g_1_r = z_a[cut_size*2:cut_size*3], ze1_g[cut_size*2:cut_size*3]
        z_a_2_r, ze1_g_2_r = z_a[cut_size*3:], ze1_g[cut_size*3:]

        # Make each row a unit vector through L2 normalization to constrain
        # embeddings to the surface of a hypersphere
        ze1_g_1_l = self.maybe_do_norm(ze1_g_1_l)
        ze1_g_2_l = self.maybe_do_norm(ze1_g_2_l)
        ze1_g_1_r = self.maybe_do_norm(ze1_g_1_r)
        ze1_g_2_r = self.maybe_do_norm(ze1_g_2_r)

        output_dict = {
            'z_app_1_l': z_a_1_l,
            'z_gaze_enc_1_l': ze1_g_1_l,
            'z_app_2_l': z_a_2_l,
            'z_gaze_enc_2_l': ze1_g_2_l,
            'z_app_1_r': z_a_1_r,
            'z_gaze_enc_1_r': ze1_g_1_r,
            'z_app_2_r': z_a_2_r,
            'z_gaze_enc_2_r': ze1_g_2_r,
        }

        # Reconstruct

        # Switch
        half = int(cut_size/2)
         
        # Switch app
        
        z_a = torch.cat((z_a_2_l[:half],z_a_1_l[half:],
        z_a_1_l[:half],z_a_2_l[half:],
        z_a_2_r[:half],z_a_1_r[half:],
        z_a_1_r[:half],z_a_2_r[half:]),dim=0)

        # Switch gaze

        ze1_g = torch.cat((ze1_g_1_l[:half],ze1_g_1_r[half:],
        ze1_g_2_l[:half],ze1_g_2_r[half:],
        ze1_g_1_r[:half],ze1_g_1_l[half:],
        ze1_g_2_r[:half],ze1_g_2_l[half:]),dim=0)

        #No need to reconstruct during inference        
        if not inference:
            x_hat = self.decode_to_image([z_a, ze1_g])
            output_dict['image_hat_1_l'] = x_hat[:cut_size]
            output_dict['image_hat_2_l'] = x_hat[cut_size:cut_size*2]
            output_dict['image_hat_1_r'] = x_hat[cut_size*2:cut_size*3]
            output_dict['image_hat_2_r'] = x_hat[cut_size*3:]

        # If loss functions specified, apply them
        if loss_functions is not None:
            losses_dict = OrderedDict()
            for key, func in loss_functions.items():
                losses = func(data, output_dict)  # may be dict or single value
                if isinstance(losses, dict):
                    for sub_key, loss in losses.items():
                        losses_dict[key + '_' + sub_key] = loss
                else:
                    losses_dict[key] = losses
            return output_dict, losses_dict

        return output_dict

class DenseNetDecoder(nn.Module):

    def __init__(self, c_in, growth_rate=8, num_blocks=4, num_layers_per_block=4,
                 p_dropout=0.0, compression_factor=1.0,
                 activation_fn=nn.ReLU, normalization_fn=nn.BatchNorm2d,
                 use_skip_connections_from=None):
        super(DenseNetDecoder, self).__init__()

        self.use_skip_connections = (use_skip_connections_from is not None)
        if self.use_skip_connections:
            c_to_concat = use_skip_connections_from.c_at_end_of_each_scale
            c_to_concat = list(reversed(c_to_concat))[1:]
        else:
            c_to_concat = [0] * (num_blocks + 2)

        assert (num_layers_per_block % 2) == 0
        c_now = c_in
        for i in range(num_blocks):
            i_ = i + 1
            # Define dense block
            self.add_module('block%d' % i_, DenseNetBlock(
                c_now,
                num_layers=num_layers_per_block,
                growth_rate=growth_rate,
                p_dropout=p_dropout,
                activation_fn=activation_fn,
                normalization_fn=normalization_fn,
                transposed=True,
            ))
            c_now = list(self.children())[-1].c_now

            # Define transition block if not last layer
            if i < (num_blocks - 1):
                self.add_module('trans%d' % i_, DenseNetTransitionUp(
                    c_now, p_dropout=p_dropout,
                    compression_factor=compression_factor,
                    activation_fn=activation_fn,
                    normalization_fn=normalization_fn,
                ))
                c_now = list(self.children())[-1].c_now
                c_now += c_to_concat[i]

        # Last up-sampling conv layers
        self.last = DenseNetDecoderLastLayers(c_now,
                                              growth_rate=growth_rate,
                                              activation_fn=activation_fn,
                                              normalization_fn=normalization_fn,
                                              skip_connection_growth=c_to_concat[-1])
        self.c_now = 1

    def forward(self, x):
        # Apply initial layers and dense blocks
        for name, module in self.named_children():
            x = module(x)
        return x


class DenseNetDecoderLastLayers(nn.Module):

    def __init__(self, c_in, growth_rate=8, activation_fn=nn.ReLU,
                 normalization_fn=nn.BatchNorm2d,
                 skip_connection_growth=0):
        super(DenseNetDecoderLastLayers, self).__init__()
        # First deconv
        self.conv1 = nn.ConvTranspose2d(c_in, 4 * growth_rate, bias=False,
                                        kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        nn.init.kaiming_normal_(self.conv1.weight.data)

        # Second deconv
        c_in = 4 * growth_rate + skip_connection_growth
        self.norm2 = normalization_fn(c_in, track_running_stats=False).to(device)
        self.act = activation_fn(inplace=True)
        self.conv2 = nn.ConvTranspose2d(c_in, 2 * growth_rate, bias=False,
                                        kernel_size=1, stride=1, padding=0,
                                        output_padding=0)
        nn.init.kaiming_normal_(self.conv2.weight.data)

        # Final conv
        c_in = 2 * growth_rate
        c_out = 1
        self.norm3 = normalization_fn(c_in, track_running_stats=False).to(device)
        self.conv3 = nn.Conv2d(c_in, c_out, bias=False,
                               kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv3.weight.data)
        self.c_now = c_out

    def forward(self, x):
        x = self.conv1(x)
        #
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        #
        x = self.norm3(x)
        x = self.act(x)
        x = self.conv3(x)
        return x


class DenseNetTransitionUp(nn.Module):

    def __init__(self, c_in, compression_factor=0.1, p_dropout=0.1,
                 activation_fn=nn.ReLU, normalization_fn=nn.BatchNorm2d):
        super(DenseNetTransitionUp, self).__init__()
        c_out = int(compression_factor * c_in)
        self.norm = normalization_fn(c_in, track_running_stats=False).to(device)
        self.act = activation_fn(inplace=True)
        self.conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=3,
                                       stride=2, padding=1, output_padding=1,
                                       bias=False).to(device)
        nn.init.kaiming_normal_(self.conv.weight.data)
        self.drop = nn.Dropout2d(p=p_dropout) if p_dropout > 1e-5 else None
        self.c_now = c_out

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x
