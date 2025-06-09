# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
from torch import nn
from copy import deepcopy

from .common_model import CompressionModel
from .layers import DepthConvBlock, DepthConvBlock2, ResidualBlockUpsample
from .vqgan_arch import VQAutoEncoder
from .loss import CodePredictionLoss


class GLC_Encoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.enc_1 = nn.Sequential(
            DepthConvBlock2(N, N, inplace=inplace),
            DepthConvBlock2(N, N, inplace=inplace),
        )
        self.enc_2 = nn.Sequential(
            DepthConvBlock2(N, N, inplace=inplace),
            DepthConvBlock2(N, N, inplace=inplace),
        )
    def forward(self, x, quant_step):
        out = self.enc_1(x)
        out = out * quant_step
        return self.enc_2(out)


class GLC_Decoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.dec_1 = nn.Sequential(
            DepthConvBlock2(N, N, inplace=inplace),
            DepthConvBlock2(N, N, inplace=inplace),
        )
        self.dec_2 = nn.Sequential(
            DepthConvBlock2(N, N, inplace=inplace),
            DepthConvBlock2(N, N, inplace=inplace),
        )
    def forward(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        return self.dec_2(out)


class GLC_Image(CompressionModel):
    def __init__(self, N=256, anchor_num=4, patch=32, inplace=False):
        super().__init__()
        self.enc = GLC_Encoder(N, inplace)
        
        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )
        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(N, N, 2, inplace=inplace),
            ResidualBlockUpsample(N, N, 2, inplace=inplace),
            DepthConvBlock(N, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 2, inplace=inplace),
            DepthConvBlock(N * 2, N * 3, inplace=inplace),
        )

        self.y_spatial_prior_reduction = nn.Conv2d(N * 3, N * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock(N * 2, N * 2, inplace=inplace)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock(N * 2, N * 2, inplace=inplace)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock(N * 2, N * 2, inplace=inplace)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(N * 2, N * 2, inplace=inplace),
            DepthConvBlock(N * 2, N * 2, inplace=inplace),
            DepthConvBlock(N * 2, N * 2, inplace=inplace),
        )

        self.dec = GLC_Decoder(N, inplace)

        self.q_enc = nn.Parameter(torch.ones((4, 256, 1, 1)))
        self.q_dec = nn.Parameter(torch.ones((4, 256, 1, 1)))

        self.N = N
        self.anchor_num = int(anchor_num)

        # vqgan part
        self.codebook_size = codebook_size = 16384
        self.ds = 16
        self.vqgan = VQAutoEncoder(256, 128, [1, 1, 2, 2, 4],   'nearest',2, [16], codebook_size, one_more_block_in_dec=True, swish_last=True, quant_conv=True, patch=patch)

        self.code_pred_loss = CodePredictionLoss(dim_embd=512, codebook_size=codebook_size, n_head=8, n_layers=9)
        self.code_pred_pix_loss = CodePredictionLoss(dim_embd=512, codebook_size=codebook_size, n_head=8, n_layers=9)

        # z vq codec part
        self.z_vq = deepcopy(self.vqgan.quantize)

    def get_all_q(self, q_index, batch):
        curr_q_enc = self.get_curr_q(self.q_enc, q_index, batch)
        curr_q_dec = self.get_curr_q(self.q_dec, q_index, batch)
        return curr_q_enc, curr_q_dec

    def test(self, x, q_index):
        curr_q_enc, curr_q_dec = self.get_all_q(q_index, 1)

        y_ori = self.vqgan.encoder(x)
        y = self.enc(y_ori, curr_q_enc)

        z = self.hyper_enc(y)
        
        index = self.z_vq.get_indices(z)
        z_hat = self.z_vq.get_quan_feat(index, (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
                y_spatial_prior_reduction=self.y_spatial_prior_reduction)

        y_hat = self.dec(y_hat, curr_q_dec)
        x_hat = self.vqgan.generator(y_hat)

        # bpp
        bit_y = self.get_y_gaussian_bits(y_q, scales_hat).sum().item()
        bit_z = z_hat.shape[-2] * z_hat.shape[-1] * math.log2(self.codebook_size)
        
        result = {
            'bit': bit_y + bit_z,
            'bit_y': bit_y,
            'bit_z': bit_z,
            'x_hat': x_hat,
        }
        return result

    def recon_with_z(self, x, q_index=None):
        curr_q_enc, curr_q_dec = self.get_all_q(q_index, 1)

        # vqgan encoder
        y_ori = self.vqgan.encoder(x)
        y = self.enc(y_ori, curr_q_enc)
        z = self.hyper_enc(y)
        
        ## vq for z
        z_hat, z_vq_loss, z_vq_info = self.z_vq(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        y_hat = self.forward_four_part_prior_recon_with_z(
            y, params,
            self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
            y_spatial_prior_reduction=self.y_spatial_prior_reduction)
        
        y_hat = self.dec(y_hat, curr_q_dec)
        x_hat = self.vqgan.generator(y_hat)

        result = {
            "x_hat": x_hat
        }
        return result
