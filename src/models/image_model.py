# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
from torch import nn
import torch.nn.functional as F
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
    def __init__(self, N=256, patch=32, inplace=False):
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

        # vqgan part
        self.codebook_size = codebook_size = 16384
        self.ds = 16
        self.vqgan = VQAutoEncoder(256, 128, [1, 1, 2, 2, 4],   'nearest',2, [16], codebook_size, one_more_block_in_dec=True, swish_last=True, quant_conv=True, patch=patch)

        self.code_pred_loss = CodePredictionLoss(dim_embd=512, codebook_size=codebook_size, n_head=8, n_layers=9)
        self.code_pred_pix_loss = CodePredictionLoss(dim_embd=512, codebook_size=codebook_size, n_head=8, n_layers=9)

        # z vq codec part
        self.z_vq = deepcopy(self.vqgan.quantize)

    @staticmethod
    def get_qp_num():
        return 4

    def interpolate_q(self): # To align with video model, interpolate q from num=4 to num=64
        assert self.q_enc.shape[0] == 4 and self.q_dec.shape[0] == 4
        interpolated_weights = F.interpolate(self.q_enc.view(1, 1, 4, 256), size=(64, 256), mode='bilinear', align_corners=True)
        self.q_enc = nn.Parameter(interpolated_weights.view(64, 256, 1, 1))
        interpolated_weights = F.interpolate(self.q_dec.view(1, 1, 4, 256), size=(64, 256), mode='bilinear', align_corners=True)
        self.q_dec = nn.Parameter(interpolated_weights.view(64, 256, 1, 1))

    def test(self, x, q_index):
        curr_q_enc = self.q_enc[q_index:q_index + 1, :, :, :]
        curr_q_dec = self.q_dec[q_index:q_index + 1, :, :, :]

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
            "ref_latent": y_hat,
        }
        return result

    def recon_with_z(self, x, q_index=None):
        curr_q_enc = self.q_enc[q_index:q_index + 1, :, :, :]
        curr_q_dec = self.q_dec[q_index:q_index + 1, :, :, :]

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