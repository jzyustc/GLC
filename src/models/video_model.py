# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from einops import rearrange

from .common_model import CompressionModel
from .layers import DepthConvBlock3
from .vqgan_arch import VQAutoEncoder


class GLC_Video_Encoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.enc_1 = nn.Sequential(
            DepthConvBlock3(N+N, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
        )
        self.enc_2 = nn.Sequential(
            DepthConvBlock3(N, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
        )

    def forward(self, x, context, quant_step):
        out = self.enc_1(torch.cat([x, context], dim=1))
        out = out * quant_step
        return self.enc_2(out)


class GLC_Video_Decoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.dec_1 = nn.Sequential(
            DepthConvBlock3(N + N, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
        )
        self.dec_2 = nn.Sequential(
            DepthConvBlock3(N, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
        )
    def forward(self, x, context, quant_step):
        out = self.dec_1(torch.cat([x, context], dim=1))
        out = out * quant_step
        return self.dec_2(out)


class PriorFusionAdaptor(nn.Module):
    def __init__(self, in_ch, out_ch, inplace=False):
        super().__init__()
        self.conv1 = DepthConvBlock3(in_ch, in_ch, inplace=inplace)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1, padding=0)

    def forward(self, x, ctx):
        x = self.conv1(torch.cat([x, ctx], dim=1))
        x = self.conv2(x)
        return x


class PriorFusion(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock3(N * 3, N * 3, inplace=inplace),
            DepthConvBlock3(N * 3, N * 3, inplace=inplace),
        )

    def forward(self, x):
        return self.conv(x)


class TemporalCombine(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock3(N + N // 2, N + N // 2, inplace=inplace),
            DepthConvBlock3(N + N // 2, N + N // 2, inplace=inplace),
        )

    def forward(self, x, cxt):
        x = torch.cat((x, cxt), dim=1)
        return self.conv(x)


class SpatialPrior(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock3(N * 4, N * 3, inplace=inplace),
            DepthConvBlock3(N * 3, N * 3, inplace=inplace),
            DepthConvBlock3(N * 3, N * 2, inplace=inplace),
        )

    def forward(self, x):
        return self.conv(x)


class WeightMapGenerator(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads=8, K=8, inplace=False) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.K = K
        self.conv = nn.Sequential(
            DepthConvBlock3(in_channel, in_channel, inplace=inplace),
            nn.LeakyReLU(),
            DepthConvBlock3(in_channel, in_channel, inplace=inplace),
            nn.LeakyReLU(),
            DepthConvBlock3(in_channel, in_channel, inplace=inplace),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, self.K * num_heads, 1)
        )
        self.activation = nn.Sigmoid() # nn.Identity() # nn.Sigmoid() # nn.Tanh()

    def forward(self, x):
        B, _, H, W = x.shape
        coe_map = self.conv(x)
        return self.activation(coe_map)


class Tokenizer(nn.Module):
    def __init__(self, input_channel, channel, K=8, num_heads=8, inplace=False) -> None:
        super().__init__()
        self.channel = channel
        self.K = K
        self.num_heads = num_heads
        self.proj = nn.Sequential(
                    DepthConvBlock3(input_channel, channel, inplace=inplace),
                    nn.LeakyReLU(),
                    nn.Conv2d(channel, channel, 1),
                )

        self.out_proj = nn.Linear(channel, channel)

        self.prior_extraction = DepthConvBlock3(input_channel, channel, inplace=inplace)

        self.enhancement = nn.Sequential(
                DepthConvBlock3(self.K * num_heads + channel, self.K * num_heads + channel, inplace=inplace),
                nn.LeakyReLU(),
                DepthConvBlock3(self.K * num_heads + channel, self.K * num_heads, inplace=inplace),
            )
        self.activation = nn.Sigmoid() # nn.Identity() #  # nn.Tanh()

    def forward(self, x, weight_map):
        B, _, H, W = x.shape

        # generate weight map
        prior = self.prior_extraction(x)

        weight_map = self.enhancement(torch.cat((prior, weight_map), dim=1))
        weight_map = self.activation(weight_map)
        weight_map = rearrange(weight_map, 'b (k n c) h w -> b k n c h w', k=self.K, c=1, n=self.num_heads)
        x = self.proj(x)

        x = rearrange(x, 'b (k n c) h w -> b k n c h w', k = 1, n=self.num_heads)

        out = x * weight_map
        out = out.view(B, self.K, self.channel, H * W).mean(dim=3)
        out = self.out_proj(out)
        out = rearrange(out, 'b (k m) c -> b c k m', k=self.K, m=1)

        return out


class InvertTokenizer(nn.Module):
    def __init__(self, channel, K=8, num_heads=8, inplace=False) -> None:
        super().__init__()
        self.channel = channel
        self.K = K
        self.num_heads = num_heads
        self.proj = nn.Conv2d(channel, channel, 1)

        self.out_proj = nn.Sequential(
                    DepthConvBlock3(channel, channel, inplace=inplace),
                    nn.LeakyReLU(),
                    nn.Conv2d(channel, channel, 1),
                )

    def forward(self, tokens, weight_map):
        B, _, H, W = weight_map.shape
        _, C, K, _ = tokens.shape

        tokens = self.proj(tokens).view(B, C, self.K)
        tokens = rearrange(tokens, 'b (n c h w) k -> b k n c h w', n=self.num_heads, k=self.K, h=1,w=1)

        weight_map = weight_map.view(B, self.K, self.num_heads, 1, H, W)
        # print(tokens.shape, weight_map.shape)
        out = (tokens*weight_map).view(B, self.K, C, H, W)
        out = out.mean(dim=1)
        out = self.out_proj(out)

        return out


class FeatureExtractor(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.context_refine = nn.Sequential(
            DepthConvBlock3(N, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            nn.Conv2d(N, N, 1)
        )

    def forward(self, x):
        x = self.context_refine(x)
        return x

class GLC_Video(CompressionModel):
    def __init__(self, N=256, patch=32, inplace=False, hyper_K=16):
        super().__init__()

        self.feature_adaptor_I = nn.Conv2d(N, N, 3, stride=1, padding=1)
        self.feature_adaptor = nn.ModuleList([nn.Conv2d(N, N, 1) for _ in range(3)])
        self.feature_extractor = FeatureExtractor(N, inplace=inplace)

        self.encoder = GLC_Video_Encoder(N, inplace=inplace)

        self.temporal_cond = nn.Conv2d(N, N // 2, 3, stride=1, padding=1)
        
        self.y_temporal_combine = TemporalCombine(N, inplace=inplace)

        self.hyper_K = hyper_K

        self.weight_generator = WeightMapGenerator(N, N, K=self.hyper_K, inplace=inplace)
        self.hyper_tokenizer = Tokenizer(N + N // 2, N, K=self.hyper_K, inplace=inplace)
        self.hyper_inv_tokenizer = InvertTokenizer(N, K=self.hyper_K, inplace=inplace)

        self.temporal_prior_encoder = nn.Conv2d(N, N, 3, stride=1, padding=1)

        self.prior_fusion_adaptor_0 = PriorFusionAdaptor(N * 2, N * 3, inplace=inplace)
        self.prior_fusion_adaptor_1 = PriorFusionAdaptor(N * 2, N * 3, inplace=inplace)

        self.y_prior_fusion = PriorFusion(N, inplace=inplace)
        self.y_spatial_prior = SpatialPrior(N, inplace=inplace)

        self.decoder = GLC_Video_Decoder(N, inplace=inplace)

        self.y_q_enc = nn.Parameter(torch.ones((self.get_qp_num(), N, 1, 1)))
        self.y_q_dec = nn.Parameter(torch.ones((self.get_qp_num(), N, 1, 1)))

        # vqgan part
        self.codebook_size = codebook_size = 16384
        self.ds = 16
        self.vqgan = VQAutoEncoder(256, 128, [1, 1, 2, 2, 4],   'nearest',2, [16], codebook_size, one_more_block_in_dec=True, swish_last=True, quant_conv=True, patch=patch)

        # z vq codec part
        self.z_vq = deepcopy(self.vqgan.quantize)

        self.dpb = {
            'ref_frame': None,
            'ref_latent': None,
            'ref_y': None,
        }
        self.frame_idx = 1

    @staticmethod
    def get_qp_num():
        return 64

    def multi_scale_feature_extractor(self, dpb, fa_idx):
        if dpb["ref_y"] is None:
            latent_feature = self.feature_adaptor_I(dpb["ref_latent"])
        else:
            latent_feature = self.feature_adaptor[fa_idx](dpb["ref_latent"])
        return self.feature_extractor(latent_feature)

    def context_generation(self, dpb, fa_idx):
        return self.multi_scale_feature_extractor(dpb, fa_idx)

    def res_prior_param_decoder(self, hierarchical_params, dpb, temporal_params):
        ref_y = dpb["ref_y"]
        if ref_y is None: # from i-frame
            params = self.prior_fusion_adaptor_0(hierarchical_params, temporal_params)
        else:
            params = self.prior_fusion_adaptor_1(hierarchical_params, temporal_params)
        params = self.y_prior_fusion(params)
        return params


    def test(self, x, dpb, q_index, fa_idx=0):
        index = torch.tensor([q_index], dtype=torch.int32, device=x.device)
        y_q_enc = torch.index_select(self.y_q_enc, 0, index)
        y_q_dec = torch.index_select(self.y_q_dec, 0, index)

        # temporal context
        context = self.context_generation(dpb, fa_idx)

        # vqgan encoder
        latent_ori = self.vqgan.encoder(x)

        # codec
        y = self.encoder(latent_ori, context, y_q_enc)

        temporal_params = self.temporal_prior_encoder(context)
        weight_map = self.weight_generator(temporal_params)

        temporal_cond = self.temporal_cond(context)
        hyper_inp = self.y_temporal_combine(y, temporal_cond)
        hyper_inp, slice_shape = self.pad_for_y(hyper_inp)

        z = self.hyper_tokenizer(hyper_inp, weight_map)
        z_hat, z_vq_loss, z_vq_info = self.z_vq(z)
    
        # count z bits
        index_unit_length = int(np.log2(self.z_vq.codebook_size))
        index_len = z_hat.shape[-1] * z.shape[-2]
        bits_z = (index_len * index_unit_length + 7) // 8 * 8

        hierarchical_params = self.hyper_inv_tokenizer(z_hat, weight_map)
        hierarchical_params = self.slice_to_y(hierarchical_params, slice_shape)

        params = self.res_prior_param_decoder(hierarchical_params, dpb, temporal_params)
        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(y, params, self.y_spatial_prior)

        rec_latent = self.decoder(y_hat, context, y_q_dec)
            
        x_hat = self.vqgan.generator(rec_latent)
        x_hat = x_hat.clamp_(-1, 1)

        _, _, H, W = x.size()
        pixel_num = H * W

        y_for_bit = y_q
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.ones_like(bpp_y) * bits_z / pixel_num

        bpp = bpp_y + bpp_z
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num

        return {"bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "dpb": {
                    "ref_frame": x_hat,
                    "ref_latent": rec_latent,
                    "latent_ori": latent_ori,
                    "ref_y": y_hat,
                },
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                "z_vq_loss": z_vq_loss,
            }
