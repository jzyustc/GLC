# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import torch.nn.functional as F

from ..utils.lpips.lpips import LPIPS
from .layers import TransformerSALayer

class CodePredictionLoss(nn.Module):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9, 
                codebook_size=1024, latent_size=256):
        super(CodePredictionLoss, self).__init__()

        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0) 
                                    for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False)
        )
        
    def forward(self, y_hat):
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, y_hat.shape[0],1)
        # BCHW -> BC(HW) -> (HW)BC
        feat_emb = self.feat_emb(y_hat.flatten(2).permute(2,0,1))
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        # output logits
        logits = self.idx_pred_layer(query_emb) # (hw)bn
        logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n
        return logits


def calculate_vqgan_results(x, vqgan):
    with torch.no_grad():
        y_vqgan = vqgan.encoder(x)
        _, _, quant_stats = vqgan.quantize(y_vqgan)

        min_encoding_indices = quant_stats['min_encoding_indices']
        idx_gt = min_encoding_indices.view(x.shape[0], -1).detach()
        return {
            "y_vqgan" : y_vqgan,
            "idx_gt" : idx_gt
        }


class CodeLevelLoss(nn.Module):

    def __init__(self, vqgan_model):
        super(CodeLevelLoss, self).__init__()
        self.vqgan = vqgan_model
        self.mse = nn.MSELoss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, y_hat, logits, results_vqgan, repeat_num, x=None, net_g=None):
        N = y_hat.shape[1]
        pixel_num = y_hat.shape[2] * y_hat.shape[3]
        
        if results_vqgan is None:
            results_vqgan = calculate_vqgan_results(x, self.vqgan)

        # part 1: mse 
        feat_mse_loss = self.mse(results_vqgan["y_vqgan"].repeat(repeat_num, 1, 1, 1), y_hat)
        feat_mse_loss = torch.sum(feat_mse_loss, dim=(1, 2, 3)) / pixel_num  / N

        # part 2: code idx prediction loss
        cross_entropy_loss = self.cross_entropy(logits.permute(0, 2, 1), results_vqgan["idx_gt"].repeat(repeat_num, 1))
        cross_entropy_loss = torch.sum(cross_entropy_loss, dim=(1)) / pixel_num
        
        return {
            "feat_mse_loss": feat_mse_loss,
            "cross_entropy_loss": cross_entropy_loss
        }

def cal_l1_Loss(x, x_hat):
    N = x.shape[1]
    pixel_num = x.shape[2] * x.shape[3]

    l1_loss = F.l1_loss(x, x_hat, reduction='none')
    l1_loss = torch.sum(l1_loss, dim=(1, 2, 3)) / pixel_num / N
    return l1_loss

def cal_mse_Loss(x, x_hat):
    N = x.shape[1]
    pixel_num = x.shape[2] * x.shape[3]

    l1_loss = F.mse_loss(x, x_hat, reduction='none')
    l1_loss = torch.sum(l1_loss, dim=(1, 2, 3)) / pixel_num / N
    return l1_loss

def cal_ce_Loss(logits, idx_gt):
    pixel_num = logits.shape[1]

    cross_entropy_loss = F.cross_entropy(logits.permute(0, 2, 1), idx_gt, reduction='none')
    cross_entropy_loss = torch.sum(cross_entropy_loss, dim=(1)) / pixel_num
    return cross_entropy_loss

def get_lpips_model():
    return LPIPS(net="vgg", spatial=False).eval()

class LPIPSLoss(nn.Module):
    def __init__(self, lpips_model, use_input_norm=True, range_norm=True,):
        super(LPIPSLoss, self).__init__()
        self.perceptual = lpips_model
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if self.range_norm:
            pred   = (pred + 1) / 2
            target = (target + 1) / 2
        if self.use_input_norm:
            pred   = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        lpips_loss = self.perceptual(target.contiguous(), pred.contiguous())
        return lpips_loss.reshape(-1)

class GANLoss(nn.Module):

    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.ReLU()
        
    def forward(self, input, target_is_real, is_disc=False):
        if is_disc:  # for discriminators in hinge-gan
            input = -input if target_is_real else input
            loss = self.loss(1 + input)
        else:  # for generators in hinge-gan
            loss = -input
        loss = torch.mean(loss, axis=(1, 2, 3))
        return loss

def calculate_adaptive_weight(net_g, y_hat, l1_loss, lpips_loss, g_gan, disc_weight_max=1.0, g_gan_adapt_layer="vqgan_dec"):
    if next(net_g.named_parameters())[0][:len("module.")] == "module.":
        net = net_g.module
    else:
        net = net_g

    if g_gan_adapt_layer == "vqgan_dec":
        last_layer = net.vqgan.generator.blocks[-1].weight
        d_weights = torch.zeros_like(l1_loss)
        for i in range(l1_loss.shape[0]):
            g_grads = torch.autograd.grad(g_gan[i], last_layer, retain_graph=True)[0]
            recon_grads = torch.autograd.grad(l1_loss[i] + lpips_loss[i], last_layer, retain_graph=True, allow_unused=True)[0]

            d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()

            d_weights[i] = d_weight

    elif g_gan_adapt_layer == "codec_dec":
        y_hat_g_grads = torch.autograd.grad(g_gan, y_hat, grad_outputs=torch.ones_like(g_gan), retain_graph=True)[0]
        y_hat_recon_grads = torch.autograd.grad(l1_loss + lpips_loss, y_hat, grad_outputs=torch.ones_like(g_gan), retain_graph=True)[0]

        d_weights = torch.norm(y_hat_recon_grads.flatten(1, 3), dim=1) / (torch.norm(y_hat_g_grads.flatten(1, 3), dim=1) + 1e-4)
        d_weights = torch.clamp(d_weights, 0.0, disc_weight_max).detach()

    return d_weights
