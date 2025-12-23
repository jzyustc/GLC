import os
import numpy as np

from PIL import Image

import torch
from torchvision.transforms import ToTensor

import lpips
from DISTS_pytorch import DISTS
from pytorch_msssim import ms_ssim
from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance

from ._update_patch_fid import update_patch_fid

totensor = ToTensor()

def read_image(image_path):
    with open(image_path, "rb") as f:
        image_pil = Image.open(f)
        image_pil = image_pil.convert("RGB")

    image = totensor(image_pil).unsqueeze(0)
    return image

def evaluate_quality(all_bpps, input_path, output_path, log_path, patch_size=256, split_patch_num=2):
    os.makedirs(log_path, exist_ok=True)


    lpips_metric = lpips.LPIPS(net='alex',version='0.1').cuda()
    dists_metric = DISTS().cuda()
    fid_metric = FrechetInceptionDistance().cuda()
    kid_metric = KernelInceptionDistance().cuda()

    # psnr, ms-ssim, bpp, fid
    bpp_list = []
    psnr_list = []
    msssim_list = []
    lpips_list= []
    dists_list= []
    content = ""
    idx = 0
    for img_name in sorted(os.listdir(input_path)):
        # print(img_name)
        img = read_image(os.path.join(input_path, img_name)).cuda()
        img_dec = read_image(os.path.join(output_path, img_name.replace(".jpg", '.png'))).cuda()
        
        bpp = all_bpps[idx]

        if patch_size != -1:
            update_patch_fid(img, img_dec, fid_metric=fid_metric, kid_metric=kid_metric, patch_size=patch_size, split_patch_num=split_patch_num)

        mse = torch.mean((img - img_dec) ** 2)
        psnr = -10 * torch.log10(mse).item()
        msssim = ms_ssim(img, img_dec, data_range=1.).item()
        lpips_item = lpips_metric.forward(img * 2 - 1, img_dec * 2 - 1).item()
        dists_item = dists_metric.forward(img, img_dec).item()

        bpp_list.append(bpp)
        psnr_list.append(psnr)
        msssim_list.append(msssim)
        lpips_list.append(lpips_item)
        dists_list.append(dists_item)

        content_item = f"idx={idx} : bpp = {bpp:.4f}, psnr = {psnr:.4f}, msssim = {msssim:.4f}, lpips={lpips_item}, dists={dists_item}"
        print(content_item)
        content += content_item + "\n"
        
        idx += 1

    with open(f"{log_path}/items.txt", "w") as f:
        f.write(content)

    if patch_size != -1:
        fid = float(fid_metric.compute())
        kid = float(kid_metric.compute()[0])

    # all
    with open(f"{log_path}/res.txt", "w") as f:
        content = f"bpp = {np.average(bpp_list):.6f}, \
                psnr = {np.average(psnr_list):.6f}, \
                ms-ssim = {np.average(msssim_list):.6f}, \
                lpips = {np.average(lpips_list):.6f}, \
                dists = {np.average(dists_list):.6f}"
        if patch_size != -1:
            content += f", \
                fid = {fid:.6f}, \
                kid = {kid:.6f}"
        f.write(content)
