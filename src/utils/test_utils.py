# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
from unittest.mock import patch
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


def from_0_1_to_minus1_1(value):
    return (value - 0.5)*2.0

def from_minus1_1_to_0_1(value):
    return ((value/2.0) + 0.5)


def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def set_torch_env():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    try:
        # require pytorch >= 2.2.0
        torch.utils.deterministic.fill_uninitialized_memory = False
    except Exception:  # pylint: disable=W0718
        pass


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f"created folder: {path}")


def consume_prefix_in_state_dict_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            state_dict[new_key] = state_dict.pop(key)

def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    if "net_g" in ckpt:
        ckpt = ckpt["net_g"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt

def init_func():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)

@patch('json.encoder.c_make_encoder', None)
def dump_json(obj, fid, float_digits=-1, **kwargs):
    of = json.encoder._make_iterencode  # pylint: disable=W0212

    def inner(*args, **kwargs):
        args = list(args)
        # fifth argument is float formater which we will replace
        args[4] = lambda o: format(o, '.%df' % float_digits)
        return of(*args, **kwargs)

    with patch('json.encoder._make_iterencode', wraps=inner):
        json.dump(obj, fid, **kwargs)


def generate_log_json(frame_num, frame_pixel_num, test_time, frame_types, bits, psnrs, ssims, 
                      lpips_list=None, dists_list=None,
                      verbose=False):
    include_yuv = len(psnrs[0]) > 1
    assert not include_yuv or (len(psnrs[0]) == 4 and len(ssims[0]) == 4)
    i_bits = 0
    i_psnr = 0
    i_psnr_y = 0
    i_psnr_u = 0
    i_psnr_v = 0
    i_ssim = 0
    i_ssim_y = 0
    i_ssim_u = 0
    i_ssim_v = 0
    i_lpips = 0
    i_dists = 0
    p_bits = 0
    p_psnr = 0
    p_psnr_y = 0
    p_psnr_u = 0
    p_psnr_v = 0
    p_ssim = 0
    p_ssim_y = 0
    p_ssim_u = 0
    p_ssim_v = 0
    p_lpips = 0
    p_dists = 0
    i_num = 0
    p_num = 0
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            i_bits += bits[idx]
            i_psnr += psnrs[idx][0]
            i_ssim += ssims[idx][0]
            if lpips_list:
                i_lpips += lpips_list[idx]
            if dists_list:
                i_dists += dists_list[idx]
            i_num += 1
            if include_yuv:
                i_psnr_y += psnrs[idx][1]
                i_psnr_u += psnrs[idx][2]
                i_psnr_v += psnrs[idx][3]
                i_ssim_y += ssims[idx][1]
                i_ssim_u += ssims[idx][2]
                i_ssim_v += ssims[idx][3]
        else:
            p_bits += bits[idx]
            p_psnr += psnrs[idx][0]
            p_ssim += ssims[idx][0]
            if lpips_list:
                p_lpips += lpips_list[idx]
            if dists_list:
                p_dists += dists_list[idx]
            p_num += 1
            if include_yuv:
                p_psnr_y += psnrs[idx][1]
                p_psnr_u += psnrs[idx][2]
                p_psnr_v += psnrs[idx][3]
                p_ssim_y += ssims[idx][1]
                p_ssim_u += ssims[idx][2]
                p_ssim_v += ssims[idx][3]

    log_result = {}
    log_result['frame_pixel_num'] = frame_pixel_num
    log_result['i_frame_num'] = i_num
    log_result['p_frame_num'] = p_num
    log_result['ave_i_frame_bpp'] = i_bits / i_num / frame_pixel_num
    log_result['ave_i_frame_psnr'] = i_psnr / i_num
    log_result['ave_i_frame_msssim'] = i_ssim / i_num
    if lpips_list:
        log_result['ave_i_frame_lpips'] = i_lpips / i_num
    if dists_list:
        log_result['ave_i_frame_dists'] = i_dists / i_num
    if include_yuv:
        log_result['ave_i_frame_psnr_y'] = i_psnr_y / i_num
        log_result['ave_i_frame_psnr_u'] = i_psnr_u / i_num
        log_result['ave_i_frame_psnr_v'] = i_psnr_v / i_num
        log_result['ave_i_frame_msssim_y'] = i_ssim_y / i_num
        log_result['ave_i_frame_msssim_u'] = i_ssim_u / i_num
        log_result['ave_i_frame_msssim_v'] = i_ssim_v / i_num
    if verbose:
        log_result['frame_bpp'] = list(np.array(bits) / frame_pixel_num)
        log_result['frame_psnr'] = [v[0] for v in psnrs]
        log_result['frame_msssim'] = [v[0] for v in ssims]
        log_result['frame_lpips'] = [v for v in lpips_list]
        log_result['frame_dists'] = [v for v in dists_list]
        log_result['frame_type'] = frame_types
        if include_yuv:
            log_result['frame_psnr_y'] = [v[1] for v in psnrs]
            log_result['frame_psnr_u'] = [v[2] for v in psnrs]
            log_result['frame_psnr_v'] = [v[3] for v in psnrs]
            log_result['frame_msssim_y'] = [v[1] for v in ssims]
            log_result['frame_msssim_u'] = [v[2] for v in ssims]
            log_result['frame_msssim_v'] = [v[3] for v in ssims]
    log_result['test_time'] = test_time
    if p_num > 0:
        total_p_pixel_num = p_num * frame_pixel_num
        log_result['ave_p_frame_bpp'] = p_bits / total_p_pixel_num
        log_result['ave_p_frame_psnr'] = p_psnr / p_num
        log_result['ave_p_frame_msssim'] = p_ssim / p_num
        if lpips_list:
            log_result['ave_p_frame_lpips'] = p_lpips / p_num
        if dists_list:
            log_result['ave_p_frame_dists'] = p_dists / p_num
        if include_yuv:
            log_result['ave_p_frame_psnr_y'] = p_psnr_y / p_num
            log_result['ave_p_frame_psnr_u'] = p_psnr_u / p_num
            log_result['ave_p_frame_psnr_v'] = p_psnr_v / p_num
            log_result['ave_p_frame_msssim_y'] = p_ssim_y / p_num
            log_result['ave_p_frame_msssim_u'] = p_ssim_u / p_num
            log_result['ave_p_frame_msssim_v'] = p_ssim_v / p_num
    else:
        log_result['ave_p_frame_bpp'] = 0
        log_result['ave_p_frame_psnr'] = 0
        log_result['ave_p_frame_msssim'] = 0
        if include_yuv:
            log_result['ave_p_frame_psnr_y'] = 0
            log_result['ave_p_frame_psnr_u'] = 0
            log_result['ave_p_frame_psnr_v'] = 0
            log_result['ave_p_frame_msssim_y'] = 0
            log_result['ave_p_frame_msssim_u'] = 0
            log_result['ave_p_frame_msssim_v'] = 0
    log_result['ave_all_frame_bpp'] = (i_bits + p_bits) / (frame_num * frame_pixel_num)
    log_result['ave_all_frame_psnr'] = (i_psnr + p_psnr) / frame_num
    log_result['ave_all_frame_msssim'] = (i_ssim + p_ssim) / frame_num
    if lpips_list:
        log_result['ave_all_frame_lpips'] = (i_lpips + p_lpips) / frame_num
    if dists_list:
        log_result['ave_all_frame_dists'] = (i_dists + p_dists) / frame_num
    if include_yuv:
        log_result['ave_all_frame_psnr_y'] = (i_psnr_y + p_psnr_y) / frame_num
        log_result['ave_all_frame_psnr_u'] = (i_psnr_u + p_psnr_u) / frame_num
        log_result['ave_all_frame_psnr_v'] = (i_psnr_v + p_psnr_v) / frame_num
        log_result['ave_all_frame_msssim_y'] = (i_ssim_y + p_ssim_y) / frame_num
        log_result['ave_all_frame_msssim_u'] = (i_ssim_u + p_ssim_u) / frame_num
        log_result['ave_all_frame_msssim_v'] = (i_ssim_v + p_ssim_v) / frame_num

    return log_result


### Image I/O ###

class OnlyImageFolder(Dataset):
    def __init__(self, root_folder_path, padding_size):
        self.root_folder_path = root_folder_path
        self.images = sorted(list(os.listdir(root_folder_path)))
        self.dataset_length = len(self.images)
        self.padding_size = padding_size
        print(f"Datasets: {self.dataset_length} images are in {root_folder_path}")

    def __getitem__(self, index):
        # 1. load image
        image_path = os.path.join(self.root_folder_path, self.images[index])
        img = Image.open(image_path).convert("RGB")

        img = np.array(img).transpose(2, 0, 1)
        img = torch.as_tensor(img.astype(np.float32) / 255.0, dtype=torch.float32)

        return img, image_path

    def __len__(self):
        return self.dataset_length


def write_image(save_png_path, x_hat):
    out_frame = from_minus1_1_to_0_1(x_hat)
    out_frame = out_frame.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out_frame = np.clip(np.rint(out_frame * 255), 0, 255).astype(np.uint8)
    Image.fromarray(out_frame).save(save_png_path)


### Video I/O ###

class PNGReader():
    def __init__(self, src_path, width, height, start_num=1):
        super().__init__()
        self.src_path = src_path
        self.width = width
        self.height = height
        self.eof = False

        pngs = os.listdir(self.src_path)
        if 'im1.png' in pngs:
            self.padding = 1
        elif 'im00001.png' in pngs:
            self.padding = 5
        else:
            raise ValueError('unknown image naming convention; please specify')
        self.current_frame_index = start_num

    def read_one_frame(self):
        if self.eof:
            return None

        png_path = os.path.join(self.src_path, f"im{str(self.current_frame_index).zfill(self.padding)}.png")
        if not os.path.exists(png_path):
            self.eof = True
            return None

        rgb = Image.open(png_path).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        _, height, width = rgb.shape
        assert height == self.height
        assert width == self.width

        self.current_frame_index += 1
        return rgb

    def close(self):
        self.current_frame_index = 1

class PNGWriter():
    def __init__(self, dst_path, width, height):
        super().__init__()
        self.dst_path = dst_path
        self.width = width
        self.padding = 5
        self.current_frame_index = 1
        os.makedirs(dst_path, exist_ok=True)

    def write_one_frame(self, rgb):
        rgb = rgb.transpose(1, 2, 0)

        png_path = os.path.join(self.dst_path, f"im{str(self.current_frame_index).zfill(self.padding)}.png")
        img = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(png_path)

        self.current_frame_index += 1

    def close(self):
        self.current_frame_index = 1

