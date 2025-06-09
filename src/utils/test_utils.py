
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


def from_0_1_to_minus1_1(value):
    return (value - 0.5)*2.0

def from_minus1_1_to_0_1(value):
    return ((value/2.0) + 0.5)

def consume_prefix_in_state_dict_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            state_dict[new_key] = state_dict.pop(key)

def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - width - padding_left
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom

def get_state_dict(ckpt_path, need="g"):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    
    if need == "g":
        ckpt = ckpt["net_g"]
        consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
        return ckpt
    else:
        consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
        return ckpt

def init_func():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)


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

