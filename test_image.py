# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.models.image_model import GLC_Image
from src.utils.test_utils import init_func, get_state_dict, from_0_1_to_minus1_1, write_image, OnlyImageFolder
from src.utils.metric_image import evaluate_quality


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--q_indexes', type=int, nargs="+", required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--fid_patch_size', type=int, default=64)   # 64 for kodak and 256 for high-resolution datasets e.g. CLIC2020 and Div2K

    args = parser.parse_args()
    return args


def main():
    # settings
    init_func()
    args = parse_args()

    i_state_dict = get_state_dict(args.model_path)
    i_frame_net = GLC_Image(inplace=True)
    i_frame_net.load_state_dict(i_state_dict, strict=True)
    i_frame_net = i_frame_net.to("cuda")
    i_frame_net.eval()
    padding_size = 64

    device = next(i_frame_net.parameters()).device
    
    # dataset
    eval_dataset = OnlyImageFolder(args.input_path, padding_size=padding_size)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=24,
        shuffle=False,
        pin_memory=False,
    )
    for q in args.q_indexes:
        save_path = f"{args.output_path}/q{q}/"
        os.makedirs(save_path, exist_ok=True)

        all_bpps = []
        for idx, (img, image_path) in enumerate(eval_dataloader):
            x = from_0_1_to_minus1_1(img.to(device))

            pic_height = x.shape[2]
            pic_width = x.shape[3]

            # pad if necessary
            padding_l, padding_r, padding_t, padding_b = GLC_Image.get_padding_size(pic_height, pic_width, padding_size)
            x_padded = torch.nn.functional.pad(
                x,
                (padding_l, padding_r, padding_t, padding_b),
                mode="replicate",
            )

            # inference
            with torch.no_grad():
                result = i_frame_net.test(x_padded, q)
            recon_frame = result["x_hat"].clamp(-1, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            bpp = result["bit"] / pic_height / pic_width
            bpp_y = result["bit_y"] / pic_height / pic_width
            bpp_z = result["bit_z"] / pic_height / pic_width

            # save image
            all_bpps.append(bpp)
            basename = os.path.splitext(os.path.basename(image_path[0]))[0]
            write_image(f"{save_path}/{basename}.png", x_hat)
            print(f"[qp={q} {idx}/{len(eval_dataloader)} {basename}] : bpp={bpp:.4f}, bpp_y={bpp_y:.4f}, bpp_z={bpp_z:.4f}")

        print(f" Average : qp={q} : bpp={sum(all_bpps) / len(all_bpps):.4f}")

        evaluate_quality(all_bpps, input_path=args.input_path, output_path=save_path, log_path=save_path, patch_size=args.fid_patch_size)


if __name__ == "__main__":
    main()
