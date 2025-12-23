# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.models.image_model import GLC_Image
from src.models.video_model import GLC_Video
from src.utils.test_utils import PNGReader, PNGWriter, str2bool, create_folder, generate_log_json, get_state_dict, dump_json, \
    from_0_1_to_minus1_1, from_minus1_1_to_0_1, set_torch_env
from src.utils.metrics_video import calc_psnr, calc_msssim_rgb


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--force_zero_thres', type=float, default=None, required=False)
    parser.add_argument('--model_path_i', type=str)
    parser.add_argument('--model_path_p',  type=str)
    parser.add_argument('--vqgan_pretrain_path',  type=str)
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--q_indexes_i', type=int, nargs="+")
    parser.add_argument('--q_indexes_p', type=int, nargs="+")
    parser.add_argument("--force_intra", type=str2bool, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument("--rate_gop_size", type=int, default=8, choices=[4, 8])
    parser.add_argument('--reset_interval', type=int, default=32, required=False)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--force_root_path', type=str, default=None, required=False)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument('--float16', type=str2bool, default=False)
    parser.add_argument("--cuda", type=str2bool, default=False)
    parser.add_argument('--cuda_idx', type=int, nargs="+", help='GPU indexes to use')
    parser.add_argument('--calc_ssim', type=str2bool, default=False, required=False)
    parser.add_argument('--check_existing', type=str2bool, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose_json', type=str2bool, default=False)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--hyperK', type=int, default=16)

    args = parser.parse_args()
    return args


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def get_src_frame(args, src_reader, device):
    assert args['src_type'] == 'png'
    rgb = src_reader.read_one_frame()
    x = np_image_to_tensor(rgb)

    if args['float16']:
        x = x.to(torch.float16)
    x = x.to(device)
    return x, rgb


def get_distortion(args, rgb_rec, rgb):
    psnr = calc_psnr(rgb, rgb_rec, data_range=1)
    if args['calc_ssim']:
        msssim = calc_msssim_rgb(rgb, rgb_rec, data_range=1)
    else:
        msssim = 0.
    curr_psnr = [psnr]
    curr_ssim = [msssim]
    return curr_psnr, curr_ssim


def run_one_point(p_frame_net, i_frame_net, lpips_metric, dists_metric, args):
    frame_num = args['frame_num']
    rate_gop_size = args['rate_gop_size']
    verbose = args['verbose']
    reset_interval = args['reset_interval']
    verbose_json = args['verbose_json']
    device = next(i_frame_net.parameters()).device
    save_decoded_frame = args['save_decoded_frame']

    frame_types = []
    psnrs = []
    msssims = []
    bits = []
    lpips_list = []
    dists_list = []
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    start_time = time.time()
    src_reader = PNGReader(args['src_path'], args['src_width'], args['src_height'])
    pic_height = args['src_height']
    pic_width = args['src_width']
    padding_l, padding_r, padding_t, padding_b = GLC_Image.get_padding_size(pic_height, pic_width, 64)

    if save_decoded_frame:
        recon_writer = PNGWriter(args['curr_rec_path'], args['src_width'], args['src_height'])
    with torch.no_grad():
        for frame_idx in range(frame_num):
            x, rgb = get_src_frame(args, src_reader, device)
            frame_start_time = time.time()

            x_padded = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")
            x_padded = from_0_1_to_minus1_1(x_padded)

            if frame_idx % args['intra_period'] == 0:    
                encoded = i_frame_net.test(x_padded, args['q_index_i'])
                dpb = {
                    "ref_frame": encoded['x_hat'],
                    "ref_latent": encoded['ref_latent'],
                    "ref_y": None
                }
                recon_frame = encoded['x_hat']

                frame_types.append(0)
                bits.append(encoded['bit'])
            else:
                if reset_interval > 0 and frame_idx % reset_interval == 1:
                    dpb["ref_y"] = None
                fa_idx = index_map[frame_idx % rate_gop_size]
                encoded = p_frame_net.test(x_padded, dpb, args['q_index_p'],
                                                        fa_idx=fa_idx)
                
                dpb = encoded['dpb']

                recon_frame = dpb["ref_frame"]
                frame_types.append(1)
                bits.append(encoded['bit'].item())

            # align input/output range
            recon_frame = recon_frame.clamp(-1, 1)
            recon_frame = from_minus1_1_to_0_1(recon_frame)

            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            frame_end_time = time.time()
        
            rgb_rec = x_hat.squeeze(0).cpu().numpy()
            curr_psnr, curr_ssim = get_distortion(args, rgb_rec, rgb)

            curr_dists = dists_metric(x, x_hat).item()
            curr_lpips = lpips_metric.forward(x * 2 - 1, x_hat * 2 - 1).item()
            psnrs.append(curr_psnr)
            msssims.append(curr_ssim)
            lpips_list.append(curr_lpips)
            dists_list.append(curr_dists)
            
            if save_decoded_frame:
                recon_writer.write_one_frame(rgb_rec)
            
            if verbose >= 2:
                print(f"frame {frame_idx}, {(frame_end_time - frame_start_time) * 1000:.3f} ms, "
                      f"bits: {bits[-1]:.3f}, PSNR: {psnrs[-1][0]:.4f}, "
                      f"MS-SSIM: {msssims[-1][0]:.4f} ")

    src_reader.close()
    if save_decoded_frame:
        recon_writer.close()
    test_time = time.time() - start_time

    log_result = generate_log_json(frame_num, pic_height * pic_width, test_time,
                                   frame_types, bits, psnrs, msssims, lpips_list=lpips_list, dists_list=dists_list, verbose=verbose_json)
    with open(args['curr_json_path'], 'w') as fp:
        json.dump(log_result, fp, indent=2)
    return log_result


i_frame_net = None  # the model is initialized after each process is spawn, thus OK for multiprocess
p_frame_net = None
lpips_metric = None
dists_metric = None


def worker(args):
    global i_frame_net
    global p_frame_net
    global lpips_metric
    global dists_metric

    sub_dir_name = args['seq']
    bin_folder = os.path.join(args['stream_path'], args['ds_name'])

    args['src_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['bin_folder'] = bin_folder
    args['curr_bin_path'] = os.path.join(bin_folder,
                                         f"{args['seq']}_q{args['q_index_i']}.bin")
    args['curr_rec_path'] = args['curr_bin_path'].replace('.bin', '')
    args['curr_json_path'] = args['curr_bin_path'].replace('.bin', '.json')

    result = run_one_point(p_frame_net, i_frame_net, lpips_metric, dists_metric, args)

    result['ds_name'] = args['ds_name']
    result['seq'] = args['seq']
    result['rate_idx'] = args['rate_idx']
    result['q_index_i'] = args['q_index_i']
    result['q_index_p'] = args['q_index_p'] if 'q_index_p' in args else args['q_index_i']

    return result


def init_func(args, gpu_num):
    set_torch_env()
    
    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        if args.cuda_idx is not None:
            gpu_id = args.cuda_idx[gpu_id]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = "cuda:0"
    else:
        device = "cpu"

    global i_frame_net
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net = GLC_Image()
    i_frame_net.load_state_dict(i_state_dict, strict=True)
    i_frame_net.interpolate_q()
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    global p_frame_net
    p_state_dict = get_state_dict(args.model_path_p)
    p_frame_net = GLC_Video(hyper_K=args.hyperK)
    p_frame_net.load_state_dict(p_state_dict, strict=False)
    p_frame_net = p_frame_net.to(device)
    p_frame_net.eval()

    from src.utils.lpips import LPIPS
    from src.utils.DISTS_pytorch import DISTS

    global lpips_metric
    lpips_metric = LPIPS(net='alex',version='0.1')
    lpips_metric = lpips_metric.to(device)
    lpips_metric.eval()

    global dists_metric 
    dists_metric = DISTS().to(device)

    if args.float16:
        if p_frame_net is not None:
            p_frame_net.half()
        i_frame_net.half()


def main():
    begin_time = time.time()

    args = parse_args()

    if args.force_zero_thres is not None and args.force_zero_thres < 0:
        args.force_zero_thres = None

    if args.cuda_idx is not None:
        cuda_device = ','.join([str(s) for s in args.cuda_idx])
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    gpu_num = 0
    if args.cuda:
        gpu_num = torch.cuda.device_count()

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num,
                                                                 initializer=init_func,
                                                                 initargs=(args, gpu_num))
    objs = []

    count_frames = 0
    count_sequences = 0

    rate_num = args.rate_num
    q_indexes_i = []
    if args.q_indexes_i is not None:
        assert len(args.q_indexes_i) == rate_num
        q_indexes_i = args.q_indexes_i
    else:
        assert 2 <= rate_num <= GLC_Video.get_qp_num()
        for i in np.linspace(0, GLC_Video.get_qp_num() - 1, num=rate_num):
            q_indexes_i.append(int(i+0.5))

    if not args.force_intra:
        if args.q_indexes_p is not None:
            assert len(args.q_indexes_p) == rate_num
            q_indexes_p = args.q_indexes_p
        else:
            q_indexes_p = q_indexes_i

    print(f"testing {rate_num} rates, using q_indexes: ", end='')
    for q in q_indexes_i:
        print(f"{q}, ", end='')
    print()

    root_path = args.force_root_path if args.force_root_path is not None else config['root_path']
    config = config['test_classes']
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq in config[ds_name]['sequences']:
            count_sequences += 1
            for rate_idx in range(rate_num):
                cur_args = {}
                cur_args['rate_idx'] = rate_idx
                cur_args['float16'] = args.float16
                cur_args['hyperK'] = args.hyperK
                cur_args['q_index_i'] = q_indexes_i[rate_idx]
                if not args.force_intra:
                    cur_args['q_index_p'] = q_indexes_p[rate_idx]
                cur_args['force_intra'] = args.force_intra
                cur_args['reset_interval'] = args.reset_interval
                cur_args['seq'] = seq
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq]['width']
                cur_args['intra_period'] = config[ds_name]['sequences'][seq]['intra_period']
                if args.force_intra:
                    cur_args['intra_period'] = 1
                elif args.force_intra_period > 0:
                    cur_args['intra_period'] = args.force_intra_period
                cur_args['frame_num'] = config[ds_name]['sequences'][seq]['frames']
                if args.force_frame_num > 0:
                    cur_args['frame_num'] = args.force_frame_num
                cur_args['rate_gop_size'] = args.rate_gop_size
                cur_args['calc_ssim'] = args.calc_ssim
                cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['visual_path'] = os.path.join(args.output_path.replace('.json', '/'))
                cur_args['check_existing'] = args.check_existing
                cur_args['stream_path'] = args.stream_path
                cur_args['save_decoded_frame'] = args.save_decoded_frame
                cur_args['ds_name'] = ds_name
                cur_args['verbose'] = args.verbose
                cur_args['verbose_json'] = args.verbose_json

                count_frames += cur_args['frame_num']

                obj = threadpool_executor.submit(worker, cur_args)
                objs.append(obj)

    results = []
    for obj in tqdm(objs):
        result = obj.result()
        results.append(result)

    log_result = {}
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}

    for res in results:
        log_result[res['ds_name']][res['seq']][f"{res['rate_idx']:03d}"] = res

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()
