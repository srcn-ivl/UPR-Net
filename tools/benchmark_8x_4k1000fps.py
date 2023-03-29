import cv2
import math
import numpy as np
import argparse
import warnings

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from core.pipeline import Pipeline
from core.dataset import X_Test
from core.utils.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")


def evaluate(ppl, test_data_path, batch_size, nr_data_worker=1):
    dataset = X_Test(test_data_path=test_data_path, multiple=8)
    val_data = DataLoader(dataset, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)

    psnr_list = []
    ssim_list = []
    nr_val = val_data.__len__()
    for i, (frames, t_value, scene_name, frame_range) in enumerate(val_data):
        torch.cuda.empty_cache()
        frames = frames.to(DEVICE, non_blocking=True) / 255.
        B, C, T, h, w = frames.size()
        t_value = t_value.to(DEVICE, non_blocking=True)
        img0 = frames[:, :, 0, :, :]
        img1 = frames[:, :, 1, :, :]
        gt = frames[:, :, 2, :, :]
        overlay_input = 0.5 * img0 + 0.5 * img1

        divisor = 256
        if (h % divisor != 0) or (w % divisor != 0):
            ph = ((h - 1) // divisor + 1) * divisor
            pw = ((w - 1) // divisor + 1) * divisor
            divisor = (0, pw - w, 0, ph - h)
            img0 = F.pad(img0, divisor, "constant", 0.5)
            img1 = F.pad(img1, divisor, "constant", 0.5)

        img0 = F.interpolate(img0, (ph, pw), mode="bilinear")
        img1 = F.interpolate(img1, (ph, pw), mode="bilinear")
        pred, _  = ppl.inference(img0, img1, time_period=t_value,
                pyr_level=PYR_LEVEL,
                nr_lvl_skipped=NR_LVL_SKIPPED)
        pred = pred[:, :, :h, :w]

        batch_psnr = []
        batch_ssim = []
        for j in range(gt.shape[0]):
            this_gt = gt[j]
            this_pred = pred[j]
            this_overlay = overlay_input[j]
            ssim = ssim_matlab(
                    this_pred.unsqueeze(0), this_gt.unsqueeze(0)
                    ).cpu().numpy()
            ssim = float(ssim)
            ssim_list.append(ssim)
            batch_ssim.append(ssim)
            psnr = -10 * math.log10(
                    torch.mean((this_gt - this_pred) * (this_gt - this_pred)
                        ).cpu().data)
            psnr_list.append(psnr)
            batch_psnr.append(psnr)

        print('batch: {}/{}; psnr: {:.4f}; ssim: {:.4f}'.format(i, nr_val,
            np.mean(batch_psnr), np.mean(batch_ssim)))

    psnr = np.array(psnr_list).mean()
    print('average psnr: {:.4f}'.format(psnr))
    ssim = np.array(ssim_list).mean()
    print('average ssim: {:.4f}'.format(ssim))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='benchmarking on 4k1000fps' +\
            'dataset for 8x multiple interpolation')

    #**********************************************************#
    # => args for data loader
    parser.add_argument('--test_data_path', type=str, required=True,
            help='the path of 4k10000fps benchmark')
    parser.add_argument('--nr_data_worker', type=int, default=1,
            help='number of the worker for data loader')
    parser.add_argument('--batch_size', type=int, default=1,
            help='batchsize for data loader')

    #**********************************************************#
    # => args for model
    parser.add_argument('--pyr_level', type=int, default=7,
            help='the number of pyramid levels of UPR-Net in testing')
    parser.add_argument('--nr_lvl_skipped', type=int, default=2,
            help='the number of skipped high-resolution pyramid levels '\
                    'of UPR-Net in testing')
    ## test base version of UPR-Net by default
    parser.add_argument('--model_size', type=str, default="base",
            help='model size, one of (base, large, LARGE)')
    parser.add_argument('--model_file', type=str,
            default="./checkpoints/upr-base.pkl",
            help='weight of UPR-Net')

    ## test large version of UPR-Net
    # parser.add_argument('--model_size', type=str, default="large",
    #         help='model size, one of (base, large, LARGE)')
    # parser.add_argument('--model_file', type=str,
    #         default="./checkpoints/upr-large.pkl",
    #         help='weight of UPR-Net')

    ## test LARGE version of UPR-Net
    # parser.add_argument('--model_size', type=str, default="LARGE",
    #         help='model size, one of (base, large, LARGE)')
    # parser.add_argument('--model_file', type=str,
    #         default="./checkpoints/upr-llarge.pkl",
    #         help='weight of UPR-Net')


    #**********************************************************#
    # => init the benchmarking environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.demo = True
    torch.backends.cudnn.benchmark = True

    #**********************************************************#
    # => init the pipeline and start to benchmark
    args = parser.parse_args()

    model_cfg_dict = dict(
            load_pretrain = True,
            model_size = args.model_size,
            model_file = args.model_file
            )
    ppl = Pipeline(model_cfg_dict)

    # resolution-aware parameter for inference
    PYR_LEVEL = args.pyr_level
    NR_LVL_SKIPPED = args.nr_lvl_skipped

    print("benchmarking on 4K1000FPS...")
    evaluate(ppl, args.test_data_path, args.batch_size, args.nr_data_worker)
