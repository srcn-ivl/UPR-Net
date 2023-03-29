import os
import sys
import shutil
import cv2
import torch
import argparse
import numpy as np
import math
from importlib import import_module

from torch.nn import functional as F
from core.utils import flow_viz
from core.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_exp_env():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.demo = True


def interp_imgs(ppl, ori_img0, ori_img1):
    img0 = (torch.tensor(ori_img0.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)
    img1 = (torch.tensor(ori_img1.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    divisor = 2 ** (PYR_LEVEL-1+2)

    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding, "constant", 0.5)
        img1 = F.pad(img1, padding, "constant", 0.5)

    print("\nInitialization is OK! Begin to interp images...")

    interp_img, bi_flow = ppl.inference(img0, img1,
            time_period=TIME_PERIOID,
            pyr_level=PYR_LEVEL)
    interp_img = interp_img[:, :, :h, :w]
    bi_flow = bi_flow[:, :, :h, :w]

    overlay_input = (ori_img0 * 0.5 + ori_img1 * 0.5).astype("uint8")
    interp_img = (interp_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
    bi_flow = bi_flow[0].cpu().numpy().transpose(1, 2, 0)

    flow01 = bi_flow[:, :, :2]
    flow10 = bi_flow[:, :, 2:]
    flow01 = flow_viz.flow_to_image(flow01, convert_to_bgr=True)
    flow10 = flow_viz.flow_to_image(flow10, convert_to_bgr=True)
    bi_flow = np.concatenate([flow01, flow10], axis=1)

    cv2.imwrite(os.path.join(SAVE_DIR, '0-img0.png'), ori_img0)
    cv2.imwrite(os.path.join(SAVE_DIR, '1-img1.png'), ori_img1)
    cv2.imwrite(os.path.join(SAVE_DIR, '2-overlay-input.png'), overlay_input)
    cv2.imwrite(os.path.join(SAVE_DIR, '3-interp-img.png'), interp_img)
    cv2.imwrite(os.path.join(SAVE_DIR, '4-bi-flow.png'), bi_flow)

    print("\nInterpolation is completed! Please see the results in %s" % (SAVE_DIR))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="interpolate for given pair of images")
    parser.add_argument("--frame0", type=str, required=True,
            help="file path of the first input frame")
    parser.add_argument("--frame1", type=str, required=True,
            help="file path of the second input frame")
    parser.add_argument("--time_period", type=float, default=0.5,
            help="time period for interpolated frame")
    parser.add_argument("--save_dir", type=str,
            default="./demo/output",
            help="dir to save interpolated frame")

    ## load base version of UPR-Net by default
    parser.add_argument('--model_size', type=str, default="base",
            help='model size, one of (base, large, LARGE)')
    parser.add_argument('--model_file', type=str,
            default="./checkpoints/upr-base.pkl",
            help='weight of UPR-Net')

    ## load large version of UPR-Net
    # parser.add_argument('--model_size', type=str, default="large",
    #         help='model size, one of (base, large, LARGE)')
    # parser.add_argument('--model_file', type=str,
    #         default="./checkpoints/upr-large.pkl",
    #         help='weight of UPR-Net')

    ## load LARGE version of UPR-Net
    # parser.add_argument('--model_size', type=str, default="LARGE",
    #         help='model size, one of (base, large, LARGE)')
    # parser.add_argument('--model_file', type=str,
    #         default="./checkpoints/upr-llarge.pkl",
    #         help='weight of UPR-Net')

    args = parser.parse_args()

    #**********************************************************#
    # => parse args and init the training environment
    # global variable
    FRAME0 = args.frame0
    FRAME1 = args.frame1
    TIME_PERIOID = args.time_period
    SAVE_DIR = args.save_dir

    # init env
    init_exp_env()

    #**********************************************************#
    # => read input frames and calculate the number of pyramid levels
    ori_img0 = cv2.imread(FRAME0)
    ori_img1 = cv2.imread(FRAME1)
    if ori_img0.shape != ori_img1.shape:
        ValueError("Please ensure that the input frames have the same size!")
    width = ori_img0.shape[1]
    PYR_LEVEL = math.ceil(math.log2(width/448) + 3)

    #**********************************************************#
    # => init the pipeline and interpolate images
    model_cfg_dict = dict(
            load_pretrain = True,
            model_size = args.model_size,
            model_file = args.model_file
            )

    ppl = Pipeline(model_cfg_dict)
    ppl.eval()
    interp_imgs(ppl, ori_img0, ori_img1)
