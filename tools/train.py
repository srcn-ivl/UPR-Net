import os
import sys
import shutil
import cv2
import math
import time
import numpy as np
import random
import argparse
import warnings
from importlib import import_module
from distutils.util import strtobool
from loguru import logger
from typing import Dict, Tuple

import torch
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from core.pipeline import Pipeline
from core.dataset import VimeoDataset

warnings.filterwarnings("ignore")


def get_learning_rate(total_step, cur_step, init_lr, min_lr=1e-6):
    if cur_step < 2000:
        mul = cur_step / 2000.
        return init_lr * mul
    else:
        mul = np.cos((cur_step - 2000) / (total_step - 2000.) * math.pi)\
                * 0.5 + 0.5
        return  (init_lr - min_lr) * mul + min_lr


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= (0.5 * (normalized_flow_map[:, :, 0]\
            + normalized_flow_map[:, :, 1]))
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def train(ppl, dataset_cfg_dict, optimizer_cfg_dict):
    model_log_dir = os.path.join(THIS_EXP_LOG_DIR, "trained-models")
    tf_log_dir = os.path.join(THIS_EXP_LOG_DIR, "tensorboard")

    # dataset config
    batch_size = dataset_cfg_dict.get("batch_size", 32)
    nr_data_worker = dataset_cfg_dict.get("nr_data_worker", 4)
    crop_h = dataset_cfg_dict.get("crop_h", 256)
    crop_w = dataset_cfg_dict.get("crop_w", 256)

    data_root = dataset_cfg_dict["data_root"]
    dataset_train = VimeoDataset(dataset_name='train', data_root=data_root)
    dataset_val = VimeoDataset(dataset_name='validation', data_root=data_root)

    sampler = DistributedSampler(dataset_train)
    train_data = DataLoader(dataset_train, batch_size=batch_size,
            num_workers=nr_data_worker,
            pin_memory=True, drop_last=True, sampler=sampler)
    val_data = DataLoader(dataset_val, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)

    # optimizer config
    total_step = int(optimizer_cfg_dict["steps"])
    save_interval = int(optimizer_cfg_dict["save_interval"])
    init_lr = optimizer_cfg_dict.get("init_lr", 2e-5)
    min_lr = optimizer_cfg_dict.get("min_lr", 2e-6)
    loss_type = optimizer_cfg_dict.get("loss_type", "l2+census")

    step = 1
    if RESUME:
        optimizer_ckpt_file = optimizer_cfg_dict["ckpt_file"]
        info_dict = torch.load(optimizer_ckpt_file)
        step = info_dict["step"] + 1
        if step > total_step:
            raise ValueError(
                    "Previous trained steps have exceeded the total step of"\
                    "this experiment!  Please check the value of current step"\
                    "and total step.")

    if LOCAL_RANK == 0:
        writer = SummaryWriter(tf_log_dir + '/train')
        writer_val = SummaryWriter(tf_log_dir + '/validate')
        is_write = False if RESUME else True
        if not RESUME:
            evaluate(ppl, step, val_data, writer_val, is_write)
            ppl.save_model(model_log_dir, LOCAL_RANK)

    time_stamp = time.time()
    step_per_epoch = len(train_data)
    epoch_counter = 0
    last_epoch = False

    while step <  total_step+1:
        if step + step_per_epoch >  total_step:
            last_epoch = True

        epoch_counter += 1
        sampler.set_epoch(epoch_counter)

        for data in train_data:
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu = data.to(
                    DEVICE, dtype=torch.float, non_blocking=True) / 255.

            img0 = data_gpu[:, :3]
            img1 = data_gpu[:, 3:6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(total_step, step, init_lr, min_lr)
            pred, extra_dict = ppl.train_one_iter(
                    img0, img1, gt,
                    learning_rate=learning_rate,
                    loss_type=loss_type)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            if step % 100 == 1 and LOCAL_RANK == 0:
                writer.add_scalar(
                        '1-loss_interp_l2', extra_dict["loss_interp_l2"] , step)
                writer.add_scalar('2-learning_rate', learning_rate, step)
            if step % 1000 == 1 and LOCAL_RANK == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255)\
                        .astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255)\
                        .astype('uint8')
                overlay = 0.5 * img0 + 0.5 * img1
                overlay = (overlay.permute(0, 2, 3, 1).detach().cpu().numpy()\
                        * 255).astype('uint8')
                bi_flow = extra_dict["bi_flow"]
                bi_flow = bi_flow.permute(0, 2, 3, 1).detach().cpu().numpy()
                nr_show = min(6, batch_size)
                for i in range(nr_show):
                    imgs = np.concatenate((overlay[i], gt[i], pred[i]), 1)\
                            [:, :, ::-1]
                    writer.add_image(
                            str(i) + '/0-overlay-gt-pred',
                            imgs, step, dataformats='HWC')
                    writer.add_image(
                            str(i) + '/1-flow_01_pred',
                            flow2rgb(bi_flow[i][:, :, :2]),
                            step, dataformats='HWC')
                writer.flush()
            if LOCAL_RANK == 0:
                print("{} => train step: {}/{}; time: {:.2f}+{:.2f}; "\
                        "loss_interp_l2: {:.4e}".format(
                            EXP_NAME, step, total_step,
                            data_time_interval, train_time_interval,
                            extra_dict["loss_interp_l2"]))

            if (LOCAL_RANK == 0) and (step % save_interval == 0):
                psnr = evaluate(ppl, step, val_data, writer_val)
                ppl.save_model(model_log_dir, LOCAL_RANK)
                ppl.save_optimizer_state(THIS_EXP_LOG_DIR, LOCAL_RANK, step)
                logger.info("{} => val step: {}; "\
                        "psnr: {:.4f}".format(EXP_NAME, step, psnr))
                if step % (save_interval * 50) == 0:
                    ppl.save_model(model_log_dir, LOCAL_RANK, save_step=step)
                    ppl.save_optimizer_state(THIS_EXP_LOG_DIR, LOCAL_RANK, step)

            step += 1
            if last_epoch and step == total_step + 1:
                break

        dist.barrier()


def evaluate(ppl, step, val_data, writer_val, is_write=True):
    psnr_list = []
    start_time_stamp = time.time()
    time_stamp = start_time_stamp
    nr_val = val_data.__len__()
    for i, data in enumerate(val_data):
        data_time_interval = time.time() - time_stamp
        time_stamp = time.time()
        data_gpu = data[0] if isinstance(data, list) else data
        data_gpu = data_gpu.to(
                DEVICE, dtype=torch.float, non_blocking=True) / 255.

        img0 = data_gpu[:, :3]
        img1 = data_gpu[:, 3:6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, extra_dict = ppl.inference(img0, img1)
        for j in range(gt.shape[0]):
            this_gt = gt[j]
            this_pred = pred[j]
            psnr = -10 * math.log10(
                    torch.mean((this_gt - this_pred) * (this_gt - this_pred))\
                            .cpu().data)
            psnr_list.append(psnr)
        eval_time_interval = time.time() - time_stamp
        time_stamp = time.time()
        if LOCAL_RANK == 0:
            print('{} => val step: {}: {}/{}; time: {:.2f}+{:.2f}'\
                    .format(EXP_NAME, step, i, nr_val,\
                    data_time_interval, eval_time_interval))

    if LOCAL_RANK == 0:
        print('eval time: {}'.format(time.time() - start_time_stamp))
        psnr = np.array(psnr_list).mean()
        if is_write:
            writer_val.add_scalar('0-psnr', psnr, step)
        logger.info('{} => val step: {}; psnr: {:.4f}'.format(
            EXP_NAME, step, psnr))

    return psnr


def init_exp_env():
    def prompt(query):
        sys.stdout.write("%s [y/n]:" % query)
        val = input()

        try:
            ret = strtobool(val)
        except ValueError:
            sys.stdout("please answer with y/n")
            return prompt(query)
        return ret

    # process the path
    if (LOCAL_RANK == 0) and (not RESUME):
        # init train log dir, model and tf dir
        if os.path.exists(THIS_EXP_LOG_DIR):
            while True:
                if prompt("Would you like to re-write"\
                        "the existing experimental saving dir?") == True:
                    shutil.rmtree(THIS_EXP_LOG_DIR)
                    break
                else:
                    print("Exit the program."\
                            "Please assign another expriment name!")
                    exit()

        train_log_dir_link = os.path.join(THIS_CODEBASE_DIR, "train-log")
        this_exp_model_dir = os.path.join(THIS_EXP_LOG_DIR, "trained-models")
        this_exp_tf_dir = os.path.join(THIS_EXP_LOG_DIR, "tensorboard")
        if not os.path.exists(TRAIN_LOG_ROOT):
            os.makedirs(TRAIN_LOG_ROOT)
        if not os.path.exists(train_log_dir_link):
            cmd = "ln -s %s %s" % (TRAIN_LOG_ROOT, train_log_dir_link)
            os.system(cmd)
        os.makedirs(THIS_EXP_LOG_DIR)
        os.makedirs(this_exp_model_dir)
        os.makedirs(this_exp_tf_dir)

    # set logger file
    if LOCAL_RANK == 0:
        logger.add(os.path.join(THIS_EXP_LOG_DIR, "runtime.log"))

    # init cuda env
    torch.distributed.init_process_group(backend="nccl", world_size=WORLD_SIZE)
    torch.cuda.set_device(LOCAL_RANK)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='train upr-net for video frame interpolation')

    # => args for basic information
    parser.add_argument('--exp_name', default="upr-base", type=str,
            help='experiment name, will be used to save all generated files')
    parser.add_argument('--train_log_root', default="../upr-train-log", type=str,
            help='root dir to save all training logs')
    parser.add_argument('--resume', default=False, type=bool,
            help='resume from previously saved experiment logs')

    #**********************************************************#
    # => args for distributed training
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')

    #**********************************************************#
    # => args for data loader and rand crop size
    parser.add_argument('--data_root', type=str, required=True,
            help='root dir of vimeo_triplet')
    parser.add_argument('--batch_size', type=int, default=8,
            help='batch size for data loader')
    parser.add_argument('--nr_data_worker', type=int, default=2,
            help='number of the worker for data loader')
    parser.add_argument('--crop_h', type=int, default=256,
            help='height of cropped patch')
    parser.add_argument('--crop_w', type=int, default=256,
            help='width of cropped patch')

    #**********************************************************#
    # => args for model
    parser.add_argument('--model_size', type=str, default="base",
            help='model size, one of (base, large, LARGE)')
    parser.add_argument('--pyr_level', type=int, default=3,
            help='the number of pyramid levels of UPR-Net during training')
    # parser.add_argument('--nr_lvl_skipped', type=int, default=0,
    #         help='the number of skipped high-resolution levels for 4K input')
    parser.add_argument('--load_pretrain', type=bool, default=False,
            help='whether load pre-trained weight')
    parser.add_argument('--model_file', type=str, default="",
            help='weight of UPR-Net')

    #**********************************************************#
    # => args for optimizer
    parser.add_argument('--init_lr', type=float, default=2e-4,
            help='init learning rate')
    parser.add_argument('--min_lr', type=float, default=2e-5,
            help='min learning rate, till the end of training')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
            help='wegith decay')
    parser.add_argument('--steps', type=float, default=0.8e6,
            help='total steps (iteration) for training')
    parser.add_argument('--save_interval', type=float, default=0.2e4,
            help='iteration interval to save model')
    parser.add_argument('--loss_type', type=str, default="l2+census",
            help='training loss')

    #**********************************************************#
    # => organize args in groups
    args = parser.parse_args()

    model_cfg_dict = dict(
            model_size = args.model_size,
            pyr_level = args.pyr_level,
            load_pretrain = args.load_pretrain,
            model_file = args.model_file
            )

    optimizer_cfg_dict = dict(
            init_lr=args.init_lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            steps=args.steps,
            save_interval=args.save_interval,
            loss_type=args.loss_type
            )

    dataset_cfg_dict = dict(
            nr_data_worker=args.nr_data_worker,
            batch_size=args.batch_size,
            crop_h=args.crop_h,
            crop_w=args.crop_w,
            data_root=args.data_root
            )


    #**********************************************************#
    # => parse args and init the training environment
    # global variable
    EXP_NAME = args.exp_name
    TRAIN_LOG_ROOT = args.train_log_root
    LOCAL_RANK = args.local_rank
    WORLD_SIZE = args.world_size
    DEVICE = torch.device("cuda", LOCAL_RANK)
    THIS_CODEBASE_DIR = os.path.split(os.path.split(__file__)[0])[0]
    THIS_EXP_LOG_DIR =os.path.join(TRAIN_LOG_ROOT, EXP_NAME)

    optimizer_cfg_dict["ckpt_file"] = os.path.join(THIS_EXP_LOG_DIR, "optimizer-ckpt.pth")
    if not os.path.exists(optimizer_cfg_dict["ckpt_file"]):
        args.resume = False
    if args.resume:
        model_cfg_dict["load_pretrain"] = True
        model_cfg_dict["model_file"] = os.path.join(
                THIS_EXP_LOG_DIR, "trained-models", "model.pkl")
    RESUME = args.resume

    # init the exp environment
    init_exp_env()

    #**********************************************************#
    # => init the pipeline and train the pipeline
    ppl = Pipeline(
            model_cfg_dict, optimizer_cfg_dict,
            LOCAL_RANK, training=True, resume=RESUME)
    logger.info("start the training task: %s" % EXP_NAME)
    train(ppl, dataset_cfg_dict, optimizer_cfg_dict)
