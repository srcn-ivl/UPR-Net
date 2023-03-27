import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from importlib import import_module
from torch.optim import AdamW
import torch.optim as optim
import itertools
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from .loss import EPE, Ternary, LapLoss

from core.models.upr_base import Model as base_model
from core.models.upr_large import Model as large_model
from core.models.upr_llarge import Model as LARGE_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Pipeline:
    def __init__(self,
            model_cfg_dict,
            optimizer_cfg_dict=None,
            local_rank=-1,
            training=False,
            resume=False
            ):
        self.model_cfg_dict = model_cfg_dict
        self.optimizer_cfg_dict = optimizer_cfg_dict
        self.epe = EPE()
        self.ter = Ternary()
        self.laploss = LapLoss()

        self.init_model()
        self.device()
        self.training = training

        # We note that in practical, the `lr` of AdamW is reset from the
        # outside, using cosine annealing during the while training process.
        if training:
            self.optimG = AdamW(itertools.chain(
                filter(lambda p: p.requires_grad, self.model.parameters())),
                lr=optimizer_cfg_dict["init_lr"],
                weight_decay=optimizer_cfg_dict["weight_decay"])

        # `local_rank == -1` is used for testing, which does not need DDP
        if local_rank != -1:
            self.model = DDP(self.model, device_ids=[local_rank],
                    output_device=local_rank, find_unused_parameters=False)

        # Restart the experiment from last saved model, by loading the state of
        # the optimizer
        if resume:
            assert training, "To restart the training, please init the"\
                    "pipeline with training mode!"
            print("Load optimizer state to restart the experiment")
            ckpt_dict = torch.load(optimizer_cfg_dict["ckpt_file"])
            self.optimG.load_state_dict(ckpt_dict["optimizer"])


    def train(self):
        self.model.train()


    def eval(self):
        self.model.eval()


    def device(self):
        self.model.to(DEVICE)


    @staticmethod
    def convert_state_dict(rand_state_dict, pretrained_state_dict):
        param =  {
            k.replace("module.", "", 1): v
            for k, v in pretrained_state_dict.items()
            }
        param = {k: v
                for k, v in param.items()
                if ((k in rand_state_dict) and (rand_state_dict[k].shape \
                        == param[k].shape))
                }
        rand_state_dict.update(param)
        return rand_state_dict


    def init_model(self):

        def load_pretrained_state_dict(model, model_file):
            if (model_file == "") or (not os.path.exists(model_file)):
                raise ValueError(
                        "Please set the correct path for pretrained model!")

            print("Load pretrained model from %s."  % model_file)
            rand_state_dict = model.state_dict()
            pretrained_state_dict = torch.load(model_file)

            return Pipeline.convert_state_dict(
                    rand_state_dict, pretrained_state_dict)

        # check args
        model_cfg_dict = self.model_cfg_dict
        model_size = model_cfg_dict["model_size"] \
                if "model_size" in model_cfg_dict else "base"
        pyr_level = model_cfg_dict["pyr_level"] \
                if "pyr_level" in model_cfg_dict else 3
        nr_lvl_skipped = model_cfg_dict["nr_lvl_skipped"] \
                if "nr_lvl_skipped" in model_cfg_dict else 0
        load_pretrain = model_cfg_dict["load_pretrain"] \
                if "load_pretrain" in model_cfg_dict else False
        model_file = model_cfg_dict["model_file"] \
                if "model_file" in model_cfg_dict else ""

        # instantiate model
        if model_size == "LARGE":
            self.model = LARGE_model(pyr_level, nr_lvl_skipped)
        elif model_size == "large":
            self.model = large_model(pyr_level, nr_lvl_skipped)
        else:
            self.model = base_model(pyr_level, nr_lvl_skipped)

        # load pretrained model weight
        if load_pretrain:
            state_dict = load_pretrained_state_dict(
                    self.model, model_file)
            self.model.load_state_dict(state_dict)
        else:
            print("Train from random initialization.")


    def save_optimizer_state(self, path, rank, step):
        if rank == 0:
            optimizer_ckpt = {
                     "optimizer": self.optimG.state_dict(),
                     "step": step
                     }
            torch.save(optimizer_ckpt, "{}/optimizer-ckpt.pth".format(path))


    def save_model(self, path, rank, save_step=None):
        if (rank == 0) and (save_step is None):
            torch.save(self.model.state_dict(), '{}/model.pkl'.format(path))
        if (rank == 0) and (save_step is not None):
            torch.save(self.model.state_dict(), '{}/model-{}.pkl'\
                    .format(path, save_step))

    def inference(self, img0, img1,
            time_period=0.5,
            pyr_level=3,
            nr_lvl_skipped=0):
        interp_img, bi_flow, _ = self.model(img0, img1,
                time_period=time_period,
                pyr_level=pyr_level,
                nr_lvl_skipped=nr_lvl_skipped)
        return interp_img, bi_flow


    def train_one_iter(self, img0, img1, gt, learning_rate=0,
            bi_flow_gt=None, time_period=0.5, loss_type="l2+census"):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        self.train()

        interp_img, bi_flow, info_dict = self.model(img0, img1, time_period)

        with torch.no_grad():
            loss_interp_l2_nograd = (((interp_img - gt) ** 2 + 1e-6) ** 0.5)\
                    .mean()

        loss_G = 0
        if loss_type == "l1":
            loss_G = (interp_img - gt).abs().mean()
        elif loss_type == "l2":
            loss_interp_l2 = (((interp_img - gt) ** 2 + 1e-6) ** 0.5).mean()
            loss_G = loss_interp_l2
        elif loss_type == "l2+census":
            loss_interp_l2 = (((interp_img - gt) ** 2 + 1e-6) ** 0.5).mean()
            loss_ter = self.ter(interp_img, gt).mean()
            loss_G = loss_interp_l2 + loss_ter
        else:
            ValueError("unsupported loss type!")


        self.optimG.zero_grad()
        loss_G.backward()
        self.optimG.step()

        extra_dict = {}
        extra_dict["loss_interp_l2"] = loss_interp_l2_nograd
        extra_dict["bi_flow"] = bi_flow

        return interp_img, extra_dict



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pass
