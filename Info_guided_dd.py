import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.utils import save_image
from Utils.diffusion_cifar_model import UNet
from Utils.diffusion_cifar_utils import InfoGuidedGaussianDiffusionSampler

from Channel import get_channel

def InfoGuidedDD(modelConfig: Dict, args):
    # load model and evaluate
    device = torch.device(modelConfig["device"])
    DDPMmodel = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
    ckpt = torch.load(os.path.join(
        modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
    DDPMmodel.load_state_dict(ckpt)
    print("model load weight done.")
    DDPMmodel.eval()

    sampler = InfoGuidedGaussianDiffusionSampler(DDPMmodel=DDPMmodel, model=None, classifier=None, 
                                                 beta_1=modelConfig["beta_1"], beta_T=modelConfig["beta_T"], T=modelConfig["T"], 
                                                 Scmi=0.5, Sce=0.5, KernelSize=32, UnfoldStride=1).to(device)
    # Sampled from standard normal distribution
    noisyImage = torch.randn(
        size=[modelConfig["batch_size"], 3, 32, 32], device=device)
    saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
    # save_image(saveNoisy, os.path.join(
    #     modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
    sampledImgs = sampler(noisyImage)
    sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]


def main(args):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 2000,
        "batch_size": 256,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./DiffusionCkpt/Cifar10DDPM/", # or /CheckpointsCifar100/
        "test_load_weight": "ckpt_1999_.pt",
        "sampled_dir": "./SampledImgsCifar10/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,
        "Dataset": "Cifar10" # or Cifar100
        }
    InfoGuidedDD(modelConfig)


if __name__ == '__main__':

    main(args)
