from channel_model import ConvNet
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torchvision.models as thmodels
from torchsummary import summary
import os


class Normlize(torch.nn.Module):
    def __init__(self):
        super(Normlize, self).__init__()
    def forward(self, x):
        norm = torch.norm(x, p='fro', dim=1, keepdim=True)
        return x/norm

def get_channel(model_name="conv3", feat_dim=128, dataset="CIFAR10", out_norm = True):
    if "conv" in model_name:
        if dataset in ["CIFAR10", "CIFAR100"]:
            size = 32
        elif dataset == "tinyimagenet":
            size = 64
        elif dataset in ["imagenet-nette", "imagenet-woof", "imagenet-100"]:
            size = 128
        else:
            size = 224
        model = ConvNet(
            net_norm="batch",
            net_act="relu",
            num_classes = feat_dim,
            net_pooling="avgpooling",
            net_depth=int(model_name[-1]),
            net_width=128,
            channel=3,
            im_size=(size, size),
        )
        if out_norm:
            model.fc = nn.Sequential(nn.Linear(in_features=model.fc.in_features,
                                               out_features=feat_dim, bias=False),
                                     Normlize())
        else:
            model.fc = nn.Linear(in_features=model.fc.in_features, 
                                 out_features=feat_dim, bias=False)

    elif model_name == "resnet18_modified":
        model = thmodels.__dict__["resnet18"](pretrained=False)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.maxpool = nn.Identity()

        if out_norm:
            model.fc = nn.Sequential(nn.Linear(in_features=model.fc.in_features,
                                               out_features=feat_dim, bias=False),
                                     Normlize())
        else:
            model.fc = nn.Linear(in_features=model.fc.in_features, 
                                 out_features=feat_dim, 
                                 bias=False)

    elif model_name == "resnet101_modified":
        model = thmodels.__dict__["resnet101"](pretrained=False)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.maxpool = nn.Identity()

        if out_norm:
            model.fc = nn.Sequential(nn.Linear(in_features=model.fc.in_features,
                                               out_features=feat_dim, bias=False),
                                     Normlize())
        else:
            model.fc = nn.Linear(in_features=model.fc.in_features, 
                                out_features=feat_dim, bias=False)
        
    else:
        
        if "resnet" not in model_name:
            exit("unknown model name: %s; Path: /Channel/utils.py -> get_model()" % model_name)
        else:
            model = thmodels.__dict__[model_name](pretrained=False)
            if out_norm:
                model.fc = nn.Sequential(nn.Linear(in_features=model.fc.in_features,
                                                out_features=feat_dim, bias=False),
                                        Normlize())
            else:
                model.fc = nn.Linear(in_features=model.fc.in_features, 
                        out_features=feat_dim, bias=False)
    return model



def avgFeature(feature, Nb, Nd):
    rev = feature.reshape(Nb, Nd,-1)
    return torch.mean(rev, 1)

def GetProbCentroid(feature, T, Nb, Nd):
    rev = feature.reshape(Nb, Nd,-1)/T
    rev = torch.softmax(rev, dim=-1)
    return rev, torch.mean(rev, 1)[:,None,:]

def enc_loss_plot(hist, path, record_iter):
    '''
    Plot a loss graph of encoder training.

    Args:
        - hist (list) : List consists of loss values.
        - path (str) : Directory to save loss graph.
        - record_iter (int) : Frequency of saving loss value in iterations. For example, if the loss value is 
                              saved in hist every ten iteration, record_iter should be ten.
    '''
    plt.switch_backend('agg')
    x = range(0, record_iter * len(hist), record_iter)
        
    plt.plot(x, hist, label='loss')
    
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, 'loss.png')
    plt.savefig(path)
    plt.close()

