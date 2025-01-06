import os
import pprint
import argparse
import yaml
import pickle

import torch.nn as nn
import torch
import torch.optim as optim

from dataset_utils import get_dataset, MultiInsDataset

from utils import avgFeature, GetProbCentroid, get_channel, enc_loss_plot

from torch.utils.data import  DataLoader
from tqdm import tqdm 
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()


# Config - Path
parser.add_argument('--dataset_root', type=str, default='../dataset',
                    help='Root directory of dataset.')
parser.add_argument('--output_root', type=str, default='output',
                    help='Root directory of training results.')

# Config - Hyperparameter
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size to train an encoder.')
parser.add_argument('--lr', type=float, default=0.03,
                    help='Learning rate to train an encoder.')
parser.add_argument('--SGD_momentum', type=float, default=0.9,
                    help='Momentum of SGD optimizer to train an encoder.')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='Weight of L2 regularization of SGD optimizer.')
parser.add_argument('--temperature', type=float, default=0.07,
                    help='Temperature for constrastive loss.')
parser.add_argument('--CMItemperature', type=float, default=0.07,
                    help='Temperature for constrastive loss.')
parser.add_argument('--CMIFactor', type=float, default=0.01,
                    help='Temperature for constrastive loss.')

parser.add_argument('--momentum', type=float, default=0.999,
                    help='Momentum of momentum encoder. m in Eq.(2). It is not the momentum of SGD optimizer.')
parser.add_argument('--shuffle_bn', action='store_true',
                    help='Turn on shuffled batch norm. See Section 3.3.')

# Config - Architecture
parser.add_argument('--feature_dim', type=int, default=128,
                    help='Output dimension of last fully connected layer in encoder.')
parser.add_argument('--channel_model', type=str, default='conv3',
                    help='channel model, ')
parser.add_argument('--num_keys', type=int, default=4096,
                    help='Size of dictionary of encoded keys.')
parser.add_argument('--NRepeat', type=int, default=8,
                    help='NRepeat.')


# Config - Setting
# parser.add_argument('--resize', type=int, default=84,
#                     help='Image is resized to this value.')
# parser.add_argument('--crop', type=int, default=64,
#                     help='Image is cropped to this value. This is the final size of image transformation.')
parser.add_argument('--max_epoch', type=int, default=400,
                    help='Maximum epoch to train an encoder.')
parser.add_argument('--plot_iter', type=int, default=1000,
                    help='Frequency of plot loss graph.')
parser.add_argument('--save_weight_epoch', type=int, default=50,
                    help='Frequency of saving weight.')
parser.add_argument('--num_workers', type=int, default=16,
                    help='Number of workers for data loader.')
parser.add_argument('--dataset', type=str, default='CIFAR100',
                    help='dataset')
# CIFAR100, CIFAR10, TinyImageNet

parser.add_argument('--resume', action='store_true',
                    help='Resume training.')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='Training is resumed at this epoch.')

config = parser.parse_args()

# Show config
print('\n======================= Training configuration =======================\n')
pprint.pprint(vars(config))
print('\n======================================================================\n')

# Make output directories
output_path = os.path.join(config.output_root, config.dataset, config.channel_model,
                           "{}_{}_{}_{}".format(config.temperature, config.CMItemperature, 
                                                config.lr, config.CMIFactor))
loss_path = os.path.join(output_path, 'loss')
weight_path = os.path.join(output_path, 'weight')
    
if not os.path.exists(loss_path):
    os.makedirs(loss_path)
if not os.path.exists(weight_path):
    os.makedirs(weight_path)

''' ######################## < Step 2 > Create instances ######################## '''

# Build dataloader
print('\n[1 / 3]. Build data loader. Depending on your environment, this may take several minutes..')

channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, dlen = \
    get_dataset(dataset =config.dataset, data_path= config.dataset_root, args = config)

# dloader = None
Multi_ins_dataset = MultiInsDataset(dataset=dst_train, dlen=dlen, NRepeat=config.NRepeat)
# Build models
dloader = DataLoader(Multi_ins_dataset, batch_size=config.batch_size, shuffle=True, 
                        pin_memory=True, num_workers=config.num_workers)

print('\n[2 / 3]. Build models.. ')
encoder = get_channel(model_name=config.channel_model, 
                      feat_dim = config.feature_dim, 
                      dataset=config.dataset)
encoder = nn.DataParallel(encoder).to(dev)

momentum_encoder = get_channel(model_name=config.channel_model,
                               feat_dim = config.feature_dim,
                               dataset=config.dataset)
momentum_encoder = nn.DataParallel(momentum_encoder).to(dev)

for param in momentum_encoder.parameters():
    param.requires_grad = False

loss_hist = []

# Optimizer
optimizer = optim.SGD(encoder.parameters(), 
                      lr=config.lr, 
                      momentum=config.SGD_momentum, 
                      weight_decay=config.weight_decay)
scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1/5, end_factor=1.0, total_iters=5, )
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.max_epoch)

scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                  schedulers=[scheduler1, scheduler2], milestones=[5])
# Loss function
crossentropy = nn.CrossEntropyLoss(reduction='none')


''' ######################## < Step 3 > Define methods ######################## '''

def momentum_step(m=1):
    '''
    Momentum step (Eq (2)).

    Args:
        - m (float): momentum value. 1) m = 0 -> copy parameter of encoder to key encoder
                                     2) m = 0.999 -> momentum update of key encoder
    '''
    params_q = encoder.state_dict()
    params_k = momentum_encoder.state_dict()
    
    dict_params_k = dict(params_k)
    
    for name in params_q:
        theta_k = dict_params_k[name]
        theta_q = params_q[name].data
        dict_params_k[name].data.copy_(m * theta_k + (1-m) * theta_q)

    momentum_encoder.load_state_dict(dict_params_k)


''' ######################## < Step 4 > Start training ######################## '''

# Initialize momentum_encoder with parameters of encoder.
momentum_step(m=0)


# Initialize queue.
print('\n[3 / 3]. Initializing a queue with %d keys.' % config.num_keys)
queue = []
with torch.no_grad():
    for i, (img, _) in enumerate(dloader):
        nb, nd, c, w, h=img.shape
   
        DifFactor = torch.rand(nb)[:,None,None,None,None]

        Noise = torch.randn(size=[nb, 1, c, w, h])
        # Noise = torch.randn_like(img)

        img = torch.sqrt(DifFactor)*img+torch.sqrt(1-DifFactor)*Noise

        img = torch.flatten(img, 0 ,1)
        key_feature = momentum_encoder(img.to(dev))
        
        # tmpres = .reshape(4,3,shape[1], shape[2])
        queue.append(avgFeature(key_feature, nb, nd))

        if i == (config.num_keys / config.batch_size) - 1:
            break
    queue = torch.cat(queue, dim=0)
    
# Training
print('\nStart training!')
epoch = 0 if not config.resume else config.start_epoch
total_iters = 0 if not config.resume else int(dlen / config.batch_size) * config.start_epoch


while(epoch < config.max_epoch):
    encoder.train()
    momentum_encoder.train()
    for i, (x_k, _) in enumerate(dloader):
        # Preprocess
        nb, nd, c, w, h=x_k.shape
        # breakpoint()
        DifFactor = torch.rand(nb)[:,None,None,None,None]
        # Noise = torch.randn_like(x_k)
        Noise = torch.randn(size=[nb, 1, c, w, h])
        x_k = torch.sqrt(DifFactor)*x_k+torch.sqrt(1-DifFactor)*Noise
        
        encoder.zero_grad()
        x_k = torch.flatten(x_k, 0 ,1)
                  
        x_k = x_k.to(dev)

        q_f = encoder(x_k) # q : (N, 128)
        k_f = momentum_encoder(x_k).detach() # k : (N, 128)
        q = avgFeature(q_f, nb, nd)
        k = avgFeature(k_f, nb, nd)

        l_pos = torch.sum(q * k, dim=1, keepdim=True) # (N, 1)

        # Negative sampling q & queue
        l_neg = torch.mm(q, queue.t()) # (N, 4096)

        # Logit and label
        logits = torch.cat([l_pos, l_neg], dim=1) # (N, 4097) witi label [0, 0, ..., 0]
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(dev)
        # Get loss and backprop
        ScaledLogits = logits / config.temperature
        loss_ = crossentropy(ScaledLogits, labels)
        loss  = torch.nan_to_num(loss_, nan=0.0, posinf=0.0, neginf=0.0).mean()
        
        Prob, Centroid = GetProbCentroid(q_f, config.CMItemperature, nb, nd)

        Log_Prob = torch.log(Prob)
        # breakpoint()
        Log_Centroid = torch.log(Centroid).detach()
        # CMI_ = torch.nn.functional.kl_div(Log_Prob, Log_Centroid,  reduction= 'batchmean', log_target=True)
        # CMI  = torch.nan_to_num(CMI_, nan=0.0, posinf=0.0, neginf=0.0)
        CMI_ = torch.nn.functional.kl_div(Log_Prob, Log_Centroid,  reduction= 'none', log_target=True)
        CMI  = torch.nan_to_num(CMI_, nan=0.0, posinf=0.0, neginf=0.0)
        # breakpoint()
        CMI = CMI.sum(-1).mean()
        # print(CMI)
        Totalloss = loss - config.CMIFactor*CMI
        # Totalloss = loss 
        Totalloss.backward()
        
        # Encoder update
        optimizer.step()

        # Momentum encoder update
        momentum_step(m=config.momentum)

        # Update dictionary
        queue = torch.cat([k, queue[:queue.size(0) - k.size(0)]], dim=0)
        
        # Print a training status, save a loss value, and plot a loss graph.
        total_iters = total_iters + 1
        print('[Epoch : %d / Total iters : %d] : loss : %f  CMI : %f ...' \
              %(epoch, total_iters, loss.item(), float(CMI)))
        
        loss_hist.append(Totalloss.item())


        if total_iters % config.plot_iter == 0:
            enc_loss_plot(loss_hist, loss_path, record_iter=1)
        
    epoch += 1
    
    # Update learning rate
    scheduler.step()
    # Save
    if epoch % config.save_weight_epoch == 0:
        path_ckpt = os.path.join(weight_path, 'ckpt_' + str(epoch-1) + '.pt')
        ckpt = {
            'encoder': encoder.state_dict(),
            'momentum_encoder': momentum_encoder.state_dict()
        }
        torch.save(ckpt, path_ckpt)
        with open(os.path.join(loss_path, 'loss.pkl'), 'wb') as f:
            pickle.dump(loss_hist, f)