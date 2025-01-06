import os
import argparse
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_channel
from torch.utils.data import  DataLoader
from classifier_model import MLP
from dataset_utils import get_dataset
from tqdm import tqdm
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print("train the model on {}".format(dev))

parser = argparse.ArgumentParser()

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
parser.add_argument('--load_pretrained_epoch', type=int, default=399,
                    help='load_pretrained_epoch.')


parser.add_argument('--channel_ckpt_root', type=str, default='ChannelCkpt',
                    help='Root directory of training results.')

parser.add_argument('--feature_dim', type=int, default=128,
                    help='Output dimension of last fully connected layer in encoder.')
parser.add_argument('--channel_model', type=str, default='conv3',
                    help='channel model, ')

parser.add_argument('--classifier_input_dim', type=int, default=128,
                    help='Output dimension of last fully connected layer in encoder.')
parser.add_argument('--classifier_latent_dim', type=int, default=256,
                    help='Output dimension of last fully connected layer in encoder.')
parser.add_argument('--classifier_depth', type=int, default=3,
                    help='Output dimension of last fully connected layer in encoder.')
parser.add_argument('--classifier_Norm', type=str, default=None,
                    help='BatchNorm, LayerNorm.')

parser.add_argument('--classifier_lr', type=float, default=0.1,
                    help='Learning rate to train an classifier.')

parser.add_argument('--max_epoch', type=int, default=400,
                    help='Maximum epoch to train an encoder.')
parser.add_argument('--num_workers', type=int, default=16,
                    help='Number of workers for data loader.')
parser.add_argument('--dataset', type=str, default='CIFAR100',
                    help='dataset')

config = parser.parse_args()

# Show config
print('\n======================= Training configuration =======================\n')
pprint.pprint(vars(config))
print('\n======================================================================\n')

Ckpt_path = os.path.join(config.channel_ckpt_root, config.dataset, config.channel_model,
                           "{}_{}_{}_{}".format(config.temperature, config.CMItemperature, 
                                                config.lr, config.CMIFactor))
Ckpt_path = os.path.join(Ckpt_path, 'weight', 'ckpt_' +str(config.load_pretrained_epoch) + '.pt')

channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _, dlen = \
    get_dataset(dataset =config.dataset, data_path= config.dataset_root, args = config)

train_loader = DataLoader(dst_train, batch_size=config.batch_size, shuffle=True, 
                        pin_memory=True, num_workers=config.num_workers)

test_loader = DataLoader(dst_test, batch_size=config.batch_size, shuffle=True, 
                        pin_memory=True, num_workers=config.num_workers)

print('\n[2 / 3]. Build models.. ')
encoder = get_channel(model_name=config.channel_model, 
                      feat_dim = config.feature_dim, 
                      dataset=config.dataset)
encoder = nn.DataParallel(encoder).to(dev)


ckpt = torch.load(Ckpt_path)

encoder.load_state_dict(ckpt['encoder'])
# encoder.module.fc = torch.nn.Identity().to(dev)
# breakpoint()
#####################################################################

# Modify the Encoder

#####################################################################

linear = MLP(config, num_classes).to(dev) 

optim_linear = optim.SGD(linear.parameters(),
                         lr=config.classifier_lr,
                         momentum=config.SGD_momentum,
                         weight_decay=config.weight_decay)

cross_entropy = nn.CrossEntropyLoss()

''' ######################## < Step 3 > Define methods ######################## '''

def get_accuracy(dloader):
    '''
    Calculate classification accuracy
    '''
    
    total_num = 0
    correct_num = 0
    # print('\n Calculate accuracy ...')


    encoder.eval()
    linear.eval()


    with torch.no_grad():
        for idx, (img, label) in enumerate(dloader):
            # if idx % 50 == 0:
                # print('    [%d / %d] ... \n' % (idx, int(tst_dlen / config.tst_batch_size)))

            img = img.to(dev)
            label = label.to(dev)
            feature = encoder(img)
            # feature = normlayer(feature)
            score = linear(feature.view(feature.size(0), feature.size(1)))
            pred = torch.argmax(score, dim=1, keepdim=True)

            total_num = total_num + img.size(0)
            correct_num = correct_num + (label.unsqueeze(dim=1) == pred).sum().item()
        # print()
    return correct_num / total_num

def update_lr(epoch):    
    '''
    Learning rate scheduling.

    Args:
        epoch (float): Set new learning rate by a given epoch.
    '''
    
    # Decay 0.1 times every 20 epoch.
    factor = int(epoch / 10)
    lr = config.lr * (0.2**factor)

    for param_group in optim_linear.param_groups:
        # print('LR is updated to %f ...' % lr)
        param_group['lr'] = lr

''' ######################## < Step 4 > Start training ######################## '''

epoch = 0
total_iters = 0
ACCs = []
accr_tst_hist = []
accr_trn_hist = []
# Train a linear classifier
features = []

for _ in tqdm(range(config.max_epoch)):
    # print(epoch, flush=True)
    encoder.eval()
    # Preprocess
    linear.train()
    for i, (img, label) in enumerate(train_loader):
        
        optim_linear.zero_grad()
        # Forward prop
        img = img.to(dev)
        label = label.to(dev)
        with torch.no_grad():
            feature = encoder(img).detach()
        # feature = normlayer(feature)

        score = linear(feature.view(feature.size(0), feature.size(1)))
        loss = cross_entropy(score, label)
        
        # Back prop
        loss.backward()
        optim_linear.step()
        
        # Print training status and save log
        total_iters += 1
        # print('[Epoch : %d / Total iters : %d] : loss : %f ...' %(epoch, total_iters, loss.item()))
    
    epoch += 1
    
    # Update learning rate
    update_lr(epoch)
    
    # Save loss value
    # loss_hist.append(loss.item())
    
    # Calculate the current accuracy and plot the graphs
    # if (epoch - 1) % config.eval_epoch == 0:
    linear.eval()

    accr_trn = get_accuracy(train_loader)
    accr_tst = get_accuracy(test_loader)

    accr_tst_hist.append(accr_tst)
    accr_trn_hist.append(accr_trn)
    # breakpoint()
    print('[Epoch : {} / Total iters : {}] : test accr : {} ...'.format(epoch, total_iters, accr_tst))
    print('[Epoch : {} / Total iters : {}] : training accr : {} ...'.format(epoch, total_iters, accr_trn))

        # util.cls_loss_plot(loss_hist, loss_path, record_epoch=1)
        # util.accr_plot(accr_hist, accr_path, record_epoch=config.eval_epoch)
       
    # Save
    # if (epoch - 1) % config.save_weight_epoch == 0:
        # path_ckpt = os.path.join(weight_path, 'ckpt_' + str(epoch-1) + '.pkl')
        # ckpt = linear.state_dict()
        # torch.save(ckpt, path_ckpt)
        
        # with open(os.path.join(loss_path, 'loss.pkl'), 'wb') as f:
        #     pickle.dump(loss_hist, f)
            
        # with open(os.path.join(accr_path, 'accr.pkl'), 'wb') as f:
        #     pickle.dump(accr_tst_hist, f)
print("Test Acc: {}, Training Acc: {} ".format(max(accr_tst_hist), max(accr_trn_hist)), flush=True)
print(" ")