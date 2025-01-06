import matplotlib.pyplot as plt
import os
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import Dataset
# import kornia as K
from torchvision.transforms import v2

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_dataset(dataset, data_path, args):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.5, 1.0)),
                                        transforms.GaussianBlur(kernel_size=3, sigma=(0.0001, 0.1)),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
        
        testtransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=testtransform)
        class_names = dst_train.classes
        dlen = 50000
    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]

        transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.5, 1.0)),
                                        transforms.GaussianBlur(kernel_size=3, sigma=(0.0001, 0.1)),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
        
        testtransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=testtransform)
        class_names = dst_train.classes
        dlen = 50000
    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        class_names = data['classes']

        images_train = data['images_train']
        labels_train = data['labels_train']
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:,c] = (images_train[:,c].clone() - mean[c])/std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c].clone() - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation
        dlen = 100000
    elif dataset == 'ImageNette':
        channel = 3
        im_size = (128, 128)
        num_classes = 10

        class_names = ["Tench", "English Springer", "Cassette Player", "Chainsaw", "Church", "French Horn", "Garbage Truck", "Gas Pump","Golf Ball", "Parachute"]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
                                        transforms.Resize(im_size),
                                        transforms.CenterCrop(im_size)])
        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform) # no augmentation
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)
        
    elif dataset == 'ImageWoof':
        channel = 3
        im_size = (128, 128)
        num_classes = 10

        class_names = ["Australian Terrier", "Border Terrier", "Samoyed", "Beagle", "Shih-Tzu" ,"English Foxhound", "Rhodesian Ridgeback", "Dingo", "Golden Retriever", "English Sheepdog"]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
                                        transforms.Resize(im_size),
                                        transforms.CenterCrop(im_size)])

        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform) # no augmentation
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, dlen

class MultiInsDataset(Dataset):
    def __init__(self, dataset, dlen, NRepeat=8):
        self.dataset = dataset
        self.dlen = dlen
        self.NRepeat = NRepeat
    def __len__(self):
        return self.dlen
    def __getitem__(self, idx):
        reimgs = []
        for _ in range(self.NRepeat):
            reimg = self.dataset.__getitem__(idx)[0]
            reimgs.append(reimg)
        return torch.stack(reimgs, dim=0), idx
