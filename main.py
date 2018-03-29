'''
Seokju Lee, 2018.03.27
Base codes from "https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch"

v1: dataset - GTSRB

'''

import argparse
import os


import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import data_transform

import datasets
from data import SiameseNetworkDataset
import pdb



# random.seed(1)
dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch SiameseNet Training on several datasets')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 300')
parser.add_argument('--dataset', metavar='DATASET', default='gtsrb_data',
                    choices=dataset_names,
                    help='dataset type : ' +
                        ' | '.join(dataset_names) +
                        ' (default: gtsrb_data)')
args = parser.parse_args()



def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.ion()
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


class Config():
    base_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB"
    tr_im_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB/Experiment02-22-43/train_impaths.txt"
    tr_gt_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB/Experiment02-22-43/train_imclasses.txt"
    te_im_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB/Experiment02-22-43/test_impaths.txt"
    te_gt_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB/Experiment02-22-43/test_imclasses.txt"
# pdb.set_trace()



def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    input_transform = transforms.Compose([
                data_transform.PILScale((100,100)),
                transforms.ToTensor(),
                # normalize
        ])
    train_set, test_set = datasets.__dict__[args.dataset](
        Config.base_path,
        Config.tr_im_path, 
        Config.tr_gt_path, 
        Config.te_im_path, 
        Config.te_gt_path, 
        transform=input_transform,
        split=100,
        should_invert=False
    )


    vis_dataloader = DataLoader(train_set,
                            shuffle=True,
                            num_workers=args.workers,
                            batch_size=args.batch_size)
    dataiter = iter(vis_dataloader)


    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())

    pdb.set_trace()















if __name__ == '__main__':
    main()