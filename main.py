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
from model import SiameseNetwork
from loss import ContrastiveLoss
import data_transform
import datetime
import datasets
from data import SiameseNetworkDataset
import csv
import time
import shutil
import progressbar

import pdb



# random.seed(1)
dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch SiameseNet Training on several datasets')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=50, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 300')
parser.add_argument('--dataset', metavar='DATASET', default='gtsrb_data',
                    choices=dataset_names,
                    help='dataset type : ' +
                        ' | '.join(dataset_names) +
                        ' (default: gtsrb_data)')
parser.add_argument('--pretrained', dest='pretrained', default = None,
                    help='path to pre-trained model')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--log-summary', default = 'progress_log_summary.csv',
                    help='csv where to save per-epoch train and test stats')
parser.add_argument('--log-full', default = 'progress_log_full.csv',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--data-parallel', default=None,
                    help='Use nn.DataParallel() model')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')




class Config():
    base_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB"
    tr_im_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB/Experiment02-22-43/train_impaths.txt"
    tr_gt_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB/Experiment02-22-43/train_imclasses.txt"
    te_im_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB/Experiment02-22-43/test_impaths.txt"
    te_gt_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB/Experiment02-22-43/test_imclasses.txt"
    tmp_path = "/media/rcv/SSD1/Logo_oneshot/GTSRB/GTSRB_template_ordered"


BEST_TEST_LOSS = -1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


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



def main():    
    global args, BEST_TEST_LOSS, save_path
    args = parser.parse_args()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    input_transform = transforms.Compose([
                data_transform.PILScale((100,100)),
                transforms.ToTensor(),
                # normalize
        ])

    print("=> fetching image/label pairs in '{}'".format(args.dataset))
    train_set, test_set = datasets.__dict__[args.dataset](
        Config.base_path,
        Config.tr_im_path, 
        Config.tr_gt_path, 
        Config.te_im_path, 
        Config.te_gt_path, 
        Config.tmp_path,
        transform=input_transform,
        split=100,
        should_invert=False
    )

    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))


    ### Data visualization
    # vis_dataloader = DataLoader(train_set,
    #                         shuffle=True,
    #                         num_workers=args.workers,
    #                         batch_size=args.batch_size)
    # dataiter = iter(vis_dataloader)
    # example_batch = next(dataiter)
    # concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    # imshow(torchvision.utils.make_grid(concatenated))
    # print(example_batch[2].numpy())
    # pdb.set_trace()

    train_loader = DataLoader(train_set,
                        shuffle=True,
                        num_workers=args.workers,
                        batch_size=args.batch_size)
    test_loader = DataLoader(test_set, 
                        shuffle=False, 
                        num_workers=args.workers, 
                        batch_size=args.batch_size)

    net = SiameseNetwork().cuda()

    if args.pretrained:
        print("=> Use pre-trained model")
        weights = torch.load(args.pretrained)
        net.load_state_dict(weights['state_dict'])
    else:
        print("=> Randomly initialize model")
        net.init_weights()

    
    criterion = ContrastiveLoss()
    optimizer = optim.Adam( net.parameters(), lr=args.lr )

    if args.data_parallel:
        net = torch.nn.DataParallel(net).cuda()


    if args.evaluate:
        eval(test_set, net, input_transform)


    ### Visualize testset with pretrained model 
    # test_dataloader = DataLoader(test_set, shuffle=True, num_workers=args.workers, batch_size=1)
    # dataiter = iter(test_dataloader)
    # x0, _, _ = next(dataiter)

    # for i in range(10):
    #     _, x1, label2, _ = next(dataiter)
    #     concatenated = torch.cat((x0,x1),0)
        
    #     output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
    #     euclidean_distance = F.pairwise_distance(output1, output2)
    #     imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))
    # pdb.set_trace()



    save_path = '{}epochs,b{},lr{}'.format(
        args.epochs,
        args.batch_size,
        args.lr)
    timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H:%M")
    save_path = os.path.join(timestamp,save_path)
    save_path = os.path.join(args.dataset,save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
            os.makedirs(save_path)



    with open(os.path.join(save_path,args.log_summary), 'w') as csvfile:    # save every epoch
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['tr_loss','te_loss'])
    with open(os.path.join(save_path,args.log_full), 'w') as csvfile:       # save every iter
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['tr_loss_iter'])


    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train_loss = train(train_loader, net, criterion=criterion, optimizer=optimizer, epoch=epoch)
        test_loss = test(test_loader, net, criterion=criterion, epoch=epoch)

        if BEST_TEST_LOSS < 0:
            BEST_TEST_LOSS = test_loss
        is_best = test_loss < BEST_TEST_LOSS
        BEST_TEST_LOSS = min(test_loss, BEST_TEST_LOSS)



        ### Save checkpoints
        if args.data_parallel:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': net.module.state_dict(),  # args.data_parallel = True
                    'BEST_TEST_LOSS': BEST_TEST_LOSS,
                    }, is_best
                )
        else:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),         # args.data_parallel = False
                    'BEST_TEST_LOSS': BEST_TEST_LOSS,
                    }, is_best
                )
        if (epoch+1)%10 == 0:
            ckptname = 'ckpt_e%04d.pth.tar' %(epoch+1)
            if args.data_parallel:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.module.state_dict(),  # args.data_parallel = True
                    'BEST_TEST_LOSS': BEST_TEST_LOSS,
                }, os.path.join(save_path,ckptname))
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),         # args.data_parallel = False
                    'BEST_TEST_LOSS': BEST_TEST_LOSS,
                }, os.path.join(save_path,ckptname))

        ### Save epoch logs
        with open(os.path.join(save_path,args.log_summary), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, test_loss])



def train(train_loader, net, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    net.train()
    end = time.time()

    for i, data in enumerate(train_loader, 0):
        # if i>10: break;
        img0, img1, label, _ = data
        img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
        data_time.update(time.time() - end)

        output1, output2 = net(img0, img1)
        
        loss_contrastive = criterion(output1,output2,label)
        losses.update(loss_contrastive.data[0], label.size(0))

        optimizer.zero_grad()
        loss_contrastive.backward()
        optimizer.step()

        # if epoch > 2 and i % args.print_freq and loss_contrastive.data[0] > 3 == 0:
        #     ED = F.pairwise_distance(output1, output2)
        #     imgVisCat = torch.cat((img0.data.cpu(), img1.data.cpu()),0)
        #     imshow(torchvision.utils.make_grid(imgVisCat))
        #     pdb.set_trace()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}|{1}/{2}] '
                  'loss:{loss.val:.3f}({loss.avg:.3f})  '
                  'Batch time: {batch_time.val:.3f}({batch_time.avg:.3f})  '
                  'Data time: {data_time.val:.3f}({data_time.avg:.3f})'.format(
                   epoch, i, len(train_loader), 
                   loss=losses,
                   batch_time=batch_time, data_time=data_time))
        with open(os.path.join(save_path,args.log_full), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss_contrastive.data[0]])

    return losses.avg


def test(test_loader, net, criterion, epoch):
    losses = AverageMeter()

    net.eval()

    for i, data in enumerate(test_loader, 0):
        # if i>10: break;
        img0, img1, label, _ = data
        img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()

        output1, output2 = net(img0, img1)

        loss_contrastive = criterion(output1, output2, label)
        losses.update(loss_contrastive.data[0], label.size(0))


        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]  '
                  'loss: {loss.val:.3f}({loss.avg:.3f})'.format(
                   i, len(test_loader),
                   loss=losses))

    print(' * loss: {loss.avg:.3f}\t'.format(loss=losses))

    return losses.avg


# def eval(test_set, net, input_transform):
#     net.eval()

#     test_loader = DataLoader(test_set, 
#                         shuffle=False, 
#                         num_workers=args.workers, 
#                         batch_size=1)

#     tmp_list = sorted( os.listdir(Config.tmp_path) )
#     tmp = []
#     for i in range(len(tmp_list)):
#         img = Image.open( os.path.join(Config.tmp_path, tmp_list[i]) )
#         img = Variable(input_transform(img)).cuda()
#         tmp.append(img)

    
#     result_table = np.zeros( (len(test_loader), 2) )
#     bar = progressbar.ProgressBar(max_value=len(test_loader))
#     for i, data in enumerate(test_loader, 0):
#         # if i>10: break;
#         bar.update(i)
#         euclidean_distance = np.zeros(len(tmp_list))
#         img0, img1, label, cls0 = data
#         img0, img1, label, cls0 = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda(), Variable(cls0).cuda()

#         for tt in range(len(tmp_list)):
#             output1, output2 = net(img0, tmp[tt].unsqueeze(0))
#             euclidean_distance[tt] = F.pairwise_distance(output1, output2).cpu().data.numpy()[0][0]
#         # pdb.set_trace()

#         result_table[i, 0] = label.cpu().data.numpy()[0][0]
#         result_table[i, 1] = cls0.cpu().data.numpy()[0][0]
#     pdb.set_trace()



def eval(test_set, net, input_transform):
    net.eval()

    test_loader = DataLoader(test_set, 
                        shuffle=False, 
                        num_workers=args.workers, 
                        batch_size=400)

    tmp_list = sorted( os.listdir(Config.tmp_path) )
    tmp = []
    for i in range(len(tmp_list)):
        img = Image.open( os.path.join(Config.tmp_path, tmp_list[i]) )
        img = Variable(input_transform(img)).cuda()
        tmp.append(img)

    result_table = []
    bar = progressbar.ProgressBar(max_value=len(test_loader))
    for i, data in enumerate(test_loader, 0):
        bar.update(i)
        img0, img1, label, cls0 = data
        img0, img1, label, cls0 = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda(), Variable(cls0).cuda()
        ED = np.zeros( (img0.size(0), len(tmp_list)) )

        for tt in range(len(tmp_list)):
            output1, output2 = net(img0, tmp[tt].unsqueeze(0).repeat(img0.size(0),1,1,1))
            for bs in range(img0.size(0)):
                ED[bs][tt] = F.pairwise_distance(output1[bs].unsqueeze(0), output2[bs].unsqueeze(0)).cpu().data.numpy()[0][0]

        for bs in range(img0.size(0)):
            result_table.append([int(cls0.cpu().data.numpy()[bs][0]), ED[bs].argmin()])

    results = np.array(result_table)

    seenList = [1,2,3,4,5,7,8,9,10,11,12,13,14,15,17,18,25,26,31,33,35,38]
    unseenList = [0,6,16,19,20,21,22,23,24,27,28,29,30,32,34,36,37,39,40,41,42]
    numCorrSeen = 0
    numWrongSeen = 0
    numCorrUnseen = 0
    numWrongUnseen = 0
    for i in range(results.shape[0]): 
        if (results[i,0] in seenList) and (results[i,0] == results[i,1]): 
            numCorrSeen += 1
        elif (results[i,0] in seenList) and (results[i,0] != results[i,1]): 
            numWrongSeen += 1
        elif (results[i,0] in unseenList) and (results[i,0] == results[i,1]):
            numCorrUnseen += 1
        elif (results[i,0] in unseenList) and (results[i,0]!= results[i,1]):
            numWrongUnseen += 1


    scoreSeen = float(numCorrSeen) / float(numCorrSeen + numWrongSeen)
    scoreUnseen = float(numCorrUnseen) / float(numCorrUnseen + numWrongUnseen)
    pdb.set_trace()




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 periodically"""
    if (epoch+1) % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2



if __name__ == '__main__':
    main()