# Seokju Lee 2018.03.29
"""
Load siamese list
"""
import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import numpy as np
import pdb
import matplotlib.pyplot as plt
from torch.utils.serialization import load_lua
import torch
from scipy import ndimage, misc
from PIL import Image
import random


class ListDataset(data.Dataset):
    def __init__(self, path_list, temp_list, transform=None, should_invert=False, use_temp=False):
        self.path_list = path_list
        self.temp_list = temp_list
        self.transform = transform
        self.should_invert = should_invert
        self.use_temp = use_temp
        # pdb.set_trace()

    def __getitem__(self, index):
        img0_list = self.path_list[index]

        # 50% of images are in the same class
        should_get_same_class = random.randint(0,1)     

        # pos:neg = 1:nneg
        nneg = 1
        if random.randint(1,nneg+1) < nneg+1: 
            should_get_same_class = 0
        else:
            should_get_same_class = 1

        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                if self.use_temp:
                    img1_list = random.choice(self.temp_list)
                else:
                    img1_list = random.choice(self.path_list)
                if img0_list[1]==img1_list[1]:
                    break
        else:
            if self.use_temp:
                img1_list = random.choice(self.temp_list)
            else:
                img1_list = random.choice(self.path_list)

        img0 = Image.open(img0_list[0])
        img1 = Image.open(img1_list[0])
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")
        # pdb.set_trace()
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        # pdb.set_trace()

        return img0, img1, torch.from_numpy(np.array([int(img1_list[1]!=img0_list[1])],dtype=np.float32)), \
                torch.from_numpy( np.array( [float(img0_list[1])] ) )


    def __len__(self):
        return len(self.path_list)