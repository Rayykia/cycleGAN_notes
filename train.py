"""
Train the cycle GAN.

Author: Rayykia
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import itertools

import utils
from cycleGAN import cycleGAN


if __name__ == "__main__":
    # # photo: 7038
    # train_A_path = glob.glob(r"E:/Study/python_code/pytorch_gan/cycleGAN/photo2monet/photo_jpg/*.jpg")
    # # monet: 300
    # train_B_path = glob.glob(r"E:/Study/python_code/pytorch_gan/cycleGAN/photo2monet/monet_jpg/*.jpg")

    # # plt.figure(figsize=(12,12))
    # # for i, img_path in enumerate(train_A_path[:2]):
    # #     monet_path = train_B_path[i]
    # #     photo_img = Image.open(img_path)
    # #     monet_img = Image.open(monet_path)
    # #     plt.subplot(2, 2, 2*i+1)
    # #     plt.imshow(photo_img)
    # #     plt.axis('off')
    # #     plt.subplot(2, 2, 2*i+2)
    # #     plt.imshow(monet_img)
    # #     plt.axis('off')
    # # plt.show()

    # set_A = utils.P2MDS(train_A_path)
    # set_B = utils.P2MDS(train_B_path)

    # BATCHSIZE = 32

    # loader_A = data.DataLoader(set_A, batch_size=BATCHSIZE, shuffle=True)
    # loader_B = data.DataLoader(set_B, batch_size=BATCHSIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # photo: 7038
    path_A = r"./horse/horse2zebra/trainA/*.jpg"
    # monet: 300
    path_B = r"./horse/horse2zebra/trainB/*.jpg"
    loader_A, loader_B = utils.set_loader(path_A, path_B, batch_size=4)

    cycGAN = cycleGAN(gen_net="resnet")
    cycGAN.fit(loader_A=loader_A, loader_B=loader_B, epoches = 201)
    

