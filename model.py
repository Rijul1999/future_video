# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:28:00 2019

@author: USER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class G_background(nn.Module):
    def __init__(self):
        super(G_background, self).__init__()
        self.model = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride = 2, padding = 1, bias = True), #[-1,512,4,4]
                nn.BatchNorm2d(512, eps = 1e-5),
                nn.ReLU(inplace = True),
                nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1, bias = True),
                nn.BatchNorm2d(256, eps = 1e-5),
                nn.ReLU(inplace = True),
                nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1, bias = True),
                nn.BatchNorm2d(128, eps = 1e-5),
                nn.ReLU(inplace = True),
                nn.ConvTranspose2d(128, 3, kernel_size = 4, stride = 2, padding = 1, bias = True),
                nn.Tanh()
                )

    def forward(self,x):
        #print('G_background Input =', x.size())
        out = self.model(x)
        #print('G_background Output =', out.size())
        return out
    
class G_video(nn.Module):
    def __init__(self):
        super(G_video, self).__init__()
        self.model = nn.Sequential(
                nn.ConvTranspose3d(1024, 1024, kernel_size=(2,1,1)), #[-1,512,4,4]
                nn.BatchNorm3d(1024, eps = 1e-5),
                nn.ReLU(inplace = True),
                nn.ConvTranspose3d(1024, 512, kernel_size = 4, stride = 2, padding = 1, bias = True),
                nn.BatchNorm3d(512, eps = 1e-5),
                nn.ReLU(inplace = True),
                nn.ConvTranspose3d(512, 256, kernel_size = 4, stride = 2, padding = 1, bias = True),
                nn.BatchNorm3d(256, eps = 1e-5),
                nn.ReLU(inplace = True),
                nn.ConvTranspose3d(256, 128, kernel_size = 4, stride = 2, padding = 1, bias = True),
                nn.BatchNorm3d(128, eps = 1e-5),
                nn.ReLU(inplace = True)
                )
    def forward(self,x):
        #print('G_video input =', x.size())
        out = self.model(x)
        #print('G_video output =', out.size())
        return out
    
    
    
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encode = G_encode()
        self.background = G_background()
        self.video = G_video()
        self.gen_net = nn.Sequential(nn.ConvTranspose3d(128, 3, kernel_size = 4, stride = 2, padding = 1, bias = True), nn.Tanh())
        self.mask_net = nn.Sequential(nn.ConvTranspose3d(128, 1, kernel_size = 4, stride = 2, padding = 1, bias = True), nn.Sigmoid())

    def forward(self,x):
        #print('Generator input = ',x.size())
        x = x.squeeze(2)
        print(x.size())
        encoded = self.encode(x)
        
        encoded = encoded.unsqueeze(2)
        video = self.video(encoded) #[-1, 128, 16, 32, 32], which will be used for generating the mask and the foreground
        #print('Video size = ', video.size())

        foreground = self.gen_net(video) #[-1,3,32,64,64]
        #print('Foreground size =', foreground.size())
        
        mask = self.mask_net(video) #[-1,1,32,64,64]
        #print('Mask size = ', mask.size())
        mask_repeated = mask.repeat(1,3,1,1,1) # repeat for each color channel. [-1, 3, 32, 64, 64]
        #print('Mask repeated size = ', mask_repeated.size())
        
        x = encoded.view((-1,1024,4,4))
        background = self.background(x) # [-1,3,64,64]
        #print('Background size = ', background.size())
        background_frames = background.unsqueeze(2).repeat(1,1,32,1,1) # [-1,3,32,64,64]
        out = torch.mul(mask,foreground) + torch.mul(1-mask, background_frames)
        #print('Generator out = ', out.size())        
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential( # [-1, 3, 32, 64, 64]
                nn.Conv3d(3, 128, kernel_size = 4, stride = 2, padding = 1, bias = True), #[-1, 64, 16, 32, 32]
                nn.LeakyReLU(negative_slope = 0.2, inplace = True), 
                nn.Conv3d(128, 256, kernel_size = 4, stride = 2, padding = 1, bias = True), #[-1, 126,8,16,16]
                nn.BatchNorm3d(256, eps = 1e-3), 
                nn.LeakyReLU(negative_slope = 0.2, inplace = True),
                nn.Conv3d(256, 512, kernel_size = 4, stride = 2, padding = 1, bias = True), #[-1,256,4,8,8]
                nn.BatchNorm3d(512, eps = 1e-3),
                nn.LeakyReLU(negative_slope = 0.2, inplace = True),
                nn.Conv3d(512, 1024, kernel_size = 4, stride = 2, padding = 1, bias = True), #[-1,512,2,4,4]
                nn.BatchNorm3d(1024, eps = 1e-3),
                nn.LeakyReLU(negative_slope = 0.2, inplace = True),
                nn.Conv3d(1024,2, kernel_size = (2,4,4), stride = (1,1,1), padding = (0,0,0), bias = True) #[-1,2,1,1,1] because (2,4,4) is the kernel size
                )
        #self.mymodules = nn.ModuleList([nn.Sequential(nn.Linear(2,1), nn.Sigmoid())])
        
    def forward(self, x):
        out = self.model(x).squeeze()
        #out = self.mymodules[0](out)
        return out
    
class G_encode(nn.Module):
    def __init__(self):
        super(G_encode, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size = 4, stride = 2, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1, bias = True),
                nn.BatchNorm2d(256, eps = 1e-5),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1, bias = True),
                nn.BatchNorm2d(512, eps = 1e-5),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1024, kernel_size = 4, stride = 2, padding = 1, bias = True),
                nn.BatchNorm2d(1024, eps = 1e-5),
                nn.ReLU(inplace=True),
                )
    def forward(self,x):
        #print('G_encode Input =', x.size())
        out = self.model(x)
        #print('G_encode Output =', out.size())
        return out
    
x = Variable(torch.rand([20, 3, 1, 64, 64]))
model = Generator()
out = model(x)
m2 = Discriminator()
o2 = m2(out)