import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from glob import glob
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from PIL import Image
from skimage import io, transform
import random
from tqdm import tqdm
from sklearn.utils import shuffle
    
class CondConvolution(nn.Module):
    def __init__(self, input_filters, output_filters, kernel_size, padding, stride,num_styles,act=True):
        super(CondConvolution, self).__init__()
        
        self.reflection2d = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(input_filters, output_filters,kernel_size=kernel_size, stride=stride)
        self.instnorm = nn.InstanceNorm2d(output_filters, affine=True)
        self.act = act
        self.gamma = torch.nn.Parameter(data=torch.Tensor(num_styles, output_filters), requires_grad=True)
        self.gamma.data.uniform_(1.0, 1.0)
        
        self.beta = torch.nn.Parameter(data=torch.Tensor(num_styles, output_filters), requires_grad=True)
        self.beta.data.uniform_(0.0, 0.0)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight,mean=0, std= 0.01)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, style_no):
        
        x = self.reflection2d(x)
        x = self.conv(x)
        x = self.instnorm(x)
        b,d,w,h =x.size()
        x = x.view(b,d,w*h)
        
#         new_gamma = self.gamma[style_no].unsqueeze(-1).expand_as(x)
#         new_beta = self.beta[style_no].unsqueeze(-1).expand_as(x)
        x = (x*self.gamma[style_no].unsqueeze(-1).expand_as(x)+self.beta[style_no].unsqueeze(-1).expand_as(x)).view(b,d,w,h)
        
        
        if self.act==True:
            x = F.relu(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,num_styles):
        super(ResBlock, self).__init__()
        self.res_conv1 = CondConvolution(128, 128, 3, 1, (1,1), num_styles)
        self.res_conv2 = CondConvolution(128, 128, 3, 1, (1,1), num_styles, False)

    def forward(self, x, style_no):
        residual = x
        out = self.res_conv1(x,style_no)
        out = self.res_conv2(out,style_no)
        out += residual
        return out
    
class Upsampling(nn.Module):
    def __init__(self, input_filters, output_filters, kernel_size, padding, stride,num_styles):
        super(Upsampling, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = CondConvolution(input_filters, output_filters, kernel_size,padding,stride,num_styles) 

    def forward(self, x,style_no):
        x = self.upsample(x)
        x = self.conv(x, style_no)
        return x
    
class PasticheModel(nn.Module):
    def __init__(self, num_styles):
        super(PasticheModel, self).__init__()
        self.conv_1 = CondConvolution(3, 32, 9, 4, (1,1),num_styles)
        self.conv_2 = CondConvolution(32, 64, 3, 1, (2,2),num_styles)
        self.conv_3 = CondConvolution(64, 128, 3, 1, (2,2),num_styles)
        
        self.upsample1 = Upsampling(128,64,3,1,(1,1),num_styles)
        self.upsample2 = Upsampling(64,32,3,1,(1,1),num_styles)
        
        self.res_block = ResBlock(num_styles)
        
        self.conv_4 = CondConvolution(32, 3, 9, 4, (1,1),num_styles, False)
        
        
    def forward(self, x, style_no):
        x = self.conv_1(x, style_no)
        x = self.conv_2(x, style_no)
        x = self.conv_3(x, style_no)
        
        # TODO: Fix bug here. it is only use one resblock and updating it. Don't for loop
        for i in range(4): 
            x=self.res_block(x, style_no)
        
        x = self.upsample1(x, style_no)
        x = self.upsample2(x, style_no)
        
        x = self.conv_4(x, style_no)
        x = F.sigmoid(x)
        return x