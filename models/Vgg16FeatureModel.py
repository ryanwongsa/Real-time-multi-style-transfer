import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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

class Vgg16FeatureModel(object):
    
    def __init__(self, layers, device):
        self.vgg = models.vgg16(pretrained=True).features
        
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        
        self.vgg = self.vgg.to(device)
        
        self.layers = layers
    
    def get_features(self, images):
        features = {}
        x = images

        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x

        return features
    
    def gram_batch_matrix(self, tensor):
        b, d, h, w = tensor.size()
        tensor = tensor.view(b, d, h * w)
        gram = tensor.bmm(tensor.transpose(1,2)) / (d * h * w) 
        return gram
    
    def get_style_gram(self, image):
        style_features = self.get_features(image)
        style_grams = {layer: self.gram_batch_matrix(style_features[layer]) for layer in style_features}
        return style_grams