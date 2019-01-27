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
from matplotlib.pyplot import imshow
import time
import os
import copy
import pandas as pd
from PIL import Image
from skimage import io, transform
import random
from tqdm import tqdm
from sklearn.utils import shuffle
from random import randint

from dataloaders.PhotoDataset import PhotoDataset
from models.Vgg16FeatureModel import Vgg16FeatureModel
from models.PasticheModel import ResBlock, CondConvolution, Upsampling, PasticheModel
from collections import defaultdict 

class Trainer(object):
    def __init__(self, feature_model,pastiche_model,style_targets,device,
                layers,content_layer,style_layers):
        self.featuremodel = feature_model
        self.pastichemodel = pastiche_model
        self.styletargets = style_targets
        self.device = device
        self.layers = layers
        self.contentlayer = content_layer
        self.stylelayers = style_layers
        
        self.num_styles = len(self.styletargets)
        
        self.mean = [0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]   
        
    def normalize_batch(self, batch):
        n_mean = batch.new_tensor(self.mean).view(-1, 1, 1)
        n_std = batch.new_tensor(self.std).view(-1, 1, 1)
        return (batch - n_mean) / n_std

    def unnormalize_batch(self, batch):
        n_mean = batch.new_tensor(self.mean).view(-1, 1, 1)
        n_std = batch.new_tensor(self.std).view(-1, 1, 1)
        return (batch * n_std) + n_mean
    
    def makedir(self, dir_name):
        try:
            os.mkdir(dir_name)
        except FileExistsError:
            pass
            
            
    def train(self, dataloader, optimizer, epoches, save_dir=None, style_factor=10000000000, content_factor=100000,epoch_start=0, save_step=1000, content_loss_dict=defaultdict(list),style_loss_dict=defaultdict(list), eval_set=None):
        
        content_loss_dict = content_loss_dict.copy()
        style_loss_dict = style_loss_dict.copy()
        mse_loss = torch.nn.MSELoss()
        self.pastichemodel = self.pastichemodel.train()
        self.makedir(save_dir)
        for i in range(epoch_start,epoches):
            step = 0
            pbar = tqdm(dataloader)
            for img, img_ids in pbar:
                img = img.to(self.device)
                features = self.featuremodel.get_features(img)
                optimizer.zero_grad()

                style_no = randint(0, self.num_styles-1)

                output_temp = self.pastichemodel(img,style_no)

                output = self.normalize_batch(output_temp)
                output_features = self.featuremodel.get_features(output)

#                 content_loss = torch.mean((features[self.contentlayer] - output_features[self.contentlayer])**2,(1,2,3))
                content_loss = mse_loss(features[self.contentlayer],output_features[self.contentlayer])

                style_loss = 0.0
                for key, value in self.layers.items():
                    output_feature = output_features[value]
                    output_gram = self.featuremodel.gram_batch_matrix(output_feature)

                    style_gram = self.styletargets[style_no][value]
#                     style_loss += self.stylelayers[value]*torch.mean(((output_gram - style_gram)**2),(1,2))
                    style_loss += self.stylelayers[value]*mse_loss(output_gram, style_gram[:len(img_ids), :, :])
               
                
                content_loss = content_factor * content_loss
                style_loss = style_factor * style_loss
                
            
                total_loss = content_loss + style_loss
                
                
                pbar.set_description(str(content_loss.item())+", "+str(style_loss.item()))
                
                total_loss.backward()
                optimizer.step()
                content_loss_dict[style_no].append(content_loss.item())
                style_loss_dict[style_no].append(style_loss.item())
                
                if step%save_step==0:
                    if eval_set != None:
                        self.pastichemodel = self.pastichemodel.eval()
                        division = int(self.num_styles**0.5)
                        plt.figure(figsize=(20,20))
                        for i in range(division):
                            for j in range(division):
                                plt.subplot(division, division, i*division+j+1, frameon=False)
                                res = self.eval_image(eval_set[0],eval_set[1],i*division+j)
                                plt.imshow(np.asarray(res))
                                plt.axis('off') 
                                del res
                        plt.show()
                        self.pastichemodel = self.pastichemodel.train()
                    if save_dir != None:
                        torch.save(self.pastichemodel.state_dict(), save_dir+"pastichemodel_"+str(i)+"-"+str(step)+".pth")
                step+=1
            if save_dir != None:
                torch.save(self.pastichemodel.state_dict(), save_dir+"pastichemodel_"+str(i)+"-FINAL.pth")
        return content_loss_dict, style_loss_dict
    
    def set_mode(self,mode):
        if mode=="eval":
            self.pastichemodel = self.pastichemodel.eval()
        else:
            self.pastichemodel = self.pastichemodel.train()
            
    def eval_image(self, img, transform, style_num):
        out = transform(img)
        res = self.pastichemodel(out.unsqueeze(0).to(self.device),style_num)
        res_img = Image.fromarray(np.uint8(np.moveaxis(res[0].cpu().detach().numpy()*255.0, 0, 2)))
        return res_img
                     
    def load_model_weights(self, dir_model):
        self.pastichemodel.load_state_dict(torch.load(dir_model))
        
    def transfer_learn_model(self, pastiche_model, dir_model, style_targets):
        
        
        prev_state_dict = torch.load(dir_model)
        
        self.pastichemodel = pastiche_model
        model_dict = self.pastichemodel.state_dict()
        self.styletargets = style_targets
        self.num_styles = len(self.styletargets)
        pretrained_dict = {}
        for name, param in prev_state_dict.items():
            if "gamma" in name or "beta" in name:
                continue
            else:
                pretrained_dict[name] = param
                
        model_dict.update(pretrained_dict) 
        self.pastichemodel.load_state_dict(model_dict)
        
        for name, param in self.pastichemodel.named_parameters():
            if "gamma" in name or "beta" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False) 
                                  
    def set_training_mode(self, mode="full"):
        if mode=="full":
            for name, param in self.pastichemodel.named_parameters():
                param.requires_grad_(True)
        elif model=="finetune":
            for name, param in self.pastichemodel.named_parameters():
                if "gamma" in name or "beta" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)    
        else:
            print("Training mode unchanged")