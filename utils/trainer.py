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


mean = [0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]   

def normalize_batch(batch):
    n_mean = batch.new_tensor(mean).view(-1, 1, 1)
    n_std = batch.new_tensor(std).view(-1, 1, 1)
    return (batch - n_mean) / n_std

def unnormalize_batch(batch):
    n_mean = batch.new_tensor(mean).view(-1, 1, 1)
    n_std = batch.new_tensor(std).view(-1, 1, 1)
    return (batch * n_std) + n_mean


def train(vggfmodel, pastichemodel,dataloader,style_targets,optimizer, epoches, device,layers,content_layer,style_layers,model_dir,style_factor,num_styles, style_images_dir, rand_style=True,style_choice=-1):
    pastichemodel = pastichemodel.train()
    step = 0
    step_show = 1000
    for i in range(epoches):
        pbar = tqdm(dataloader)
        for img, img_ids in pbar:
            img = img.to(device)
            features = vggfmodel.get_features(img)
            optimizer.zero_grad()
            
            if rand_style:
                style_no = randint(0, num_styles-1)
            else:
                style_no = style_choice
            
            output_temp = pastichemodel(img,style_no)

            output = normalize_batch(output_temp)
            output_features = vggfmodel.get_features(output)

            content_loss = torch.mean((features[content_layer] - output_features[content_layer])**2,(1,2,3))


            style_loss = torch.zeros(img.size()[0]).to(device)
            for key, value in layers.items():
                output_feature = output_features[value]
                output_gram = vggfmodel.gram_batch_matrix(output_feature)

                style_gram = style_targets[style_no][value]
                style_loss += style_layers[value]*torch.mean(((output_gram - style_gram)**2),(1,2))

            content_mean_loss = torch.mean(content_loss)
            style_mean_loss = style_factor*torch.mean(style_loss)

            total_loss = content_mean_loss + style_mean_loss
            prnt_content_loss = str(content_mean_loss.cpu().detach().numpy())
            prnt_style_loss = str(style_mean_loss.cpu().detach().numpy())

            pbar.set_description(prnt_content_loss+","+prnt_style_loss)

            total_loss.backward()
            optimizer.step()

            if step%step_show==0:
                test_org = unnormalize_batch(img)
                test_org = test_org[0].cpu().detach().numpy()*255.0
                test_org = np.moveaxis(test_org, 0, 2)
                im_org = Image.fromarray(np.uint8(test_org))

                test_img = output_temp[0].cpu().detach().numpy()*255.0
                test_img = np.moveaxis(test_img, 0, 2)
                im = Image.fromarray(np.uint8(test_img))

                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(20,20))
                ax1.imshow(np.asarray(im_org))
                ax1.axis('off')  
                ax2.imshow(np.asarray(im))
                ax2.axis('off')  
                plt.title(style_images_dir[style_no])
                plt.show()
                torch.save(pastichemodel.state_dict(), model_dir+"pastichemodel-"+str(step)+".pth")
            step+=1
    torch.save(pastichemodel.state_dict(), model_dir+"pastichemodel-FINAL.pth")

# def evaluate(vggfmodel, pastichemodel,dataloader,style_targets, device,layers,content_layer,style_layers,model_dir,style_factor,num_styles):
#     pastichemodel = pastichemodel.eval()
#     step = 0
#     step_show = 1000
#     for i in range(epoches):
#         pbar = tqdm(dataloader)
#         for img, img_ids in pbar:
#             img = img.to(device)
#             features = vggfmodel.get_features(img)
#             optimizer.zero_grad()

#             style_no = randint(0, num_styles-1)
#             output_temp = pastichemodel(img,style_no)

#             output = normalize_batch(output_temp)
#             output_features = vggfmodel.get_features(output)

#             content_loss = torch.mean((features[content_layer] - output_features[content_layer])**2,(1,2,3))


#             style_loss = torch.zeros(img.size()[0]).to(device)
#             for key, value in layers.items():
#                 output_feature = output_features[value]
#                 output_gram = vggfmodel.gram_batch_matrix(output_feature)

#                 style_gram = style_targets[style_no][value]
#                 style_loss += style_layers[value]*torch.mean(((output_gram - style_gram)**2),(1,2))

#             content_mean_loss = torch.mean(content_loss)
#             style_mean_loss = style_factor*torch.mean(style_loss)

#             total_loss = content_mean_loss + style_mean_loss
#             prnt_content_loss = str(content_mean_loss.cpu().detach().numpy())
#             prnt_style_loss = str(style_mean_loss.cpu().detach().numpy())

#             pbar.set_description(prnt_content_loss+","+prnt_style_loss)

#             total_loss.backward()
#             optimizer.step()

#             if step%step_show==0:
#                 torch.save(pastichemodel.state_dict(), model_dir+"pastichemodel-"+str(step)+".pth")
#             step+=1
#     torch.save(pastichemodel.state_dict(), model_dir+"pastichemodel-FINAL.pth")
    
    
def eval_image(pastichemodel,device,image_loc, image_size, style_num):
    pastichemodel = pastichemodel.eval()
    im2 = Image.open(image_loc).convert('RGB')
    data_transform_2 = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])
    out2 = data_transform_2(im2)
    res = pastichemodel(out2.unsqueeze(0).to(device),style_num)
    res_img = res[0].cpu().detach().numpy()*255.0
    res_img = np.moveaxis(res_img, 0, 2)
    res_im = Image.fromarray(np.uint8(res_img))
    return res_im