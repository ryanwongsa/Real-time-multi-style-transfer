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
from train.Trainer import Trainer

import argparse


def train(args):
    dataset_images = args.dataset_images + "/"
    styles_dir = args.styles_dir
    num_workers = args.num_workers
    batch_size = args.batch_size
    model_dir=args.model_dir
    epochs = args.epochs
    eval_image_dir = args.eval_image_dir
    eval_dir = args.eval_dir
    lr=args.lr
    content_factor = args.content_factor
    style_factor = args.style_factor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_images_dir = glob(os.path.join(styles_dir, '*.jpg'))
    num_styles = len(style_images_dir)
    image_size = 256
    mean = [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    layers = {
                '3':"relu1_2",
                '8':"relu2_2",
                '15':"relu3_3",
                '22':"relu4_3",
            }
    content_layer = "relu3_3"
    style_layers = {
        "relu1_2": 1.0,#0.4,
        "relu2_2": 1.0,#0.3,
        "relu3_3": 1.0,#0.2,
        "relu4_3": 1.0,#0.1
    } 
    
    eval_image_size = 512

    data_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ])

    photo_dataset = PhotoDataset(root_dir=dataset_images, transform = data_transform)

    dataloader = DataLoader(photo_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)

    vggfmodel = Vgg16FeatureModel(layers, device)

    pastichemodel = PasticheModel(num_styles)
    pastichemodel = pastichemodel.to(device)

    optimizer = optim.Adam(pastichemodel.parameters(), lr=lr, betas=(0.9, 0.999))

    style_targets = [vggfmodel.get_style_gram(data_transform(Image.open(style_image_dir).convert('RGB')).unsqueeze(0).to(device)) for style_image_dir in style_images_dir]

    eval_set = None
    if eval_image_dir!=None:
        im = Image.open(eval_image_dir).convert('RGB')
        transform = transforms.Compose([
                transforms.Resize(eval_image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)
            ])
        eval_set = (im,transform, eval_dir)
    trainer = Trainer(vggfmodel,pastichemodel,style_targets,device, layers,content_layer,style_layers)
    
    content_loss, style_loss = trainer.train(dataloader, optimizer, epochs, style_factor=style_factor, content_factor=content_factor, save_dir=model_dir,eval_set=eval_set)

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for training mutli-style-transfer")
    
    main_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    main_arg_parser.add_argument("--batch-size", type=int, default=8,
                                  help="number of items in a batch, default is 8")
    
    main_arg_parser.add_argument("--num-workers", type=int, default=0,
                                  help="number of workers")
    
    main_arg_parser.add_argument("--dataset-images", type=str, required=True,
                                  help="path to folder containing training dataset")
    
    main_arg_parser.add_argument("--styles-dir", type=str, required=True,
                                  help="path to folder containing style images")
    main_arg_parser.add_argument("--lr", type=float, default=0.0001,
                                  help="learning rate for training")
    
    main_arg_parser.add_argument("--content-factor", type=int, default=1,
                                  help="content factor, default is 1")
    main_arg_parser.add_argument("--style-factor", type=int, default=1000000,
                                  help="style factor, default is 1000000")
    
    main_arg_parser.add_argument("--eval-dir", type=str, default=None,
                                  help="directory to save image evaluation results during training. Requires eval-image-dir")
    main_arg_parser.add_argument("--eval-image-dir", type=str, default=None,
                                  help="image to use to save im the eval-dir")
    
    main_arg_parser.add_argument("--model-dir", type=str, default=None,
                                  help="directory to save the model in")
    
    args = main_arg_parser.parse_args()
    train(args)
    

if __name__ == '__main__':
    main()
    