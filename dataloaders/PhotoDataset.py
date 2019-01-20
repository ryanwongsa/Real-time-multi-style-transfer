import torch
from glob import glob
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class PhotoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.list_images = glob(os.path.join(self.root_dir, '*.jpg'))
                                
    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        image_loc = self.list_images[idx]
        image = Image.open(image_loc).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_loc.split('/')[-1].split('.')[0].split('\\')[1]