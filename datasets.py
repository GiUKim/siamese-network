from glob import glob
import os
from pathlib import Path
from torchsummary import summary as summary
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
import warnings
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import pandas as pd
import time
from config import *

config = Config()
class Person_Dataset():
    def __init__(self, data_dir, phase, transformer=None):
        self.person_items = []
        for (root, dirs, files) in os.walk(os.path.join(data_dir, phase)):
            if len(files) > 0:
                for file_name in files:
                    self.person_items.append(os.path.join(root, file_name))
        self.transformer = transformer

    def __len__(self):
        return len(self.person_items)

    def __getitem__(self, index):
        personA_path = self.person_items[index]
        person = Path(personA_path).parent.name
        same_person = np.random.randint(2)
        if same_person:
            same_person_dir = Path(personA_path).parent
            same_person_fn = [fn for fn in os.listdir(same_person_dir) if fn.endswith('.jpg')]
            personB_path = os.path.join(same_person_dir, np.random.choice(same_person_fn))
        else:
            while True:
              personB_path = np.random.choice(self.person_items)
              if person != Path(personB_path).parent.name:
                break
        personA_image = cv2.imread(personA_path)
        personB_image = cv2.imread(personB_path)

        if self.transformer:
            personA_image = self.transformer(personA_image)
            personB_image = self.transformer(personB_image)
        # same person 일 때 0(pos), another person일 때1(neg)
        return personA_image, personB_image, np.array([1 - same_person])

def build_transformer(image_size=48): # 100x100 resize
    transformers = {}
    transformers['train'] = transforms.Compose([
          transforms.ToPILImage(),
          transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
          transforms.RandomHorizontalFlip(),
          transforms.Resize((image_size, image_size)),
          transforms.ToTensor(),
          transforms.Normalize((0.5), (0.5)),
    ])
    transformers['val'] = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize((image_size, image_size)),
          transforms.ToTensor(),
          transforms.Normalize((0.5), (0.5)),
    ])
    return transformers
## debug
##
def build_dataloader(data_dir, batch_size=config.BATCH_SIZE):
    dataloaders = {}
    transformers = build_transformer()
    train_dataset = Person_Dataset(data_dir=config.data_dir, phase='train', transformer=transformers['train'])
    val_dataset = Person_Dataset(data_dir=config.data_dir, phase='val', transformer=transformers['val'])
    dataloaders['train'] = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    dataloaders['val'] = DataLoader(val_dataset, shuffle=False, batch_size=1)
    return dataloaders

