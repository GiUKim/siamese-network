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
from model import *
from datasets import *
from config import *
from utils import *

def train_one_epoch(dataloaders, epoch, model, criterion, optimizer, device):
    losses = {}
    accuracies = {}
    for phase in ['train', 'val']:
        running_loss = 0.0
        running_acc = 0.0
        if phase == 'train':
            model.train()
        else:
            model.eval()
        for index, batch in enumerate(dataloaders[phase]):
            # if index > 8 and phase=='train':
            #     break
            imgA = batch[0].to(device)
            imgB = batch[1].to(device)
            label = batch[2].to(device)
            with torch.set_grad_enabled(phase == 'train'):
                codeA = model(imgA)
                codeB = model(imgB)
            loss, acc = criterion(codeA, codeB, label)
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            running_acc += acc.item()
        losses[phase] = running_loss / len(dataloaders[phase])
        accuracies[phase] = running_acc / len(dataloaders[phase])
        if phase == 'val':
            start = time.time()
            visualize_on_training(model, dataloaders['val'], epoch, accuracies['val'])
            print("visualization time: {} sec".format(round(float(time.time() - start), 2)))
    return losses, accuracies

if __name__ == '__main__':
    config = Config()
    data_dir = config.data_dir
    IMAGE_SIZE = config.IMAGE_SIZE
    transformers = build_transformer(image_size=config.IMAGE_SIZE)
    warnings.simplefilter("ignore", DeprecationWarning)
    BATCH_SIZE=config.BATCH_SIZE
    dataloaders = build_dataloader(config.data_dir, BATCH_SIZE)
    is_cuda = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() and is_cuda else 'cpu')
    print('train img lenght: ', len(dataloaders['train']) * BATCH_SIZE)
    print('validation img lenght: ', len(dataloaders['val']))
    model = SiameseNetwork(input_channel=3) if config.isColor else SiameseNetwork(input_channel=1)
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model = model.to(DEVICE)
    if config.isColor:
        summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))
    else:
        summary(model, (1, IMAGE_SIZE, IMAGE_SIZE))
    criterion = ContrastiveLoss(margin=config.margin)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.max_lr, momentum=0.9)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=config.min_lr, verbose=True)
    num_epochs = config.epochs
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    max_val_accuracy = -1
    dset = Person_Dataset_Visualization(data_dir)
    transformer_visualization = build_transformer(image_size=config.IMAGE_SIZE)['val']
    dataset_visualization = Person_Dataset_Visualization(data_dir, transformer=transformer_visualization)
    dataloader_visualization = DataLoader(dataset_visualization, shuffle=False, batch_size=1)
    for epoch in range(num_epochs):
        losses, accs = train_one_epoch(dataloaders, epoch, model, criterion, optimizer, DEVICE)
        train_loss.append(losses['train'])
        train_acc.append(accs['train'])
        val_loss.append(losses['val'])
        val_acc.append(accs['val'])
        scheduler.step()
        print(f"{epoch} / {num_epochs} - train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
        print(f"{epoch} / {num_epochs} - train acc: {accs['train']:.4f}, val acc: {accs['val']:.4f}")
        print('-'*40)
        if max_val_accuracy < accs['val'] :
            max_val_accuracy = accs['val']
            save_model(model.state_dict(), f"model_{epoch + 1}_{accs['val']:.4f}.pth")
    
      
