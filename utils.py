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
import torch.nn.functional as F
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import pandas as pd
import time
from config import *
from model import * 

config = Config()
class Person_Dataset_Visualization():
    def __init__(self, data_dir, transformer=None):
        self.person_items = []
        for (root, dirs, files) in os.walk(data_dir):
            if len(files) > 0:
                for file_name in files:
                    self.person_items.append(os.path.join(root, file_name))
        self.transformer = transformer

    def __len__(self):
        return len(self.person_items)

    def __getitem__(self, index):
        person_path = self.person_items[index]
        person_image = cv2.imread(person_path)
        if self.transformer:
            person_image = self.transformer(person_image)
        person_name = Path(person_path).parent.name
        return person_image, person_name

class Visualize():
    def __init__(self, all_embeds, all_images_real, epoch, acc):
        self.all_embeds = all_embeds
        self.all_images_real = all_images_real
        self.epoch = epoch
        self.acc = acc

    def visualize(self):
        tsne = TSNE(n_components=2, perplexity=10, n_iter=300)
        tsne_ref = tsne.fit_transform(self.all_embeds)
        df = pd.DataFrame(tsne_ref, index=tsne_ref[0:,1])
        df['x'] = tsne_ref[:, 0]
        df['y'] = tsne_ref[:, 1]
        coord_x = df['x'].values.tolist()
        coord_y = df['y'].values.tolist()
        plt.rcParams['figure.facecolor'] = 'black'
        fig, ax = plt.subplots(figsize=(18, 9))
        plt.margins(0, 0)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        self.imscatter(coord_x, coord_y, self.all_images_real, zoom=1.0, ax=ax)
        ax.plot(coord_x, coord_y)
        fig.canvas.draw()
        plt.close()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img=img, text=f"Epoch: {self.epoch + 1} Accuracy: {round(self.acc, 4)}", org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("plot", img)
        cv2.waitKey(1)

    def imscatter(self, x, y, image, ax=None, zoom=1):
        if ax is None:
            ax = plt.gca()
        artists = []
        for idx, v in enumerate(x):
            x0 = x[idx]
            y0 = y[idx]
            im = OffsetImage(image[idx], zoom=zoom)
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        return artists

def visualize_on_training(model, dataloaders, epoch, acc):
    all_images_real = []
    all_embeds = []
    for index, batch in enumerate(dataloaders):
        if index > 299:
            break
        image = batch[0]
        with torch.no_grad():
            embed = model(image.to('cpu'))
        embed = embed.cpu().numpy()
        image = make_grid(image, normalize=True).permute(1, 2, 0)
        image = cv2.resize(np.array(image), dsize=(40, 40), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_images_real.append(image)
        all_embeds.append(embed)
    all_embeds = torch.Tensor(np.stack(all_embeds, axis=0).squeeze(1))
    make_plot = Visualize(all_embeds=all_embeds, all_images_real=all_images_real, epoch=epoch, acc=acc)
    make_plot.visualize()

class ContrastiveLoss(nn.Module):
  def __init__(self, margin):
    super().__init__()
    self.margin = margin

  def forward(self, z1, z2, label):
    dist = F.pairwise_distance(z1, z2, keepdim=True)
    loss = torch.mean((1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(self.margin - dist, min=0), 2))
    acc = ((dist > 0.6) == label).float().mean()
    return loss, acc

def save_model(model_state, model_name):
  os.makedirs("checkpoints", exist_ok = True)
  torch.save(model_state, os.path.join("checkpoints", model_name))

def load_model(ckpt_path, device):
#  checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint = torch.load(ckpt_path)
    model = SiameseNetwork(input_channel=3) if config.isColor else SiameseNetwork(input_channel=1)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model.to(device)



