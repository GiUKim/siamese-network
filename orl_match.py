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
def build_dataloader(data_dir, batch_size=32):
  dataloaders = {}
  transformers = build_transformer()
  train_dataset = Person_Dataset(data_dir=data_dir, phase='train', transformer=transformers['train'])
  val_dataset = Person_Dataset(data_dir=data_dir, phase='val', transformer=transformers['val'])
  dataloaders['train'] = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
  dataloaders['val'] = DataLoader(val_dataset, shuffle=False, batch_size=1)
  return dataloaders

def ConvBlock(in_channel, out_channel):
  return nn.Sequential(
      nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(out_channel),
  )

class SE_Block(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class SiameseNetwork(nn.Module):
  def __init__(self, input_channel):
    super().__init__()

    self.features = nn.Sequential(
        nn.Conv2d(input_channel, 32, kernel_size=7, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        SE_Block(32, 4),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(32, 64, kernel_size=4, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        SE_Block(64, 4),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(64, 256, kernel_size=4, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 30, (1,1)),
        nn.AdaptiveAvgPool2d(1)
    )
  def forward(self, x1):#, x2):
    z1 = self.features(x1)
#    z2 = self.features(x2)
    return z1.squeeze(-1).squeeze(-1)#, z2.squeeze(-1).squeeze(-1)

import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
  def __init__(self, margin):
    super().__init__()
    self.margin = margin

  def forward(self, z1, z2, label):
    dist = F.pairwise_distance(z1, z2, keepdim=True)
    loss = torch.mean((1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(self.margin - dist, min=0), 2))
    acc = ((dist > 0.6) == label).float().mean()
    return loss, acc

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

def visualize_on_training(dataloaders, epoch, acc):
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
            visualize_on_training(dataloaders['val'], epoch, accuracies['val'])
            print("visualization time: {} sec".format(round(float(time.time() - start), 2)))

    return losses, accuracies

def save_model(model_state, model_name):
  os.makedirs("checkpoints", exist_ok = True)
  torch.save(model_state, os.path.join("checkpoints", model_name))

def load_model(ckpt_path, device):
#  checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint = torch.load(ckpt_path)
    model = SiameseNetwork(input_channel=3)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model.to(device)

data_dir = 'datasets'
phase = 'train'
person_items = []
for (root, dirs, files) in os.walk(os.path.join(data_dir, phase)):
  if len(files) > 0:
    for file_name in files:
      person_items.append(os.path.join(root, file_name))
trn_ds = Person_Dataset(data_dir, phase="train")
index = 0
personA, personB, other = trn_ds[index]
IMAGE_SIZE = 48
transformers = build_transformer(image_size=IMAGE_SIZE)
warnings.simplefilter("ignore", DeprecationWarning)
train_dataset = Person_Dataset(data_dir=data_dir, phase='train', transformer=transformers['train'])
val_dataset = Person_Dataset(data_dir=data_dir, phase='val', transformer=transformers['val'])
BATCH_SIZE=32
dataloaders = build_dataloader(data_dir, BATCH_SIZE)
is_cuda = True
IMAGE_SIZE = 48
BATCH_SIZE=64
DEVICE = torch.device('cuda' if torch.cuda.is_available() and is_cuda else 'cpu')
dataloaders = build_dataloader(data_dir, BATCH_SIZE)
model = SiameseNetwork(input_channel=3)
model = nn.DataParallel(model, device_ids=[0,1,2,3])
model = model.to(DEVICE)
summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))
criterion = ContrastiveLoss(margin=2.0)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0.001, verbose=True)
num_epochs = 100
train_loss, train_acc = [], []
val_loss, val_acc = [], []
max_val_accuracy = -1
dset = Person_Dataset_Visualization(data_dir)
transformer_visualization = build_transformer(image_size=IMAGE_SIZE)['val']
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

  
