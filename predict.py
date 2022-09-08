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
import shutil

model_path = 'checkpoints/model_21_0.5912.pth'
config = Config()
IMAGE_SIZE = config.IMAGE_SIZE
is_cuda = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() and is_cuda else 'cpu')
model = load_model(model_path, DEVICE)
#model = SiameseNetwork(input_channel=3) if config.isColor else SiameseNetwork(input_channel=1)
#model = nn.DataParallel(model, device_ids=[0,1,2,3])
#model = model.to(DEVICE)

if config.isColor:
    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))
else:
    summary(model, (1, IMAGE_SIZE, IMAGE_SIZE))

#train_dataset = Person_Dataset(data_dir=config.data_dir, phase='train', transformer=transformers['val'])
#dataloaders['train'] = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)

imgs = glob('datasets/train/*/*.jpg')
norm = transforms.Normalize((0.5), (0.5))
out_list = []
idx = -1
class_list = sorted([f.split('/')[-1] for f in glob('datasets/train/*')])
for i in range(len(class_list)):
    out_list.append([])
from tqdm import tqdm
for img in tqdm(imgs):
    idx += 1
    i = cv2.imread(img)
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    i = cv2.resize(i, (IMAGE_SIZE, IMAGE_SIZE))
    i = i / 255.
    i = i.astype('float32')
    data = torch.from_numpy(i)
    data = norm(data)
    data = data.unsqueeze(0)
    data = data.permute(0, 3, 1, 2)
    
    data = data.to(DEVICE)
    out = model(data)
    out = out.detach().cpu()#.numpy()
    out_list[class_list.index(img.split('/')[-2])].append(out)
    #if idx > 100:
    #    break
#output = torch.cat(out_list, 1)
output = out_list
new_out_list = []
for out in out_list:
    out = torch.cat(out, 0)
    out_mean = torch.mean(out, dim=0)
    new_out_list.append(out_mean)
print(new_out_list)

if not os.path.exists('pred_result'):
    os.makedirs('pred_result')
for cls in class_list:
    if os.path.exists(os.path.join('pred_result', cls)):
        shutil.rmtree(os.path.join('pred_result', cls))
    os.makedirs(os.path.join('pred_result', cls))

PREDICT_PATH = "/home/kgu/tmp4/datasets/merge"
pred_imgs = glob(PREDICT_PATH + '/*.jpg')
for img in tqdm(pred_imgs):
    i = cv2.imread(img)
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    i = cv2.resize(i, (IMAGE_SIZE, IMAGE_SIZE))
    i = i / 255.
    i = i.astype('float32')
    data = torch.from_numpy(i)
    data = norm(data)
    data = data.unsqueeze(0)
    data = data.permute(0, 3, 1, 2)

    data = data.to(DEVICE)
    out = model(data)
    #out = out.squeeze(0)
    out = out.detach().cpu()
    min_distance = 9e+10
    min_idx = -1
    for idx, new_out in enumerate(new_out_list):
        new_out = new_out.unsqueeze(0)
        cur_dist = F.pairwise_distance(new_out, out, keepdim=True)
        if min_distance > cur_dist:
            min_distance = cur_dist
            min_idx = idx
    pred_class = class_list[min_idx]
    shutil.copy(img, os.path.join('pred_result/' + pred_class, img.split('/')[-1]))



