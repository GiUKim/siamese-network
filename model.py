import torch
from torch import nn

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

