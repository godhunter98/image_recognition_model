import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

class f_cnn(nn.Module):
  def __init__(self):
    super().__init__()

    # time for our 2 convolution layers
    self.cnl1 = nn.Conv2d(1,16,3,1)
    self.cnl2 = nn.Conv2d(16,32,3,1)
    self.cnl3 = nn.Conv2d(32,64,3,1)

    # pooling layers
    self.pool = nn.MaxPool2d(2,2)

    # now for our fully connected layers
    self.Fc1 = nn.Linear(64*5*5,200)
    self.Fc2 = nn.Linear(200,120)
    self.Fc3 = nn.Linear(120,60)
    self.Fc4 = nn.Linear(60,10)

    # our flattening layer
    self.flatten = nn.Flatten()

  def forward(self,x):
    x = F.relu(self.cnl1(x))
    x = F.relu(self.cnl2(x))
    x = self.pool(x)
    x = F.relu(self.cnl3(x))
    x = self.pool(x)

    # flattening our data
    x = self.flatten(x)

    # passing x to our FC layers
    x = F.relu(self.Fc1(x))
    x = F.relu(self.Fc2(x))
    x = F.relu(self.Fc3(x))

    # Final layer
    logits = self.Fc4(x)
    # out = F.softmax(x,dim=1)
    return logits

''' In the final layer, we've not applied a relu as the outputs raw magnitude are impoortant for
determining a models prediction, also softmax is autoapplied in cross-entropy.
Applying ReLU to the final output would zero out all negative values,
potentially losing important information about the model's confidence in certain classes.'''