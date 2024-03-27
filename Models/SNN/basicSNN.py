import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#from torch.utils.tensorboard import SummaryWriter
from torchvision.models import alexnet, convnext_base

import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

#import albumentations as A
#from albumentations.pytorch import ToTensorV2
#import cv2
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
import os
import time
from PIL import Image

# imports
import snntorch as snn
from snntorch import spikeplot as splt
#import torch

############################## Dataset Setup ###################################

# Data Transforms
image_transforms = { 
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            #transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Grayscale(),
            transforms.Resize((256,192))
        ]),
        'valid': transforms.Compose([
            #transforms.Resize(size=256),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Grayscale(),
            transforms.Resize((256,192))
        ]),
        'test': transforms.Compose([
            #transforms.Resize(size=256),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Grayscale(),
            transforms.Resize((256,192))
        ])
}


# Data Test
#img = Image.open("./Dataset/Processed/Optical_Dataset/FinalOpticalDataset/test/StratifiedSmooth/frame(10)112.png")
#img_t = image_transforms['test'](img)
#plt.imshow(img_t.permute(1,2,0))
#plt.show(block=True)
#exit()


# Train and Test Folders
dataset = './Dataset/Processed/Event_Dataset'
train_directory = os.path.join(dataset, 'train')
test_directory = os.path.join(dataset, 'test')


# Batch size
bs = 128


# Class Information
num_classes = len(os.listdir(train_directory))


# Get Data
data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}


# Debug Info
#idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
#print(idx_to_class)


# Train and Test Sizes
train_data_size = len(data['train'])
test_data_size = len(data['test'])


# Data Loaders
train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
test_data_loader = DataLoader(data['test'], batch_size=bs, shuffle=True)
#print(train_data_size, test_data_size)
################################################################################





# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





dtype = torch.float








########################## Model Setup #########################################


# Network Architecture
num_classes = 5

# Temporal Dynamics
num_steps = 15
beta = 0.95

# Define Network
class VGG_NN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 5):
        super(VGG_NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.lif1 = snn.Leaky(beta=beta)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.lif2 = snn.Leaky(beta=beta)
        self.fc1 = nn.Linear(16*64*48, num_classes)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        for step in range(num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(self.pool1(cur1), mem1)
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.pool1(cur2), mem2)
            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
        
# Load the network onto CUDA if available
convnet = VGG_NN(in_channels=1, num_classes=num_classes)
convnet = convnet.to(device)


loss_weights = 40508.0/ torch.tensor([18032, 6081, 3721, 10203, 2471])

loss = nn.CrossEntropyLoss(weight=loss_weights).to(device)
optimizer = torch.optim.Adam(convnet.parameters(), lr=1e-4, betas=(0.9, 0.999))

num_epochs = 55
loss_hist = []
test_loss_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):
    losses = []
    accuracies = []
    epoch_start = time.time()
    running_loss = 0.0

    print(f"Epoch: {epoch+1}/{num_epochs}")

    # Set to training mode
    convnet.train()

    # Minibatch training loop
    for batch_dix, (data, targets) in enumerate(tqdm(train_data_loader)):
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        spk_rec, _ = convnet(data)

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        loss_val = loss(spk_rec.sum(0), targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

    mean_loss = sum(loss_hist)/len(loss_hist)
    print(f"loss at epoch {epoch} was {mean_loss:.5f}")  


def measure_accuracy(model, dataloader):
  with torch.no_grad():
    model.eval()
    running_length = 0
    running_accuracy = 0

    for data, targets in iter(dataloader):
      data = data.to(device)
      targets = targets.to(device)

      # forward-pass
      spk_rec, _ = model(data)
      spike_count = spk_rec.sum(0)
      _, max_spike = spike_count.max(1)

      # correct classes for one batch
      num_correct = (max_spike == targets).sum()

      # total accuracy
      running_length += len(targets)
      running_accuracy += num_correct
    
    accuracy = (running_accuracy / running_length)

    return accuracy.item()

print(f"Test set accuracy: {measure_accuracy(convnet, test_data_loader)}")
torch.save(convnet.state_dict(), 'eventModelSNNBasic15E55.pth')
