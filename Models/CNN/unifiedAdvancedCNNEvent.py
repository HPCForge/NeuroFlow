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
dataset = '../Dataset/Processed/Unified_Event_Dataset'
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
idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
print(idx_to_class)


# Train and Test Sizes
train_data_size = len(data['train'])
test_data_size = len(data['test'])


# Data Loaders
train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
test_data_loader = DataLoader(data['test'], batch_size=bs, shuffle=True)
print(train_data_size, test_data_size)
################################################################################





# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




########################## Model Setup #########################################
# Conv Layers

class VGG_NN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 7):
        super(VGG_NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        #self.fc1 = nn.Linear(16*256*192, num_classes)
        #self.fc1 = nn.Linear(16*128*96, num_classes)
        self.fc1 = nn.Linear(32*64*48, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
#########################################
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
#########################################
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

################################################################################


model = VGG_NN(in_channels=1, num_classes=num_classes)
model = model.to(device)


# Optimizer and Weighted Loss
weight_tensor = 50502/ torch.tensor([18032, 8759, 1235, 6081, 3721, 10203, 2471])
criterion = nn.CrossEntropyLoss(weight=weight_tensor).to(device)
optimizer = optim.Adam(model.parameters())
#print(optimizer)





def train_model(model, criterion, optimizer, epochs):
    start = time.time()

    for epoch in range(epochs):
        losses = []
        accuracies = []
        epoch_start = time.time()
        running_loss = 0.0

        print(f"Epoch: {epoch+1}/{epochs}")

        # Set to training mode
        model.train()

        for batch_idx, (data, targets) in enumerate(tqdm(train_data_loader)):

            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predictions = torch.max(scores.data, 1)
            num_correct = (predictions == targets).sum()
            running_acc = float(num_correct)/float(data.shape[0])

            #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(batch_idx, loss.item(), running_acc))

        mean_loss = sum(losses)/len(losses)
        print(f"loss at epoch {epoch} was {mean_loss:.5f}")


def evaluate_model(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = torch.max(scores.data, 1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()


num_epochs = 25

train_model(model, criterion, optimizer, num_epochs)
evaluate_model(model, train_data_loader)
evaluate_model(model, test_data_loader)
torch.save(model.state_dict(), 'unifiedEventCNN.pth')
