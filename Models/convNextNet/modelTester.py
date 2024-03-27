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
from sklearn.metrics import confusion_matrix                                    
#import seaborn as sn                                                           
import pandas as pd    


############################## Dataset Setup ###################################

# Data Transforms
image_transforms = { 
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.Resize(size=256),
            #transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Grayscale()
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Grayscale()
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Grayscale()
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
}


# Data Test
#img = Image.open("./Dataset/Processed/Optical_Dataset/FinalOpticalDataset/test/StratifiedSmooth/frame(10)112.png")
#img_t = image_transforms['test'](img)
#plt.imshow(img_t.permute(1,2,0))
#plt.show(block=True)
#exit()


# Train and Test Folders
dataset = '../Dataset/Processed/Event_Dataset'
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




########################## Model Setup #########################################

#Weights~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#model = convnext_base(pretrained=True).to(device)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model = alexnet(pretrained=False).to(device)

# Freeze Parameters for Transfer Learning
for param in model.parameters():
    param.requires_grad = False



# Modifying Model for Transfer Learning with different num_classes
model.classifier[6] = nn.Linear(4096, num_classes)
model.classifier.add_module("7", nn.LogSoftmax(dim = 1))
#print(model)
################################################################################



model = model.to(device)

#train_model(model, criterion, optimizer, num_epochs)
#evaluate_model(model, train_data_loader)
#evaluate_model(model, test_data_loader)
#torch.save(model.state_dict(), 'eventAlexnetWeighted.pth')

model.load_state_dict(torch.load("./eventAlexnet.pth",torch.device('cuda')))
model = model.to(device)
model.eval()
y_pred = []                                                                     
y_true = [] 

time_array = np.array([])


with torch.no_grad():
    num_correct = 0
    num_samples = 0
    for i, (x, y) in enumerate(tqdm(test_data_loader)):                                     
        x = x.to(device)
        y = y.to(device)
        starttime = time.time()                         
        scores = model(x)
        print(x.shape[0])
        time_array = np.append(time_array,(time.time()-starttime)/x.shape[0])

        predictions = (torch.max(torch.exp(scores), 1)[1]).data.cpu().numpy()
        #print(predictions)
        y_pred.extend(predictions)
        
        y = y.data.cpu().numpy()
        y_true.extend(y)

        num_correct += (predictions == y).sum()
        num_samples += predictions.shape[0]

    print(f"Got {num_correct} / {num_samples}")


classes = ('Annular','Slug','StratifiedSmooth','StratifiedWavy','Unstable')
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
#plt.figure(figsize = (12,7))
#sn.heatmap(df_cm, annot=True)
#plt.savefig('output.png')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
print(df_cm)

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2

print('model size: {:.3f}MB'.format(size_all_mb))
print(np.mean(time_array))

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
