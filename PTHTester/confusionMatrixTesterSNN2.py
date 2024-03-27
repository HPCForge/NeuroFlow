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
#from basicCNN import VGG_NN                                                     
import snntorch as snn

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
            transforms.Resize((1024/2,768/2))
        ]),
        'valid': transforms.Compose([
            #transforms.Resize(size=256),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Grayscale(),
            transforms.Resize((1024/2,768/2))
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
#img = Image.open("./FinalEventDataset/train/StratifiedSmooth/0out(20022)_image.png")
#img_t = image_transforms['test'](img)
#plt.imshow(img_t.permute(1,2,0))
#plt.show(block=True)
#exit()


# Train and Test Folders
dataset = '../Dataset/Processed/Event_Dataset/'
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
#print(train_data_size, test_data_size)
################################################################################


# Network Architecture
num_classes = 5

# Temporal Dynamics
num_steps = 10
beta = 0.95

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

        self.time_array = np.array([])

    def forward(self, x):
        starttime = time.time()
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
        #print((time.time()-starttime)/x.shape[0])
        self.time_array = np.append(self.time_array,(time.time()-starttime)/x.shape[0])
        #print(np.mean(self.time_array))
        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

y_pred = []                                                                     
y_true = []                                                                     

device = torch.device("cuda")

model = VGG_NN().to(device)
model.load_state_dict(torch.load("./eventModelSNNBasic15E55.pth",torch.device('cuda')))
model = model.to(device)
model.eval()

with torch.no_grad():
    num_correct = 0
    num_samples = 0
    for i, (x, y) in enumerate(tqdm(test_data_loader)):
        x = x.to(device)
        y = y.to(device)                                     
        spk_rec, _ = model(x)
        spike_count = spk_rec.sum(0)
        _, max_spike = spike_count.max(1)
        predictions = max_spike.cpu().numpy()
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
print(df_cm)

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2

print('model size: {:.3f}MB'.format(size_all_mb))
print(np.mean(model.time_array))
