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
import seaborn as sn                                                           
import pandas as pd                                                             
#from basicCNN import VGG_NN                                                     


############################## Dataset Setup ###################################

# Data Transforms
image_transforms = { 
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            #transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #transforms.Grayscale(),
            transforms.Resize((1024/2,768/2))
        ]),
        'valid': transforms.Compose([
            #transforms.Resize(size=256),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #transforms.Grayscale(),
            transforms.Resize((1024/2,768/2))
        ]),
        'test': transforms.Compose([
            #transforms.Resize(size=256),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Grayscale(),
            transforms.Resize((256,192))
            #transforms.Resize((240,128))
        ])
}


# Data Test
#img = Image.open("./FinalEventDataset/train/StratifiedSmooth/0out(20022)_image.png")
#img_t = image_transforms['test'](img)
#plt.imshow(img_t.permute(1,2,0))
#plt.show(block=True)
#exit()


# Train and Test Folders
#dataset = '../Documents/Research/Won Lab/Videos_Raw/FinalOpticalDataset/'
dataset = '../Dataset/Processed/Unified_Event_Dataset/'

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



class VGG_NN(nn.Module):                                                        
    def __init__(self, in_channels = 1, num_classes = 7):                       
        super(VGG_NN, self).__init__()                                          
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))            
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(32*64*48, num_classes)
        #self.fc1 = nn.Linear(32*60*32, num_classes)

        self.time_array = np.array([])
                                                                                
    def forward(self, x):
        starttime = time.time()
        x = F.relu(self.conv1(x))                                               
        #x = self.pool1(x)                                                       
        x = F.relu(self.conv2(x))                                               
        x = self.pool1(x)                                                       
        x = F.relu(self.conv3(x))
        x = self.pool1(x)                                                       
        x = x.reshape(x.shape[0], -1)                                           
        x = self.fc1(x)                                                         
        self.time_array = np.append(self.time_array,(time.time()-starttime)/x.shape[0])
        #print(time.time()-starttime)
        return x

y_pred = []                                                                     
y_true = []                                                                     

device = torch.device("cuda")

model = VGG_NN(in_channels=1).to(device)
model.load_state_dict(torch.load("./unifiedEventCNN.pth",torch.device('cuda')))
model = model.to(device)
model.eval()

with torch.no_grad():
    num_correct = 0
    num_samples = 0
    for i, (x, y) in enumerate(tqdm(test_data_loader)):                                     
        x = x.to(device)
        y = y.to(device)                                     
        scores = model(x)

        predictions = (torch.max(torch.exp(scores), 1)[1]).data.cpu().numpy()
        #print(predictions)
        y_pred.extend(predictions)
        
        y = y.data.cpu().numpy()
        y_true.extend(y)

        num_correct += (predictions == y).sum()
        num_samples += predictions.shape[0]

    print(f"Got {num_correct} / {num_samples}")


classes = ('A','B', 'EB', 'S','SS','SW','U')
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
df_cm = df_cm.reindex(["B", "EB", "S", "SS", "SW", "U", "A"])
df_cm = df_cm.reindex(columns=["B", "EB", "S", "SS", "SW", "U", "A"])
df_cm = df_cm.round(3)
plt.figure(figsize = (15,10))
plt.title('Event CNN', fontsize="40")
ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 34}, cbar=False, fmt=".2g")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
plt.savefig('unifiedCNNReordered.png')
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

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
