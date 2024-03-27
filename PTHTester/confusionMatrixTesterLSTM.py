import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import glob
from pandas.core.common import flatten
import copy
import random
import pandas as pd

import numpy as np
from sklearn.metrics import confusion_matrix                                    
import time
import matplotlib.pyplot as plt
import seaborn as sn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


sequence_len = 5000
input_len = 3
hidden_size = 128
num_layers = 3
num_classes = 5
num_epochs = 15
learning_rate = 0.001

batchsize = 128


############################## Dataset Setup ###################################

train_data_path = '/pub/sarani/KDD/Dataset/Processed/LSTMDataset/output/train'
test_data_path = '/pub/sarani/KDD/Dataset/Processed/LSTMDataset/output/test'

train_image_paths = [] #to store image paths in list
classes = [] #to store class values

#1.
# get all the paths from train_data_path and append image paths and class to to respective lists
# eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
# eg. class -> 26.Pont_du_Gard
for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('/')[-1])
    train_image_paths.append(glob.glob(data_path + '/*'))

train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)

#print('train_image_path example: ', train_image_paths[0])
#print('class example: ', classes[0])
#2.
# split train valid from train paths (80,20)
train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):]

#3.
# create the test_image_paths
test_image_paths = []
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))

test_image_paths = list(flatten(test_image_paths))

#print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))

idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

#print(idx_to_class)
#print(class_to_idx)

class BubbleDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = pd.read_csv(image_filepath)
        image = image.iloc[:,1:]
        image = torch.tensor(image.values)
        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]

        return image, label

train_dataset = BubbleDataset(train_image_paths)
valid_dataset = BubbleDataset(valid_image_paths) #test transforms are applied
test_dataset = BubbleDataset(test_image_paths)

print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
print('The label for 50th image in train dataset: ',train_dataset[49][1])


train_loader = DataLoader(train_dataset, batch_size=batchsize)
test_loader = DataLoader(test_dataset, batch_size=batchsize)



class LSTM(nn.Module):
    def __init__(self,input_len, hidden_size, num_classes, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)

        self.time_array = np.array([])


    def forward(self, x):
        starttime = time.time()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0,c0))
        out = self.output_layer(out[:, -1, :])

        self.time_array = np.append(self.time_array,(time.time()-starttime)/x.shape[0])

        return out


model = LSTM(input_len, hidden_size, num_classes, num_layers)
#model.load_state_dict(torch.load("../eventLSTMl31e3Weighted.pth",torch.device('cuda')))
model.load_state_dict(torch.load("../eventLSTMl31e3.pth",torch.device('cuda')))
model = model.to(device)
#print(model)

y_pred = []                                                                     
y_true = []                                                                     

model.eval()
print(model)

with torch.no_grad():
    num_correct = 0
    num_samples = 0
    for i, (x, y) in enumerate(tqdm(test_loader)):                                     
        x = x.float()
        x = x.reshape(-1, sequence_len, input_len)
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


classes = ('A','S','SS','SW','U')
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
df_cm = df_cm.reindex(["S", "SS", "SW", "U", "A"])
df_cm = df_cm.reindex(columns=["S", "SS", "SW", "U", "A"])

plt.figure(figsize = (15,10))
plt.title('Event LSTM', fontsize="40")
ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 37}, cbar=False, fmt=".2g")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
plt.savefig('eventLSTMReordered')
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
