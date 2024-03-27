import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_size = 256
sequence_length = 256
num_layers = 2
hidden_size = 256
num_classes = 5
learning_rate = 0.0001
batch_size = 64
num_epochs = 25

class BRNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0,c0))
        out = self.fc(out[:, -1, :])

        return out



############################## Dataset Setup ###################################

# Data Transforms
image_transforms = { 
        'train': transforms.Compose([
            transforms.Resize(size=(256,256)),
            #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            #transforms.RandomRotation(degrees=15),
            #transforms.RandomHorizontalFlip(),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Grayscale()
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=(256,256)),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Grayscale()
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=(256,256)),
            #transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Grayscale()
        ])
}


# Data Test
#img = Image.open("./FinalEventDataset/train/StratifiedSmooth/0out(20022)_image.png")
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
train_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
test_loader = DataLoader(data['test'], batch_size=bs, shuffle=True)
#print(train_data_size, test_data_size)
################################################################################




# Initialize network
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)
print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
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

        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

            data = data.to(device).squeeze(1)
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


# Check accuracy on training & test to see how good our model
def evaluate_model(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)
            _, predictions = torch.max(scores.data, 1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()


train_model(model, criterion, optimizer, num_epochs)
evaluate_model(model, train_loader)
evaluate_model(model, test_loader)
torch.save(model.state_dict(), 'eventModelLSTM.pth')
