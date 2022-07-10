import torch 
import cv2
from tqdm import tqdm
import utils


class MNIST784Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()            

        image = self.X[idx]
        label = self.y[idx]
        if self.transform:
            _out = self.transform(image=image)
            image = _out['image']         

        return image, label
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=50, output_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)  
        )
    
    def forward(self, x):
        return self.layers(x)
    
    
def train(model, criterion, optimizer, dataloader_train, dataloader_test, epochs=200):
    early_stopping = utils.EarlyStopping(model=model, delta=0.001, verbose=False)
    num_epochs = epochs
    train_history = {'train_loss':[], 'test_loss':[], 'train_acc':[], 'test_acc':[]}

    for epoch in range(num_epochs):
        train_loss, test_loss, train_acc, test_acc = 0., 0., 0., 0.

        for images, label_idxs in dataloader_train:
            optimizer.zero_grad()
            images, label_idxs = images.float().to('cuda'), label_idxs.to('cuda')
            output = model(images)
            batch_loss = criterion(output, label_idxs)
            train_loss += batch_loss / len(dataloader_train)
            train_acc += utils.accuracy(output.argmax(1).cpu().numpy(), label_idxs.cpu().numpy()) / len(dataloader_train)
            batch_loss.backward()
            optimizer.step()


        for images, label_idxs in dataloader_test:
            with torch.no_grad():
                images, label_idxs = images.float().to('cuda'), label_idxs.to('cuda')
                output = model(images)
                batch_loss = criterion(output, label_idxs)
                test_loss += batch_loss / len(dataloader_test)
                test_acc += utils.accuracy(output.argmax(1).cpu().numpy(), label_idxs.cpu().numpy()) / len(dataloader_test)

        print(f'Epoch: {epoch:03d}/{num_epochs:03d} '
              f'| Test loss: {train_loss.cpu().detach().numpy():.2f} | Test loss: {test_loss.cpu().detach().numpy():.2f}'
              f'| Train acc: {train_acc:.3f} | Test acc: {test_acc:.3f}')
        train_history['train_loss'].append(float(train_loss.cpu().detach()))
        train_history['test_loss'].append(float(test_loss.cpu().detach()))
        train_history['train_acc'].append(train_acc)
        train_history['test_acc'].append(test_acc)

        early_stopping(test_acc)
        if early_stopping.early_stop:
            model.state_dict = early_stopping.best_model_checkpoint
            break
    return train_history


class MNIST784Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()            

        image = self.X[idx]
        label = self.y[idx]
        if self.transform:
            _out = self.transform(image=image)
            image = _out['image']         

        return image, label