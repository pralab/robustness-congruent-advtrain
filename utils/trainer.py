from tqdm import tqdm
import torch
from torch import nn
import numpy as np


def train_epoch(model, device, train_loader, optimizer, epoch, loss_fn):
    model = model.to(device)
    model.train()

    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            try:
                loss = loss_fn(output, target)
            except:
                loss = loss_fn(output, target.type(torch.LongTensor).to(device))
            loss.backward()
            optimizer.step()

            t.set_postfix(
                epoch='{}'.format(epoch),
                completed='[{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)),
                loss='{:.4f}'.format(loss.item()))
            t.update()
    return

def pc_train_epoch(model, device, train_loader, optimizer, epoch, loss_fn):
    model = model.to(device)
    model.train()
    batch_size = train_loader.batch_size

    ce_cumul = []
    pc_cumul = []
    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target, batch_idx, batch_size, data.shape[0])
            loss[0].backward()
            optimizer.step()

            # ce_cumul.append(loss[1].item())
            # pc_cumul.append(loss[2].item())

            t.set_postfix(
                epoch='{}'.format(epoch),
                compl='[{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)),
                loss_ce='{:.4f}'.format(loss[1].item()),
                loss_pc='{:.4f}'.format(loss[2].item()))
                # loss_ce='{:.4f}'.format(np.array(ce_cumul).mean()),
                # loss_pc='{:.4f}'.format(np.array(pc_cumul).mean()))
            t.update()
    return

def adv_train_epoch():
    pass


def freeze_network(model, n_layer=1):
    max_id = len(list(model.children())) - n_layer

    for i, child in enumerate(model.children()):
        if i < max_id:
            for param in child.parameters():
                param.requires_grad = False


class MLP(nn.Module):
    def __init__(self, input_size=784, output_size=10, hidden_units=120):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_units)
        self.l2 = nn.Linear(hidden_units, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.shape[0], 28*28)
        x = self.relu(self.l1(x))
        return self.relu(self.l2(x))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output