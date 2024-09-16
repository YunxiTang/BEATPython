import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from accelerate import Accelerator


class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = F.relu

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()


def parallel_train(model, train_loader, test_loader, optimizer):
    accelerator = Accelerator()
    device = accelerator.device
    print(device)

    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    
    for epoch in range(15):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = F.nll_loss(output, target)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        print(f'{epoch=}')

        # Evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        print(f'Accuracy: {100. * correct / len(test_loader.dataset)}')


def train(model, train_loader, test_loader, optimizer, device):
    for epoch in range(15):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            print(f'Accuracy: {100. * correct / len(test_loader.dataset)}')


if __name__ == '__main__':
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
    ])

    train_dset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)

    model = BasicNet()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    parallel_train(model, train_loader, test_loader, optimizer)

