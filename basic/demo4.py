import os 
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

#超参数定义
# 批次的大小
batch_size = 16 #可选32、64、128
# 优化器的学习率
lr = 1e-4
#运行epoch
max_epochs = 10

# cpu 加速就是 'cpu'
# 第 1 块 gpu 加速就是 'cuda:0'，第 2 块 gpu 加速就是 'cuda:1'，以此类推...
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample1():
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(root=os.path.join(os.getcwd(), "dataset"), train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=os.path.join(os.getcwd(), "dataset"), train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    

if __name__ == '__main__':
    
    sample1()

