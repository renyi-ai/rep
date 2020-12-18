import os
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.bin import device


def get_transformation():
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
    return trans

def get_data_loader(data_dir, batch_size):
    trans = get_transformation()
    dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=trans)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              num_workers=4, pin_memory=True)
    return data_loader

def run_model_on_data_loader(model, data_loader, verbose=True):
    activations = []
    data_loader = tqdm(data_loader) if verbose else data_loader
    for x,y in data_loader:
        x = x.to(device)
        batch_activation = list(model(x).detach().cpu().numpy())
        activations.extend(batch_activation)
    activations = np.array(activations)
    return activations
