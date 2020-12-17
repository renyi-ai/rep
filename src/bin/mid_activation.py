import sys
import os
import argparse
import torch
import pickle
from tqdm import tqdm
import numpy as np
import torchvision
from torchvision.transforms import transforms
from pytorch_lightning import Trainer

if './' not in sys.path:
    sys.path.append('./')

from src.bin import get_classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('classifier', type=str)
    parser.add_argument('index', type=int)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default='res/cifar10/data')
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args(args)

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

def run_model_on_data(model, data_loader):
    global device
    activations = []
    for x,y in tqdm(data_loader):
        x = x.to(device)
        batch_activation = list(model(x).detach().cpu().numpy())
        activations.extend(batch_activation)
    activations = np.array(activations)
    return activations


def get_model(args):
    global device
    model = get_classifier(args.classifier, pretrained=True)
    model.front(args.index)
    model.eval().to(device)
    return model

def save_activations(args, activations):
    os.makedirs(args.save_dir, exist_ok=True)
    filename = args.classifier+'__'+str(args.index)+'.pkl'
    save_path = os.path.join(args.save_dir, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(activations, f)

def main(args):
    args = _parse_args(args)
    data_loader = get_data_loader(args.data_dir, args.batch_size)
    
    # Front model
    model = get_model(args)

    # Run model on data
    activations = run_model_on_data(model, data_loader)

    # Save
    print('Saving..')
    save_activations(args, activations)

    print('Done.')

if __name__ == '__main__':
    main(sys.argv[1:])
