import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pkbar
import torchvision
import torch

if './' not in sys.path:
    sys.path.append('./')

from src.bin import device, get_classifier
from src.utils.front import get_data_loader_from_dataset
from src.utils.common import get_transformation, get_aug_transformation


def _accuracy(labels, preds):
    equality = torch.sum(labels == preds)
    accuracy = equality / labels.nelement()
    return accuracy

def _get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _init_metrics():
    return {'loss':0., 'accuracy':0.}

def _batch_metrics(labels, preds, loss):
    data = {}
    data['loss'] = loss.item()
    data['accuracy'] = _accuracy(labels, preds)
    return data

def _update_running_metrics(orig, new):
    for k, v in new.items():
        orig[k] += new[k]
    return orig

def get_model(classifier, device):
    model = get_classifier(classifier, pretrained=True)
    model.train().to(device)
    return model

# def create_train_directory(save_folder_root, classifier):
#     # Create new folder with current date
#     save_folder = os.path.join(save_folder_root, classifier)
#     os.makedirs(save_folder, exist_ok=True)
#     return save_folder

def main(args):

    n_iterations_ran = 0
    device = _get_device()
    stats = pd.DataFrame()
    save_folder = args.save_dir #create_train_directory(args.save_dir, args.model)

    # Define just for initialization + download if needed
    trans = get_transformation()
    aug_trans = get_aug_transformation()
    datasets = {
        'train' : torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                           download=True, transform=aug_trans),
        'val' : torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                           download=True, transform=trans)
    }
    data_loaders = {
        'train' : None, # defining it later
        'val' : get_data_loader_from_dataset(datasets['val'], args.batch_size)
    }
    model = get_model(args.model, device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)

    epoch = 0

    # Exit when desired number of iterations reached
    while n_iterations_ran < args.n_iter:

        # Dict containing losses and accuracies
        epoch_data = {}

        # Init data loader with new seed
        data_loaders['train'] = get_data_loader_from_dataset(datasets['train'], args.batch_size, seed=epoch)
        
        # Progress bar 
        kbar_n_iter = len(data_loaders['train'].dataset) // data_loaders['train'].batch_size
        kbar = pkbar.Kbar(target=kbar_n_iter+1,
                            epoch=epoch,
                            width=8,
                            always_stateful=False)

        for phase in ['train', 'val']:

            # Make every running metric zero
            running_metrics =_init_metrics()
            # Use either training or validation data loader
            data_loader = data_loaders[phase]
            n_iter = len(data_loader.dataset) // data_loader.batch_size

            # Iterate through data in batches
            for i, (inputs, labels) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.max(outputs.data, 1)[1]
                    new_metrics = _batch_metrics(labels, preds, loss)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update loss & acc for epoch end
                running_metrics = _update_running_metrics(running_metrics, new_metrics)

                # Update progress bar
                new_values = [(phase + '_' + name, value)
                                for (name, value) in new_metrics.items()]
                

                # Save model checkpoint
                if phase == 'train':
                    kbar.update(i, values=new_values)
                    n_iterations_ran += 1
                    if n_iterations_ran % args.save_frequency == 0:
                        filename = args.model+'--'+str(n_iterations_ran) + '.pt'
                        path = os.path.join(save_folder, filename)
                        torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)

                if n_iterations_ran >= args.n_iter:
                    break

                # Save information about this epoch
                for name, value in running_metrics.items():
                    if name != 'loss':
                        value = value.detach().cpu().numpy()
                    epoch_data[phase + '_' + name] = value / n_iter

            if phase=='val':
                val_values = [(x,y) for (x,y) in epoch_data.items() if x.startswith('val_')]
                kbar.add(1, values=val_values)

        if n_iterations_ran < args.n_iter:
            stats = stats.append(epoch_data, ignore_index=True)
            save_path = os.path.join(save_folder, args.model+'-training_log.csv')
            stats.to_csv(save_path, index=False)

        epoch += 1

    print()
    print('{} iterations done.'.format(args.n_iter))
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m','--model', type=str, default='vgg16_bn')
    parser.add_argument('-n', '--n-iter', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('-b','--batch_size', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='res/cifar10/data')
    parser.add_argument('--gpus', default='0,') # use None to train on CPU
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    # Added by Gergo
    parser.add_argument('-o','--save-dir', type=str, default='res/cifar10/models/')
    parser.add_argument('-s','--save-frequency', type=int, default=50)
    args = parser.parse_args()
    main(args)
