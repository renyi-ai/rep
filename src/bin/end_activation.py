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

from src.bin import get_model, save_activations
from src.utils.end import load_activations, run_model_on_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('classifier', type=str)
    parser.add_argument('index', type=int)
    parser.add_argument('input_file', type=str)
    parser.add_argument('--output-filename', '-o', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default='res/cifar10/data')
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args(args)

def main(args):

    # Get activations
    args = _parse_args(args)
    in_data = load_activations(args.input_file)
    
    # Front model
    model = get_model(args, front=False)

    # Run model on data
    predictions = run_model_on_data(model, in_data)

    # Save
    print('Saving..')
    save_activations(args, predictions, prefix='end_')
    print('Done.')

if __name__ == '__main__':
    main(sys.argv[1:])
