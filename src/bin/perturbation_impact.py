import sys
import os
import argparse

if './' not in sys.path:
    sys.path.append('./')

from src.bin import device, get_classifier, get_model, save_activations
from src.utils.front import get_data_loader, run_model_on_data_loader
from src.utils.end import run_model_on_data
from src.functions import get_comparator, get_manipulator

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('classifier', type=str)
    parser.add_argument('index', type=int)
    parser.add_argument('manipulator', type=str)
    parser.add_argument('comparator', type=str)
    parser.add_argument('--output-filename', '-o', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default='res/cifar10/data')
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args(args)

def get_full_model(args):
    full_model = get_classifier(args.classifier, pretrained=True)
    full_model.eval().to(device)
    return full_model

def main(args):
    args = _parse_args(args)

    print('1) Defining models.. ', end='')
    data_loader = get_data_loader(args.data_dir, args.batch_size)
    print('done.')
    
    print('2) Defining models.. ', end='')
    full_model = get_full_model(args)
    front_model = get_model(args, front=True)
    end_model = get_model(args, front=False)
    print('done.')

    #import pdb; pdb.set_trace()

    print('3) Calculating true logits.. ', end='')
    true_logits = run_model_on_data_loader(full_model, data_loader, verbose=False)
    print('done.')

    data_loader = get_data_loader(args.data_dir, args.batch_size)

    print('4) Running perturbation')

    print('   (a) Retrieving front model\'s activations.. ', end='')
    activations = run_model_on_data_loader(front_model, data_loader, verbose=False)
    print('done.')

    print('   (b) Applying custom activation manipulation.. ', end='')
    manipulation = get_manipulator(args.manipulator)(activations)
    print('done.')

    print('   (c) Retrieving logits after manipulation.. ', end='')
    y_hat = run_model_on_data(end_model, manipulation, verbose=False)
    print('done.')

    print('5) Running comparison..', end='')
    compare_value = get_comparator(args.comparator)(true_logits, y_hat)
    print('done.')

    print()
    print('-------------------------------------------------------------')
    print('Result: {}'.format(compare_value))

if __name__ == '__main__':
    main(sys.argv[1:])
