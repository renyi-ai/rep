import sys
import os
import argparse

if './' not in sys.path:
    sys.path.append('./')

from src.bin import get_model, save_activations
from src.utils.front import get_data_loader, run_model_on_data_loader, get_labels_from_data_loader
from src.utils.end import run_model_on_data as rmod_end
from src.utils.common import str2bool

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('classifier', type=str)
    parser.add_argument('index', type=int)
    parser.add_argument('--output-filename', '-o', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default='res/cifar10/data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--with_gradients', type=str2bool, nargs='?', const=True, default=False, help='Save gradients.')
    return parser.parse_args(args)

def main(args):
    args = _parse_args(args)
    data_loader = get_data_loader(args.data_dir, args.batch_size)
    
    # Front model
    model = get_model(args)

    # Run model on data
    activations = run_model_on_data_loader(model, data_loader)

    # Return or save
    print('Saving vector of shape {}..'.format(activations.shape))
    save_activations(args, activations, prefix='front_')


    if args.with_gradients:
        # Back model
        model = get_model(args, front=False)

        # Run model on data
        activations, grads = rmod_end(model, activations, with_grads=True, data_labels=get_labels_from_data_loader(data_loader), batch_size=1)

        # Return or save
        print('Saving grad vector of shape {}..'.format(grads.shape))
        save_activations(args, grads, prefix='front_grads_')

    print('Done.')

if __name__ == '__main__':
    main(sys.argv[1:])
