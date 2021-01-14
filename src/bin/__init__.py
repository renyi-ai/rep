import sys
import os
import torch
import pickle

if './' not in sys.path:
    sys.path.append('./')

from ..models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(classifier, n_iter, cut_index, front=True):
    global device
    model = get_classifier(classifier, pretrained=True, n_iter=n_iter)
    if front:
        model.front(cut_index)
    else:
        model.end(cut_index)
    model.eval().to(device)
    return model

def save_activations(args, activations, prefix=''):
    os.makedirs(args.save_dir, exist_ok=True)
    if args.output_filename is None:
        filename = prefix+args.classifier+'__'+str(args.index)+'.pkl'
    else:
        filename = str(args.output_filename)
    save_path = os.path.join(args.save_dir, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(activations, f)

def get_classifier(classifier, pretrained, n_iter=None):
    if classifier == 'vgg11_bn':
        return vgg11_bn(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'vgg13_bn':
        return vgg13_bn(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'vgg16_bn':
        return vgg16_bn(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'vgg19_bn':
        return vgg19_bn(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'resnet18':
        return resnet18(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'resnet34':
        return resnet34(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'resnet50':
        return resnet50(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'densenet121':
        return densenet121(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'densenet161':
        return densenet161(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'densenet169':
        return densenet169(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'mobilenet_v2':
        return mobilenet_v2(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'googlenet':
        return googlenet(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'inception_v3':
        return inception_v3(n_iter=n_iter, pretrained=pretrained)
    elif classifier == 'lenet':
        return lenet(n_iter=n_iter)
    else:
        raise NameError('Please enter a valid classifier')


