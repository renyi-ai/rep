import sys
import os
import torch
import pickle

if './' not in sys.path:
    sys.path.append('./')

from ..models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(args, front=True):
    global device
    model = get_classifier(args.classifier, pretrained=True)
    if front:
        model.front(args.index)
    else:
        model.end(args.index)
    model.eval().to(device)
    return model

def save_activations(args, activations, prefix=''):
    os.makedirs(args.save_dir, exist_ok=True)
    if args.output_filename is None:
        filename = prefix+args.classifier+'__'+str(args.index)+'.pkl'
    else:
        filename = args.output_filename
    save_path = os.path.join(args.save_dir, prefix + filename)
    with open(save_path, 'wb') as f:
        pickle.dump(activations, f)

def get_classifier(classifier, pretrained):
    if classifier == 'vgg11_bn':
        return vgg11_bn(pretrained=pretrained)
    elif classifier == 'vgg13_bn':
        return vgg13_bn(pretrained=pretrained)
    elif classifier == 'vgg16_bn':
        return vgg16_bn(pretrained=pretrained)
    elif classifier == 'vgg19_bn':
        return vgg19_bn(pretrained=pretrained)
    elif classifier == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif classifier == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif classifier == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif classifier == 'densenet121':
        return densenet121(pretrained=pretrained)
    elif classifier == 'densenet161':
        return densenet161(pretrained=pretrained)
    elif classifier == 'densenet169':
        return densenet169(pretrained=pretrained)
    elif classifier == 'mobilenet_v2':
        return mobilenet_v2(pretrained=pretrained)
    elif classifier == 'googlenet':
        return googlenet(pretrained=pretrained)
    elif classifier == 'inception_v3':
        return inception_v3(pretrained=pretrained)
    elif classifier == 'lenet':
        return lenet()
    else:
        raise NameError('Please enter a valid classifier')


