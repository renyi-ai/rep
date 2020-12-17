import os
import sys
import shutil
import torch
import torchvision
from torchvision.transforms import transforms
from argparse import ArgumentParser
from pytorch_lightning import Trainer

if './' not in sys.path:
    sys.path.append('./')

from . import CIFAR10_Module


def main(hparams):
    # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
    if type(hparams.gpus) == str:
        if len(hparams.gpus) == 2:  # GPU number and comma e.g. '0,' or '1,'
            torch.cuda.set_device(int(hparams.gpus[0]))

    transform_val = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root=hparams.data_dir, train=False, download=True, transform=transform_val)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=hparams.batch_size, num_workers=4, pin_memory=True)

    all_labels = []
    for i, data in enumerate(dataloader, 0):
        images, labels = data
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, 0)

    model = CIFAR10_Module(hparams, pretrained=True)
    trainer = Trainer(gpus=hparams.gpus, default_save_path=os.path.join(
        os.getcwd(), 'test_temp'))
    trainer.test(model)

    layer_acts = []
    for k, v in model.activations.items():
        acts = torch.cat(v, 0)
        layer_acts.append(acts)
    acts = layer_acts[1]
    print(acts.size())
    print(all_labels.size())

    model = CIFAR10_Module(hparams, pretrained=True, start_idx=2)
    model.val_x = acts
    model.val_y = all_labels

    trainer = Trainer(gpus=hparams.gpus, default_save_path=os.path.join(
        os.getcwd(), 'test_temp'))
    trainer.test(model)

    shutil.rmtree(os.path.join(os.getcwd(), 'test_temp'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='/data/huy/cifar10/')
    parser.add_argument('--gpus', default='0,')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    args = parser.parse_args()
    main(args)
