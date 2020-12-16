import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
from cifar10_models import *
from functools import partial
import collections

def get_classifier(classifier, pretrained, start_idx=None):
    if classifier == 'vgg11_bn':
        return vgg11_bn(pretrained=pretrained)
    elif classifier == 'vgg13_bn':
        return vgg13_bn(pretrained=pretrained)
    elif classifier == 'vgg16_bn':
        return vgg16_bn(pretrained=pretrained, start_idx=start_idx)
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
    else:
        raise NameError('Please enter a valid classifier')
        
class CIFAR10_Module(pl.LightningModule):
    def __init__(self, hparams, pretrained=False, start_idx=0):
        super().__init__()
        self.val_x = None
        self.val_y = None
        self.hparams = hparams
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.model = get_classifier(hparams.classifier, pretrained, start_idx=start_idx)
        self.train_size = len(self.train_dataloader().dataset)
        self.val_size = len(self.val_dataloader().dataset)

        self.model.trim_until(start_idx)

        self.activations = collections.defaultdict(list)
        def save_activation(name, mod, inp, out):
            self.activations[name].append(out.cpu())

        for name, m in self.model.named_modules():
            if type(m)==torch.nn.modules.activation.ReLU:
                # partial to assign the layer name to each hook
                print(name)
                m.register_forward_hook(partial(save_activation, name))

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = torch.sum(torch.max(predictions, 1)[1] == labels.data).float() / batch[0].size(0)
        return loss, accuracy
    
    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        logs = {'loss/train': loss, 'accuracy/train': accuracy}
        return {'loss': loss, 'log': logs}
        
    def validation_step(self, batch, batch_nb):
        avg_loss, accuracy = self.forward(batch)
        loss = avg_loss * batch[0].size(0)
        corrects = accuracy * batch[0].size(0)
        logs = {'loss/val': loss, 'corrects': corrects}
        return logs
                
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss/val'] for x in outputs]).sum() / self.val_size
        accuracy = torch.stack([x['corrects'] for x in outputs]).sum() / self.val_size
        logs = {'loss/val': loss, 'accuracy/val': accuracy}
        return {'val_loss': loss, 'log': logs}
    
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)
    
    def test_epoch_end(self, outputs):
        accuracy = self.validation_epoch_end(outputs)['log']['accuracy/val']
        accuracy = round((100 * accuracy).item(), 2)
        return {'progress_bar': {'Accuracy': accuracy}}
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_decay, momentum=0.9, nesterov=True)
            
        scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, 
                                                                     steps_per_epoch=self.train_size//self.hparams.batch_size,
                                                                     epochs=self.hparams.max_epochs),
                     'interval': 'step', 'name': 'learning_rate'}
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform_train, download=True)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        if self.val_x is not None:
            val_x_t = torch.Tensor(self.val_x)
            val_y_t = torch.LongTensor(self.val_y)
            dataloader = DataLoader(TensorDataset(val_x_t, val_y_t), batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True)
            return dataloader

        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform_val, download=True)

        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True)
        return dataloader
    
    def test_dataloader(self):
        return self.val_dataloader()