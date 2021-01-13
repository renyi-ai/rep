import torch
import torch.nn as nn
import os
from copy import deepcopy

__all__ = ['lenet']

class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

class CutLeNet(LeNet5):
        
    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.orig_features = deepcopy(self.features)
        self.n_features = len(list(self.orig_features.children()))
        self._start = 0
        self._end = self.n_features

    # ================================================================
    # PUBLIC
    # ================================================================

    @property
    def layer_info(self):
        layer_info = list(self.orig_features.children())
        layer_info = [str(x) for x in layer_info]
        return layer_info

    def front(self, idx):
        self._cut(start=0, end=idx)

    def end(self, idx):
        self._cut(start=idx, end=self.n_features)

    # ================================================================
    # PRIVATE
    # ================================================================
        
    def _cut(self, start, end):
        self._start = start
        self._end = end
        children_list = list(self.orig_features.children())
        chosen_children = children_list[start:end]
        self.features = torch.nn.Sequential(*chosen_children)

    def forward(self, x):
        is_end = self._end == self.n_features
        end_side_network = super().forward
        front_side_network = self.features
        return end_side_network(x) if is_end else front_side_network(x)




def lenet():
    return CutLeNet()