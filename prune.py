import model_loader
import torch
import torch.nn as nn
from torch.nn.utils import prune


def prune_weights(net, method, amount):
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
             prune_weights_method(module, amount, method)
    return net


def prune_weights_method(module, amount, method):
    if method == 'l1':
        prune.l1_unstructured(module, 'weight', amount)
        prune.remove(module, "weight")
    elif method == 'random':
        prune.random_unstructured(module, 'weight', amount)
        prune.remove(module, "weight")
