# efficientnet_utils.py - pytorch adaptation
# adapted from tensorflow model_utils.py
# copyright 2019 the tensorflow authors, adapted for pytorch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os  # added: filesystem access for weight loading


# drop connect is implemented in the efficientnetModel.py


# Batch Normalization

class BatchNorm2d(nn.BatchNorm2d):
    """standard batch normalization wrapper"""
    # adapted: replaced tf.layers.BatchNormalization / TPU batchnorm with pytorch batchnorm
    # removed: cross-replica / distributed TPU-specific logic
    def __init__(self, num_features, eps=1e-3, momentum=0.99, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)


# Conv2d wrapper

class Conv2d(nn.Conv2d):
    """conv2d with optional activation function"""
    # adapted: keep pytorch nn.Conv2d, add swish or relu if needed
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, activation_fn=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.activation_fn = activation_fn  # optional activation function applied after conv

    def forward(self, x):
        x = super().forward(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


# Depthwise Conv2d wrapper

class DepthwiseConv2d(nn.Conv2d):
    """depthwise conv wrapper for pytorch"""
    # adapted: converted tf.keras DepthwiseConv2D to pytorch depthwise conv
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, in_channels, kernel_size,
                         stride=stride, padding=padding,
                         groups=in_channels, bias=bias)


# Learning Rate Scheduler

def build_learning_rate(initial_lr, global_step, steps_per_epoch=None,
                        lr_decay_type='exponential', decay_factor=0.97,
                        decay_epochs=2.4, total_steps=None, warmup_epochs=5):
    """build pytorch learning rate given global step"""
    # adapted: replaced tf exponential/cosine/poly with pytorch equivalents
    # removed: tf.train ops, tf.cond, logging

    if lr_decay_type == 'exponential':
        assert steps_per_epoch is not None
        decay_steps = steps_per_epoch * decay_epochs
        lr = initial_lr * (decay_factor ** (global_step / decay_steps))
    elif lr_decay_type == 'cosine':
        assert total_steps is not None
        lr = 0.5 * initial_lr * (1 + np.cos(np.pi * global_step / total_steps))
    elif lr_decay_type == 'constant':
        lr = initial_lr
    elif lr_decay_type == 'poly':
        assert steps_per_epoch is not None and total_steps is not None
        warmup_steps = int(steps_per_epoch * warmup_epochs)
        step = max(global_step - warmup_steps, 0)
        lr = initial_lr * (1 - step / (total_steps - warmup_steps + 1)) ** 2
    else:
        raise ValueError(f'unknown lr_decay_type: {lr_decay_type}')

    # warmup schedule: gradually increase lr from 0 to initial_lr over warmup_epochs
    if warmup_epochs:
        warmup_steps = int(steps_per_epoch * warmup_epochs)
        if global_step < warmup_steps:
            lr = initial_lr * (global_step / warmup_steps)
    return lr


# Optimizer Builder

def build_optimizer(model_params, optimizer_name='rmsprop', lr=0.01,
                    momentum=0.9, weight_decay=0.0):
    """build pytorch optimizer"""
    # adapted: tf optimizers replaced by torch.optim
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_params, lr=lr)
    elif optimizer_name == 'momentum':
        optimizer = torch.optim.SGD(model_params, lr=lr, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'unknown optimizer: {optimizer_name}')
    return optimizer


# Load Model Weights

def load_model_weights(model, weight_path, device=None):
    """load pytorch model weights from a .pth file"""
    # added: utility to load pretrained weights into pytorch model
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f'weight file not found: {weight_path}')
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    state_dict = torch.load(weight_path, map_location=device)

    # adapted: remove potential "module." prefix from DataParallel checkpoints
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()  # added: set model to eval mode after loading
    print(f'loaded weights from {weight_path} on {device}')
    return model


# Additional Utilities

def get_activation_fn(name='swish'):
    """return activation function given string name"""
    # added: helper for optional activation_fn in Conv2d
    if name == 'swish':
        return lambda x: x * torch.sigmoid(x)
    elif name == 'relu':
        return F.relu
    elif name is None:
        return None
    else:
        raise ValueError(f'unknown activation function: {name}')


def compute_same_padding(kernel_size, stride, input_size):
    """compute padding needed to preserve spatial dimensions"""
    # added: useful for manual conv padding calculations
    return max((math.ceil(input_size / stride) - 1) * stride + kernel_size - input_size, 0)



# Summary of changes

# adapted:
# - all tensorflow ops replaced with pytorch equivalents (conv2d, batchnorm, depthwise conv)
# - learning rate schedules adapted from tf.train to python/numpy math
# - optimizer builder uses torch.optim instead of tf.train optimizers

# removed:
# - TPU-specific batch norm, cross-replica sums
# - tensorflow placeholders, sessions, and variable scoping
# - tf.gfile, checkpoint archiving, EMA helpers

# added:
# - pytorch-friendly forward methods
# - depthwise conv wrapper for pytorch
# - optional activation functions in conv wrapper
# - simplified drop_connect compatible with pytorch tensors
# - load_model_weights function for .pth files
# - get_activation_fn helper
# - compute_same_padding helper
