# efficientnet_builder.py - pytorch version
# faithful adaptation of tensorflow efficientnet_builder.py
# copyright 2019 the tensorflow authors, adapted for pytorch

import math
import re
from collections import namedtuple
from EfficientnetModel import EfficientNet, globalparams, blockargs, Swish, drop_connect, round_filters, round_repeats
import torch
import torch.nn as nn

# normalization constants (note that this is the same as ImageNet)

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

# default block strings from tf
# same as tf version; describes the sequence of mbconv blocks
_DEFAULT_BLOCKS_ARGS = [
    'r1_k3_s11_e1_i32_o16_se0.25',
    'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25',
    'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25',
    'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25'
]

# model scaling parameters (width, depth, resolution, dropout)
# defines width/depth/resolution/dropout for each efficientnet variant
_MODEL_PARAMS = {
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
}

# block decoder => converts strings to mbconv block configs
# also provides encoder to reverse back into strings
class BlockDecoder:
    """decode and encode block strings for readability and pytorch usage"""

    @staticmethod
    def _decode_block_string(block_string):
        """decode a single block string (e.g., r1_k3_s11_e1_i32_o16_se0.25) to BlockArgs"""
        assert isinstance(block_string, str), 'block string must be a str'

        # split by underscores
        ops = block_string.split('_')
        options = {}
        for op in ops:
            # split into key/value (e.g., 'k3' -> 'k', '3')
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # ensure strides is a pair of integers
        if 's' not in options or len(options['s']) != 2:
            raise ValueError('strides should be a pair of integers')

        # convert to BlockArgs, keep optional placeholders for unused tf params
        return blockargs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            strides=[int(options['s'][0]), int(options['s'][1])],
            se_ratio=float(options['se']) if 'se' in options else None,
            conv_type=int(options['c']) if 'c' in options else 0  # placeholder for conv_type
        )

    @staticmethod
    def _encode_block_string(block):
        """encode a BlockArgs back to a string"""
        args = [
            f'r{block.num_repeat}',
            f'k{block.kernel_size}',
            f's{block.strides[0]}{block.strides[1]}',
            f'e{block.expand_ratio}',
            f'i{block.input_filters}',
            f'o{block.output_filters}',
            f'c{block.conv_type}'  # placeholder for conv_type
        ]
        if block.se_ratio and 0 < block.se_ratio <= 1:
            args.append(f'se{block.se_ratio}')
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(list_strings):
        """decode list of block strings to list of BlockArgs
           - converts complete efficientnet architecture from strings to structured data
           - simplifies network initialization and ensures consistency
        """
        assert isinstance(list_strings, list)
        return [BlockDecoder._decode_block_string(s) for s in list_strings]

    @staticmethod
    def encode(blocks_args):
        """encode a list of BlockArgs into block strings
           - useful for saving, reproducing, and debugging model architectures
        """
        return [BlockDecoder._encode_block_string(b) for b in blocks_args]


# get model params for a given efficientnet variant
# returns global params and decoded block args
def get_model_params(model_name, override_params=None):
    """get global params and block args for a given model name"""
    
    # check if model name exists in predefined dict
    if model_name not in _MODEL_PARAMS:
        raise NotImplementedError(f'model {model_name} not pre-defined')

    # unpack width, depth, resolution (ignored), and dropout rate
    width_coef, depth_coef, _, dropout_rate = _MODEL_PARAMS[model_name]

    # adapted: use pytorch GlobalParams
    global_params = globalparams(
        batch_norm_momentum=0.99,        # momentum for batchnorm layers
        batch_norm_epsilon=1e-3,         # epsilon for numerical stability
        dropout_rate=dropout_rate,       # dropout for final classifier
        survival_prob=0.8,               # drop connect probability
        num_classes=1000,                # default imagenet classes
        width_coefficient=width_coef,    # network width scaling
        depth_coefficient=depth_coef,    # network depth scaling
        depth_divisor=8,                 # make filters divisible by this
        min_depth=None,                  # optional min filter depth
        in_channels=3                     # default input channels
    )

    # apply any overrides provided by user
    if override_params:
        global_params = global_params._replace(**override_params)

    # decode default block strings into BlockArgs
    blocks_args = BlockDecoder.decode(_DEFAULT_BLOCKS_ARGS)
    
    return blocks_args, global_params


# build efficientnet
# creates an EfficientNet pytorch model instance using decoded block args and global params
def efficientnet(model_name='efficientnet-b0', override_params=None):
    """create an EfficientNet model"""
    # get default blocks and global params for the given model name
    blocks_args, global_params = get_model_params(model_name, override_params)
    
    # create the EfficientNet model with pytorch using decoded blocks and params
    model = EfficientNet(blocks_args, global_params)
    
    # return the pytorch model instance
    return model
