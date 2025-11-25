# efficientnet_model.py - pytorch adaptation with full support for B0-B8 + L2
# includes mbconv, se, drop connect, and dynamic input channels

# imports
import math
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# namedtuples for model configuration
# adapted from tensorflow: globalparams and blockargs
# removed tf defaults like 'relu_fn', 'batch_norm', 'use_se', etc. for simplicity
# these defaults were mostly for tf-specific logging, training flags, or options like 'condconv'
# which are not essential for a pytorch implementation

globalparams = namedtuple('globalparams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'in_channels'  # added in_channels for flexibility
])
globalparams.__new__.__defaults__ = (None,) * len(globalparams._fields)

blockargs = namedtuple('blockargs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type'
])
blockargs.__new__.__defaults__ = (None,) * len(blockargs._fields)

# utility functions

def round_filters(filters, global_params):
    # round number of filters based on width multiplier and depth divisor
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    # round number of block repeats based on depth multiplier
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

# swish activation
class Swish(nn.Module):
    # multiply input by its sigmoid (tf swish equivalent)
    def forward(self, x):
        return x * torch.sigmoid(x)

# drop connect / stochastic depth
def drop_connect(x, survival_prob, training):
    # skip drop connect if not training or survival_prob is 1
    if not training or survival_prob == 1.0:
        return x
    keep_prob = survival_prob
    # create random mask for each sample in the batch, broadcast over channels, height, width
    mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep_prob).float()
    # scale output by keep_prob and apply mask
    return x * mask / keep_prob

# squeeze-and-excitation
class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio):
        super().__init__()
        # compute reduced channel size
        reduced_channels = max(1, int(in_channels * se_ratio))
        # first conv layer (squeeze)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1)
        # second conv layer (excitation)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1)
        # swish activation
        self.activation = Swish()

    def forward(self, x):
        # global average pool
        se = F.adaptive_avg_pool2d(x, 1)
        # squeeze + activation
        se = self.activation(self.fc1(se))
        # excitation + sigmoid
        se = torch.sigmoid(self.fc2(se))
        # scale input by se output
        return x * se

# mbconvblock with drop connect
class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params, block_idx=0, total_blocks=1):
        super().__init__()
        # store block args
        self.block_args = block_args
        # check if se block is used
        self.has_se = block_args.se_ratio is not None and 0 < block_args.se_ratio <= 1
        # whether to use skip connection
        self.id_skip = block_args.id_skip
        # global drop connect probability
        self.survival_prob = global_params.survival_prob

        in_channels = block_args.input_filters
        out_channels = block_args.output_filters
        expand_channels = in_channels * block_args.expand_ratio

        # expansion phase (1x1 conv if expand_ratio != 1)
        if block_args.expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expand_channels, 1, bias=False)
            self.bn0 = nn.BatchNorm2d(expand_channels, momentum=global_params.batch_norm_momentum,
                                      eps=global_params.batch_norm_epsilon)

        # depthwise conv
        self.depthwise_conv = nn.Conv2d(
            expand_channels, expand_channels, block_args.kernel_size, 
            stride=block_args.strides[0], padding=block_args.kernel_size//2,
            groups=expand_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_channels, momentum=global_params.batch_norm_momentum,
                                  eps=global_params.batch_norm_epsilon)

        # squeeze-and-excitation
        if self.has_se:
            self.se_block = SEBlock(expand_channels, block_args.se_ratio)

        # projection phase (1x1 conv to output channels)
        self.project_conv = nn.Conv2d(expand_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=global_params.batch_norm_momentum,
                                  eps=global_params.batch_norm_epsilon)

        # swish activation
        self.activation = Swish()

        # compute adjusted survival prob per block for stochastic depth
        if self.survival_prob is not None:
            self.block_survival_prob = 1.0 - (1.0 - self.survival_prob) * (block_idx / total_blocks)
        else:
            self.block_survival_prob = 1.0  # no drop connect

    def forward(self, x):
        # save input for skip connection
        identity = x

        # expansion conv
        if hasattr(self, 'expand_conv'):
            x = self.activation(self.bn0(self.expand_conv(x)))

        # depthwise conv
        x = self.activation(self.bn1(self.depthwise_conv(x)))

        # se block
        if self.has_se:
            x = self.se_block(x)

        # projection conv
        x = self.bn2(self.project_conv(x))

        # skip connection with drop connect
        if self.id_skip and x.shape == identity.shape:
            x = drop_connect(x, self.block_survival_prob, self.training)
            x = x + identity

        return x

# efficientnet model
class EfficientNet(nn.Module):
    def __init__(self, blocks_args, global_params):
        super().__init__()
        # store global params
        self.global_params = global_params
        # compute stem channels after scaling
        in_channels = round_filters(32, global_params)
        # dynamic input channels, fallback to 3
        in_ch = global_params.in_channels or 3

        # stem conv
        self.stem_conv = nn.Conv2d(in_ch, in_channels, 3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(in_channels, momentum=global_params.batch_norm_momentum,
                                      eps=global_params.batch_norm_epsilon)
        # swish activation
        self.activation = Swish()

        # build mbconv blocks
        self.blocks = nn.ModuleList([])
        # total number of blocks for stochastic depth
        total_blocks = sum(round_repeats(b.num_repeat, global_params) for b in blocks_args)
        block_idx = 0
        for block_args in blocks_args:
            # scale filters and repeats
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, global_params),
                output_filters=round_filters(block_args.output_filters, global_params),
                num_repeat=round_repeats(block_args.num_repeat, global_params)
            )

            # first block
            self.blocks.append(MBConvBlock(block_args, global_params, block_idx, total_blocks))
            block_idx += 1
            # remaining repeats
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1])
                for _ in range(block_args.num_repeat - 1):
                    self.blocks.append(MBConvBlock(block_args, global_params, block_idx, total_blocks))
                    block_idx += 1

        # head conv
        head_in = block_args.output_filters
        head_out = round_filters(1280, global_params)  # scale for larger models
        self.head_conv = nn.Conv2d(head_in, head_out, 1, bias=False)
        self.head_bn = nn.BatchNorm2d(head_out, momentum=global_params.batch_norm_momentum,
                                      eps=global_params.batch_norm_epsilon)
        # global avg pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # dropout before fc
        self.dropout = nn.Dropout(global_params.dropout_rate) if global_params.dropout_rate else nn.Identity()
        # final fully connected layer
        self.fc = nn.Linear(head_out, global_params.num_classes)

    def forward(self, x):
        # stem conv
        x = self.activation(self.stem_bn(self.stem_conv(x)))
        # pass through mbconv blocks
        for block in self.blocks:
            x = block(x)
        # head conv
        x = self.activation(self.head_bn(self.head_conv(x)))
        # global avg pool
        x = self.avg_pool(x)
        # flatten for fc
        x = torch.flatten(x, 1)
        # dropout
        x = self.dropout(x)
        # final classifier
        x = self.fc(x)
        return x
