import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.modules.utils import _pair
from torch import Tensor
from torch.jit import Final
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union, List
from einops import rearrange, reduce
from collections import OrderedDict

from ..backbone.UniRepLKNet import get_bn, get_conv2d, NCHWtoNHWC, GRNwithNHWC, SEBlock, NHWCtoNCHW, fuse_bn, merge_dilated_into_large_kernel
from ..backbone.rmt import RetBlock, RelPos2d
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
from .attention import *
from .ops_dcnv3.modules import DCNv3
from .transformer import LocalWindowAttention
from .dynamic_snake_conv import DySnakeConv
from .RFAConv import RFAConv, RFCAConv, RFCBAMConv
from .rep_block import *
from .shiftwise_conv import ReparamLargeKernelConv
from .mamba_vss import VSSBlock
from .orepa import OREPA
from .fadc import AdaptiveDilatedConv
from .hcfnet import PPA, LocalGlobalAttention
from .deconv import DEConv
from .SMPConv import SMPConv
from .kan_convs import FastKANConv2DLayer, KANConv2DLayer, KALNConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer
from .wtconv2d import WTConv2d
from .camixer import CAMixer
from .tsdn import DTAB, LayerNorm
from .metaformer import MetaFormerBlock, MetaFormerCGLUBlock, SepConv
from .savss import *
from ..backbone.MambaOut import GatedCNNBlock_BCHW, LayerNormGeneral
from .efficientvim import EfficientViMBlock, EfficientViMBlock_CGLU
from ..backbone.overlock import RepConvBlock
from .filc import *
from .DCMPNet import LEGM
from .mobileMamba.mobilemamba import MobileMambaBlock
from .semnet import SBSM
from ..backbone.lsnet import LSConv, Block as LSBlock

from .dsan import *
from .MaIR import *


from ultralytics.utils.torch_utils import fuse_conv_and_bn, make_divisible
from timm.layers import CondConv2d, DropPath, trunc_normal_, use_fused_attn, to_2tuple

__all__ = [  'BasicBlock_PConv','BasicBlock_PConv_Rep', ]



class Partial_conv3_Rep(Partial_conv3):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__(dim, n_div, forward)
        
        self.partial_conv3 = RepConv(self.dim_conv3, self.dim_conv3, k=3, act=False, bn=False)


class BasicBlock_PConv_Rep(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)

        self.branch2b = nn.Sequential(
            Partial_conv3_Rep(dim=ch_out),
            nn.BatchNorm2d(num_features=ch_out),
            nn.ReLU()
        )

class BasicBlock_PConv(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = nn.Sequential(
            Partial_conv3(dim=ch_out),
            nn.BatchNorm2d(num_features=ch_out),
            nn.ReLU()
        )
