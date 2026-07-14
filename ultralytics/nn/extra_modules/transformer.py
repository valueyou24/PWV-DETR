import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

from ..modules.conv import Conv, autopad
from ..modules.transformer import TransformerEncoderLayer
from .attention import DAttention, HiLo, EfficientAdditiveAttnetion, AttentionTSSA
from .prepbn import RepBN, LinearNorm
from .ast import AdaptiveSparseSA
from .filc import FMFFN
from .semnet import SEFN
from .mona import Mona
from .transMamba import SpectralEnhancedFFN
from .EVSSM import EDFFN
from .srconvnet import MixFFN

ln = nn.LayerNorm
linearnorm = partial(LinearNorm, norm1=ln, norm2=RepBN, step=60000)

__all__ = [ 'TransformerEncoderLayer_ASSA', 'TransformerEncoderLayer_ASSA_SEFN','TransformerEncoderLayer_VEFN']

######################################## TransformerEncoderLayer_LocalWindowAttention start ########################################

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def switch_to_deploy(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class TransformerEncoderLayer_ASSA(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.assa = AdaptiveSparseSA(c1, num_heads=num_heads, sparseAtt=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)

        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        BS, C, H, W = src.size()
        src2 = self.assa(src).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


#hang-1
class TransformerEncoderLayer_VEFN(nn.Module):
    def __init__(self, c1=256, cm=512, num_heads=8, dropout=0.1, act=nn.GELU()):
        """
        垂直增强特征融合模块 (VEFN)
        适用于 RT-DETR 中输入通道为 256 的情况。
        Args:
            c1: 输入通道数 (默认 256)
            cm: 前馈网络隐藏维度 (默认 512)
            num_heads: 注意力头数
            dropout: Dropout 概率
        """
        super().__init__()
        # 全局自注意力模块 (自适应稀疏注意力)
        self.assa = AdaptiveSparseSA(c1, num_heads=num_heads, sparseAtt=True)

        # 垂直卷积特征提取（增强垂直方向信息）
        self.vconv = nn.Conv2d(c1, c1, kernel_size=(5, 1), padding=(2, 0), groups=c1)
        #self.vconv = nn.Conv2d(c1, c1, kernel_size=(7, 1), padding=(7//2, 0), groups=c1)  #测试7x1的垂直卷积
        #self.vconv = nn.Conv2d(c1, c1, kernel_size=(9, 1), padding=(9 // 2, 0), groups=c1)  # 测试9x1的垂直卷积
        #self.vconv = nn.Conv2d(c1, c1, kernel_size=(11, 1), padding=(11 // 2, 0), groups=c1)  # 测试9x1的垂直卷积

        # 通道注意力（轻量级重标定）
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   #原本为avgPool
            #nn.AdaptiveMaxPool2d(1),    #测试maxPool
            nn.Conv2d(c1, c1 // 8, 1),
            nn.GELU(),
            nn.Conv2d(c1 // 8, c1, 1),
            nn.Sigmoid()
        )

        #
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)

        # 归一化与 Dropout
        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, src):
        B, C, H, W = src.shape  # e.g., [1, 256, 20, 20]

        # 自注意力
        src2 = self.assa(src)  # [B, C, H*W]
        src2 = src2.view(B, C, H, W)

        # 垂直特征增强
        v_weight = self.vconv(src).sigmoid()
        src2 = src2 * v_weight

        # 通道重校准
        src2 = src2 * self.ca(src2)

        # 残差 + 归一化
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络 + 残差
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)


