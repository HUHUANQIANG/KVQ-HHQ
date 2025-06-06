import torch
import torch.nn as nn
import time
from torch.nn.functional import adaptive_avg_pool3d
from functools import partial, reduce, lru_cache
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
import math
from operator import mul
from einops import rearrange
from copy import deepcopy

class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """
    def __init__(self, qk_dim, num_heads, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.num_heads = num_heads
        self.scale = (qk_dim//num_heads) ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)
    
    def forward(self, query, key):
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key) # per-window pooling -> (n, p^2, c) 
        B, nW, N, C = query.shape

        attn_logit = (query_hat.view(B, -1, C)*self.scale) @ key_hat.view(B, -1, C).transpose(-2, -1) # (n, p^2, p^2)
        attn_map = F.softmax(attn_logit,-1).sum(1).view(B, nW, N).detach()

        query_hat = query_hat.view(B, nW, N, self.num_heads, C // self.num_heads).permute(0,3,1,2,4).mean(3)
        key_hat = key_hat.view(B, nW, N, self.num_heads, C // self.num_heads).permute(0,3,1,2,4).mean(3)
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1) # (n, p^2, p^2)

        topk_attn_logit, topk_index = torch.topk(attn_logit, k=min(self.topk, attn_logit.shape[-1]), dim=-1) # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit) # (n, p^2, k)
        
        return r_weight, topk_index, attn_map

class VQAHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
        self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Conv3d(self.in_channels, self.hidden_channels, (1, 1, 1))
        self.fc_last = nn.Conv3d(self.hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, rois=None):
        x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
        return qlt_score

class VARHead(nn.Module):
    """MLP Regression Head for Video Action Recognition.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
        self, in_channels=768, out_channels=400, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc = nn.Conv3d(self.in_channels, self.out_channels, (1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, rois=None):
        x = self.dropout(x)
        x = self.avg_pool(x)
        out = self.fc(x)
        return out

class SalientHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
        self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Conv3d(self.in_channels, self.hidden_channels, (1, 1, 1))
        self.fc_last = nn.Conv3d(self.hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))


    def forward(self, x, rois=None):
        x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
        return qlt_score

def get_adaptive_window_size(
    base_window_size,
    input_x_size,
    base_x_size,
):
    tw, hw, ww = base_window_size
    tx_, hx_, wx_ = input_x_size
    tx, hx, wx = base_x_size
    print((tw * tx_) // tx, (hw * hx_) // hx, (ww * wx_) // wx)
    return (tw * tx_) // tx, (hw * hx_) // hx, (ww * wx_) // wx

class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(B, -1, reduce(mul, window_size), C)
    )
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def window_process(x, pad_shape, window_size, shift_size, shift, resized_window_size=None):
    B, D, H, W, C = x.shape
    window_size, shift_size = get_window_size(
        (D, H, W), 
        window_size if resized_window_size is None else resized_window_size, 
        shift_size
    )
    
    # shape --> (B, H+pad_b, W+pad_r, C)
    query = F.pad(x, pad_shape)
    _, Dp, Hp, Wp, _ = x.shape
    if shift:
        shifted_query = torch.roll(
            query,
            shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
            dims=(1, 2, 3))
    else:
        shifted_query = query
    
    return shifted_query

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class CrossWindowAttention3D(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        shift_size,
        num_heads,
        topk=4,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        cross_flag = False,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.shift_size = shift_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.cross_flag = cross_flag
        self.topk=topk
        self.router = TopkRouting(qk_dim=self.dim,
                num_heads=self.num_heads,
                qk_scale=self.scale,
                topk=self.topk,
                diff_routing=False,
                param_routing=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.pos_embed_flag = False
        if self.pos_embed_flag:
            self.pos_embed = nn.Conv3d(dim, dim,  kernel_size=3, padding=1, groups=dim)

    def forward(self, x, x_origin_shape, x_pad_shape, pad_shape, window_size, shift_size, resized_window_size=None):
        B, nW, N, C = x.shape
        D, H, W = x_origin_shape
        D_pad, H_pad, W_pad = x_pad_shape

        if any(i > 0 for i in shift_size):
            attn_mask = compute_mask(D_pad, H_pad, W_pad, window_size, shift_size, x.device)
        else:
            attn_mask = None

        qkv = (
            self.qkv(x)
            .reshape(B, nW, N, 3, -1)
            .permute(3, 0, 1, 2, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        r_weight, r_idx, attn_map = self.router(q, k)
        topk = min(self.topk, r_idx.shape[2])

        q = q.view(B, nW, N, self.num_heads, C // self.num_heads).permute(0,3,1,2,4)
        k = k.view(B, nW, N, self.num_heads, C // self.num_heads).permute(0,3,1,2,4)
        v = v.view(B, nW, N, self.num_heads, C // self.num_heads).permute(0,3,1,2,4)

        k = torch.gather(k.view(B, self.num_heads, 1, nW, N, C // self.num_heads).expand(-1, -1, nW, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                dim=3,
                                index=r_idx.view(B, self.num_heads, nW, topk, 1, 1).expand(-1, -1, -1, -1, N, C // self.num_heads) # (n, p^2, k, w^2, c_kv)
                            )
        v = torch.gather(v.view(B, self.num_heads, 1, nW, N, C // self.num_heads).expand(-1, -1, nW, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                dim=3,
                                index=r_idx.view(B, self.num_heads, nW, topk, 1, 1).expand(-1, -1, -1, -1, N, C // self.num_heads) # (n, p^2, k, w^2, c_kv)
                            )

        k = r_weight.view(B, self.num_heads, nW, topk, 1, 1) * k
        v = r_weight.view(B, self.num_heads, nW, topk, 1, 1) * v

        k = k.permute(0, 2, 1, 3, 4, 5).contiguous().view(B*nW, self.num_heads, topk, N, C // self.num_heads).view(B*nW, self.num_heads, N*topk, C // self.num_heads)
        v = v.permute(0, 2, 1, 3, 4, 5).contiguous().view(B*nW, self.num_heads, topk, N, C // self.num_heads).view(B*nW, self.num_heads, N*topk, C // self.num_heads)
        q = q.permute(0, 2, 1, 3, 4).view(B*nW, self.num_heads, N, C // self.num_heads)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).view(B, nW, N, self.num_heads, C // self.num_heads).contiguous().view(B, nW, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)


        return x, attn_map


class WindowAttention3D(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        shift_size,
        num_heads,
        topk=4,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        cross_flag = False,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.shift_size = shift_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.cross_flag = cross_flag

        if self.cross_flag:
            self.cross_windowattn = CrossWindowAttention3D(dim,
                window_size=self.window_size,
                shift_size=self.shift_size,
                num_heads=num_heads,
                topk=topk,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                cross_flag = cross_flag,
            )        

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1),
                num_heads,
            )
        )  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w)
        )  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
            2 * self.window_size[2] - 1
        )
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)


    def forward_iwa(self, x, x_origin_shape, x_pad_shape, window_size, shift_size, resized_window_size=None):
        B, nW, N, C = x.shape
        D, H, W = x_origin_shape
        D_pad, H_pad, W_pad = x_pad_shape

        if any(i > 0 for i in shift_size):
            attn_mask = compute_mask(D_pad, H_pad, W_pad, window_size, shift_size, x.device)
        else:
            attn_mask = None


        qkv = (
            self.qkv(x)
            .reshape(B*nW, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        if resized_window_size is None:
            rpi = self.relative_position_index[:N, :N]
        else:
            relative_position_index = self.relative_position_index.reshape(*self.window_size, *self.window_size)
            d, h, w = resized_window_size
            
            rpi = relative_position_index[:d,:h,:w,:d,:h,:w]
        relative_position_bias = self.relative_position_bias_table[
            rpi.reshape(-1), 0:self.num_heads
        ].reshape(
            N, N, -1
        )  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if attn_mask is not None:
            ori_nW = attn_mask.shape[0]
            attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1, -1)
            attn = attn.view(B, nW, self.num_heads, N, N) + attn_mask.unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, nW, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def forward(self, x, x_origin_shape, x_pad_shape, pad_shape, window_size, shift_size, docwa = True,  resized_window_size=None):
        if self.cross_flag and docwa:
            x_windows = self.forward_iwa(x, x_origin_shape, x_pad_shape, window_size, shift_size, resized_window_size=None)
            x_cross, attn_map = self.cross_windowattn(x, x_origin_shape, x_pad_shape, pad_shape, window_size, shift_size, resized_window_size=None)
            x_windows = x_windows + x_cross
        else:
            x_windows = self.forward_iwa(x, x_origin_shape, x_pad_shape, window_size, shift_size, resized_window_size=None)
            attn_map = None
        return x_windows, attn_map

class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        topk=4,
        window_size=(4, 4, 4),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        cross_flag=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert (
            0 <= self.shift_size[0] < self.window_size[0]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[1] < self.window_size[1]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[2] < self.window_size[2]
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=self.window_size,
            shift_size = self.shift_size,
            num_heads=num_heads,
            topk=topk,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cross_flag=cross_flag,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.pos_embed_flag = False
        if self.pos_embed_flag:
            self.pos_embed = nn.Conv3d(dim, dim,  kernel_size=3, padding=1, groups=dim)


    def forward_part(self, x_windows, shift, x_origin_shape, x_pad_shape, pad_shape, docwa, resized_window_size=None):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        D, H, W = x_pad_shape
        D_origin, H_origin, W_origin = x_origin_shape
        _, _, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1 = pad_shape
        window_size, shift_size = get_window_size(
            (D_origin, H_origin, W_origin), 
            self.window_size, 
            self.shift_size
        )
        if self.pos_embed_flag:
            x_windows = x_windows + self.pos_embed(x_windows.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        shifted_query = window_process(x_windows, pad_shape, self.window_size, self.shift_size, shift, resized_window_size)
        x_windows = window_partition(shifted_query, window_size)
        B, nW, N, C = x_windows.shape

        res_identity = x_windows
        x_windows = self.norm1(x_windows)
        attn_windows, attn_map = self.attn(x_windows, x_origin_shape, x_pad_shape, pad_shape, window_size, shift_size, docwa, resized_window_size=window_size if resized_window_size is not None else None,)
        x_windows = self.drop_path(attn_windows) + res_identity


        x_windows = x_windows.view(B, -1, C)
        identity = x_windows
        x_windows = x_windows + self.drop_path(self.mlp(self.norm2(x_windows)))

        x_windows = x_windows.view(B, nW, N, C)

        # merge windows
        x_windows = x_windows.view(-1, window_size[0], window_size[1], window_size[2], C)
        shifted_x = window_reverse(x_windows, window_size, B, D, H, W)  # B H' W' C
        if attn_map is not None:
            attn_map = attn_map.view(-1, window_size[0], window_size[1], window_size[2], 1)
            attn_map = window_reverse(attn_map, window_size, B, D, H, W)

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3),
            )
            if attn_map is not None:
                attn_map = torch.roll(
                    attn_map,
                    shifts=(shift_size[0], shift_size[1], shift_size[2]),
                    dims=(1, 2, 3),
                )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D_origin, :H_origin, :W_origin, :].contiguous()

        return x, attn_map

    def forward(self, x_windows, shift, x_origin_shape, x_pad_shape, pad_shape, docwa =True, resized_window_size=None):
        if self.use_checkpoint:
            x, attn_map = checkpoint.checkpoint(self.forward_part, x_windows, shift, x_origin_shape, x_pad_shape, pad_shape, docwa, resized_window_size)
        else:
            x, attn_map = self.forward_part(x_windows, shift, x_origin_shape, x_pad_shape, pad_shape, docwa, resized_window_size)

        return x, attn_map

class PatchMerging(nn.Module):
    """Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

# cache each stage results
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.view(-1, mask_windows.shape[2], mask_windows.shape[3])
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=(1, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        cross_flag = 0.,
        topk = 0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    topk = topk,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    cross_flag = cross_flag,
                )
                for i in range(depth)
            ]
        )
        if not isinstance(cross_flag, list):
            self.cross_flag = [cross_flag for i in range(depth)]
        else:
            self.cross_flag = cross_flag

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x, cross_flag, epoch, resized_window_size=None):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        
        window_size, shift_size = get_window_size(
            (D, H, W), 
            self.window_size if resized_window_size is None else resized_window_size, 
            self.shift_size,
        )
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        pad_shape = [0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1]

        attn_map_list = None
        x = rearrange(x, "b c d h w -> b d h w c")
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        x_pad_shape = [Dp, Hp, Wp]
        x_origin_shape = [D, H, W]

        for i, blk in enumerate(self.blocks):
            shift=False if i % 2 == 0 else True            
            x, attn_map = blk(x, shift, x_origin_shape, x_pad_shape, pad_shape, cross_flag, resized_window_size)
            if  attn_map is not None and attn_map_list is None:
                attn_map_list = attn_map
            elif attn_map is not None:
                attn_map_list = torch.cat([attn_map_list, attn_map], dim=-1)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")

        return x, attn_map_list

class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x

class SwinTransformer3D(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(
     
        self,
        pretrained=None,
        pretrained2d=False,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        topk = [1, 1, 8, 4],
        window_size=(8, 7, 7),
        cross_layer=[0, 0, 1, 1],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        frozen_stages=-1,
        use_checkpoint=True,
        base_x_size=(32, 224, 224),
    ):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.base_x_size = base_x_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer]
                if isinstance(window_size, list)
                else window_size,
                topk = topk[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                cross_flag = cross_layer[i_layer],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self._freeze_stages()
        
        self.init_weights()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location="cpu")
        state_dict = checkpoint["model"]
        

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in state_dict.keys() if "relative_position_index" in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]
            

        state_dict["patch_embed.proj.weight"] = (
            state_dict["patch_embed.proj.weight"]
            .unsqueeze(2)
            .repeat(1, 1, self.patch_size[0], 1, 1)
            / self.patch_size[0]
        )

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in state_dict.keys() if "relative_position_bias_table" in k
        ]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = (
                        torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1, 0).view(
                                1, nH1, S1, S1
                            ),
                            size=(
                                2 * self.window_size[1] - 1,
                                2 * self.window_size[2] - 1,
                            ),
                            mode="bicubic",
                        )
                    )
                    relative_position_bias_table_pretrained = (
                        relative_position_bias_table_pretrained_resized.view(
                            nH2, L2
                        ).permute(1, 0)
                    )
            state_dict[k] = relative_position_bias_table_pretrained.repeat(
                2 * wd - 1, 1
            )
        
        msg = self.load_state_dict(state_dict, strict=False)
        print(msg)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def load_swin(self, load_path, strict=False):
        print("loading swin lah")
        from collections import OrderedDict

        model_state_dict = self.state_dict()
        state_dict = torch.load(load_path)["state_dict"]

        clean_dict = OrderedDict()
        for key, value in state_dict.items():
            if "backbone" in key:
                clean_key = key[9:]
                clean_dict[clean_key] = value
                if "relative_position_bias_table" in clean_key:
                    forked_key = clean_key.replace(
                        "relative_position_bias_table", "fragment_position_bias_table"
                    )
                    if forked_key in clean_dict:
                        print("load_swin_error?")
                    else:
                        clean_dict[forked_key] = value

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in clean_dict.keys() if "relative_position_bias_table" in k
        ]
        for k in relative_position_bias_table_keys:
            print(k)
            relative_position_bias_table_pretrained = clean_dict[k]
            relative_position_bias_table_current = model_state_dict[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if isinstance(self.window_size, list):
                i_layer = int(k.split(".")[1])
                L2 = (2 * self.window_size[i_layer][1] - 1) * (
                    2 * self.window_size[i_layer][2] - 1
                )
                wd = self.window_size[i_layer][0]
            else:
                L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
                wd = self.window_size[0]
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int((L1 / 15) ** 0.5)
                    print(
                        relative_position_bias_table_pretrained.shape, 15, nH1, S1, S1
                    )
                    relative_position_bias_table_pretrained_resized = (
                        torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1, 0)
                            .view(nH1, 15, S1, S1)
                            .transpose(0, 1),
                            size=(
                                2 * self.window_size[i_layer][1] - 1,
                                2 * self.window_size[i_layer][2] - 1,
                            ),
                            mode="bicubic",
                        )
                    )
                    relative_position_bias_table_pretrained = (
                        relative_position_bias_table_pretrained_resized.transpose(
                            0, 1
                        ).view(nH2, 15, L2)
                    )
            clean_dict[k] = relative_position_bias_table_pretrained  # .repeat(2*wd-1,1)

        ## Clean Mismatched Keys
        for key, value in model_state_dict.items():
            if key in clean_dict:
                if value.shape != clean_dict[key].shape:
                    print(key)
                    clean_dict.pop(key)

        self.load_state_dict(clean_dict, strict=strict)

    def init_weights(self, pretrained=None):
        print(self.pretrained, self.pretrained2d)
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            #logger = get_root_logger()
            #logger.info(f"load model from: {self.pretrained}")

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights()
            else:
                # Directly load 3D model.
                self.load_swin(self.pretrained, strict=False)  # , logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x, cross_flag, epoch, adaptive_window_size=False):
        
        """Forward function."""
        if adaptive_window_size:
            resized_window_size = get_adaptive_window_size(self.window_size, x.shape[2:], self.base_x_size)
        else:
            resized_window_size = None
        
        x = self.patch_embed(x)

        x = self.pos_drop(x)
        feats = [x]
        attn_map = []

        for l, mlayer in enumerate(self.layers):
            x, attn_map_list = mlayer(x.contiguous(), cross_flag, epoch, resized_window_size)
            if attn_map_list is not None:
                attn_map.append(attn_map_list)
            feats += [x]
            

        x = rearrange(x, "n c d h w -> n d h w c")
        x = self.norm(x)

        x = rearrange(x, "n d h w c -> n c d h w")

        return x, attn_map

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D, self).train(mode)
        self._freeze_stages()

class KVQ(nn.Module):
    def __init__(
        self,
        backbone = None,
        vqa_head=dict(in_channels=768),
    ):
        super().__init__()
        
        self.act_mag_flag = backbone['fragments']['act_mag_flag']
        if backbone!=None:
            self.fragments_backbone = SwinTransformer3D(cross_layer = backbone['fragments']['cross_layer'], )
        else:
            self.fragments_backbone = SwinTransformer3D()
        self.vqa_head = VQAHead(**vqa_head)
        
        self.weight_head = SalientHead()
        self.weight_act = nn.Softmax(-1)
        if self.act_mag_flag:
            self.weight_conv = nn.Conv3d(8,1,(1,1,1))
            nn.init.normal_(self.weight_conv.weight, std=8 ** -0.5)
            nn.init.zeros_(self.weight_conv.weight)
            nn.init.zeros_(self.weight_conv.bias)
        self.temperature = 1.0


    def forward(self, vclips, epoch = 10, inference=True, return_pooled_feats=False, reduce_scores=True, pooled=False, cross_flag=True, patch_flag=False, patch_video=None, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                scores_patch = []
                scores = []
                feats = {}
                for key in vclips:
                    feat, act_mag = self.fragments_backbone(vclips[key], cross_flag, epoch, **kwargs)
                    scores += [self.vqa_head(feat)]
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x,y:x+y, scores)
                    else:
                        scores = scores[0]

            if self.act_mag_flag ==True and len(act_mag)>0: 
                weight_all = None
                scores_local = scores
                weight = self.weight_head(feat)      
                for i in range(len(act_mag)):
                    act_mag_stage = act_mag[i].permute(0,4,1,2,3)
                    kernel = act_mag_stage.shape[-1]//scores.shape[-1]
                    act_mag_stage =  F.avg_pool3d(act_mag_stage, kernel_size = (1,kernel, kernel), stride = (1,kernel, kernel))
                    act_mag[i] = act_mag_stage
                    if weight_all == None:
                        weight_all = act_mag_stage
                    else:
                        weight_all = torch.cat([weight_all, act_mag_stage], 1)
                weight_all = self.weight_conv(weight_all) + weight
                B, C, T, H, W = weight_all.shape
                temperature = self.temperature
                weight_all = self.weight_act(weight_all.view(B,C,T,H*W)/temperature).view(B, C, T, H, W)*H*W
                scores = scores * weight_all
            else:
                weight_all = None
                scores_local = scores
                weight_all = self.weight_head(feat)
                B, C, T, H, W = weight_all.shape
                weight_all = self.weight_act(weight_all.view(B,C,T,H*W)/self.temperature).view(B, C, T, H, W)*H*W
                scores = scores * weight_all

            self.train()
            return scores, feat, act_mag, scores_local, weight_all
        else:
            self.train()
            scores = []
            scores_patch = []
            feats = {}
            for key in vclips:
                feat, act_mag = self.fragments_backbone(vclips[key], cross_flag, epoch, **kwargs)
                scores += [getattr(self, "vqa_head")(feat)]
                if patch_flag:
                    feat_patch, _ = self.fragments_backbone(patch_video, cross_flag, epoch, **kwargs)
                    scores_patch += [getattr(self, "vqa_head")(feat_patch)]
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x,y:x+y, scores)
                    scores_patch = reduce(lambda x,y:x+y, scores_patch)
                else:
                    scores = scores[0]
                    if len(scores_patch)>0:
                        scores_patch = scores_patch[0]

            if self.act_mag_flag ==True and len(act_mag)>0: 
                weight_all = None
                scores_local = scores
                weight = self.weight_head(feat)      
                for i in range(len(act_mag)):
                    act_mag_stage = act_mag[i].permute(0,4,1,2,3)
                    kernel = act_mag_stage.shape[-1]//scores.shape[-1]
                    act_mag_stage =  F.avg_pool3d(act_mag_stage, kernel_size = (1,kernel, kernel), stride = (1,kernel, kernel))
                    act_mag[i] = act_mag_stage
                    if weight_all == None:
                        weight_all = act_mag_stage
                    else:
                        weight_all = torch.cat([weight_all, act_mag_stage], 1)
                weight_all = self.weight_conv(weight_all) + weight
                B, C, T, H, W = weight_all.shape
                temperature = self.temperature
                weight_all = self.weight_act(weight_all.view(B,C,T,H*W)/temperature).view(B, C, T, H, W)*H*W
                scores = scores * weight_all
            else:
                weight_all = None
                scores_local = scores
                weight_all = self.weight_head(feat)
                B, C, T, H, W = weight_all.shape
                weight_all = self.weight_act(weight_all.view(B,C,T,H*W)/self.temperature).view(B, C, T, H, W)*H*W
                scores = scores * weight_all

            
            return scores, feat, act_mag, scores_local, weight_all, scores_patch



if __name__ == "__main__":
    model = KVQ()
    batchsize = 3
    data_input={}
    data_input['fragments'] = torch.autograd.Variable(torch.randn([batchsize, 3, 32, 224, 224]))
    model(data_input)
