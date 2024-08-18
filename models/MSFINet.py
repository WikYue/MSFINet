# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format
from einops import rearrange
from functools import partial
import copy
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
import math
from models.CIT import CIT
from models.CRA import CRA

from mmcv.cnn import constant_init, kaiming_init
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class OverlapPatchEmbed_up(nn.Module):

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] * stride, img_size[1] * stride
        self.num_patches = self.H * self.W

        self.proj_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_chans,embed_dim, kernel_size=1, stride=1),
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj_1(x)
        B, C, H, W = x.shape

        return x, H, W

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class PPM(nn.Module):
    def __init__(self, pooling_sizes=(1, 3, 5)):
        super().__init__()
        self.layer = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(size, size)) for size in pooling_sizes])

    def forward(self, feat):
        b, c, h, w = feat.shape
        output = [layer(feat).view(b, c, -1) for layer in self.layer]
        output = torch.cat(output, dim=-1)
        return output

class ESA_qkv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.ppm = PPM(pooling_sizes=(1, 3, 5))

    def forward(self, x):
        # input x (b, c, h, w)
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)  # q/k/v shape: (b, inner_dim, h, w)
        q = rearrange(q, 'b (head d) h w -> b head (h w) d', head=self.heads)  # q shape: (b, head, n_q, d)

        k, v = self.ppm(k), self.ppm(v)  # k/v shape: (b, inner_dim, n_kv)

        return q,k,v

# Efficient self attention

class ESA_layer(nn.Module):
    def __init__(self, dim,heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q,k,v):

        k = rearrange(k, 'b (head d) n -> b head n d', head=self.heads)  # k shape: (b, head, n_kv, d)
        v = rearrange(v, 'b (head d) n -> b head n d', head=self.heads)  # v shape: (b, head, n_kv, d)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # shape: (b, head, n_q, n_kv)
        attn = self.attend(dots)
        out = torch.matmul(attn, v)  # shape: (b, head, n_q, d)
        out = rearrange(out, 'b head n d -> b n (head d)')
        return self.to_out(out)

class Cat_kv(nn.Module):
    def __init__(self,dim):
        super(Cat_kv, self).__init__()
        self.num = len(dim)
        self.dim = dim
        self.dim_cat = [sum(self.dim[0::]),sum(self.dim[1::]),sum(self.dim[2::]),sum(self.dim[3::])]
        self.dim_change = nn.ModuleList()
        for i_layter in range(self.num):
            layter = nn.Sequential(
                    # nn.Conv2d(self.dim_cat[i_layter],self.dim[i_layter],kernel_size=1,stride=1,padding=0),
                    nn.Conv2d(self.dim[i_layter],self.dim[i_layter],kernel_size=1,stride=1,padding=0),
                    nn.BatchNorm2d(self.dim[i_layter]),
                    nn.ReLU(inplace=True),)
            self.dim_change.append(layter)


    def forward(self,feature):
        list_cat = []
        num = len(feature)
        for i in range(num):
            # feature[i] = torch.cat(feature[i::],dim=1)
            feature[i] = sum(feature[i::])
            feature[i] = self.dim_change[i](feature[i].unsqueeze(-1))
            list_cat.append(feature[i].squeeze(-1))

        return list_cat



class ESA_blcok(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=512, dropout=0.):
        super().__init__()
        self.num = len(dim)
        self.dim = dim
        self.ESAqkv = nn.ModuleList([ESA_qkv(self.dim[i],heads=heads, dim_head=dim_head, dropout=dropout)
                                     for i in range(self.num)])

        self.ESAlayer = nn.ModuleList([ESA_layer(self.dim[i], heads=heads, dim_head=dim_head, dropout=dropout)
                                       for i in range(self.num)])

        self.ff = nn.ModuleList([PreNorm(self.dim[i], FeedForward(self.dim[i], mlp_dim, dropout=dropout))
                                 for i in range(self.num)])

        self.cat_kv =Cat_kv([512,512,512,512])

    def forward(self, f):
        H_list = []
        q_list = []
        k_list = []
        v_list = []
        out_list = []
        out_list_1 = []
        # num = len(f)
        for i in range(self.num):
            out = rearrange(f[i], 'b c h w -> b (h w) c')
            q,k,v = self.ESAqkv[i](f[i])
            b, c, h, w = f[i].size()
            H_list.append(h)
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)
            out_list.append(out)
        k_list = self.cat_kv(k_list)
        v_list = self.cat_kv(v_list)

        for i in range(self.num):

            out = self.ESAlayer[i](q_list[i],k_list[i],v_list[i]) + out_list[i]
            out = self.ff[i](out) + out
            out = rearrange(out, 'b (h w) c -> b c h w', h=H_list[i])
            # out = rearrange(out, 'b (h w) c -> b c h w')
            out_list_1.append(out)
        return out_list_1

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_mul', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out + out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out
class ChannelAttention_(nn.Module):
  def __init__(self, in_channels, reduction_ratio=16):
    super(ChannelAttention_, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(in_channels, in_channels // reduction_ratio),
      nn.ReLU(inplace=True),
      nn.Linear(in_channels // reduction_ratio, in_channels)
    )
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avg_out = self.avg_pool(x)
    max_out = self.max_pool(x)
    avg_out = avg_out.view(avg_out.size(0), -1)
    max_out = max_out.view(max_out.size(0), -1)
    avg_out = self.fc(avg_out)
    max_out = self.fc(max_out)
    out = avg_out + max_out
    out = self.sigmoid(out)
    out = out.view(out.size(0), out.size(1), 1, 1)
    out = out.expand_as(x)
    out = x * out
    return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class ConvBranch(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask
        return res2 + res1


class BDO(nn.Module):

    def __init__(self, input_dim=64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = BasicConv2d(input_dim // 2, input_dim // 2, 1)
        self.d_in2 = BasicConv2d(input_dim // 2, input_dim // 2, 1)

        self.conv = nn.Sequential(BasicConv2d(input_dim, input_dim, 3, 1, 1),
                                  nn.Conv2d(input_dim, 32, kernel_size=1, bias=False))
        self.fc1 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, H_feature, L_feature):
        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)

        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)

        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature,
                                                                                       size=L_feature.size()[2:],
                                                                                       align_corners=False)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature,
                                                                                       size=H_feature.size()[2:],
                                                                                       align_corners=False)

        H_feature = Upsample(H_feature, size=L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class MKDA(nn.Module):

    def __init__(self, input_dim=512, embed_dim=32, k_s=3):
        super().__init__()
        group_size = input_dim // 4
        k_size = 3
        up_channel = 32
        d_list = [1, 2, 5, 7]
        self.ca = ChannelAttention_(in_channels=up_channel*4)
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size, group_size, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=group_size )
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size , data_format='channels_first'),
            nn.Conv2d(group_size , group_size , kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=group_size)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size , data_format='channels_first'),
            nn.Conv2d(group_size , group_size , kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                      dilation=d_list[2], groups=group_size )
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size , data_format='channels_first'),
            nn.Conv2d(group_size , group_size , kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                      dilation=d_list[3], groups=group_size )
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=(group_size ) * 4, data_format='channels_first'),
            nn.Conv2d((group_size ) * 4, up_channel*2, 1)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(up_channel*4, up_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(up_channel))
        self.conv_3_ = nn.Sequential(
            nn.Conv2d(up_channel * 3, up_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(up_channel))
        self.conv1_1 = BasicConv2d(input_dim, up_channel*2, 1)
        self.conv1_1_1 = BasicConv2d(input_dim, embed_dim, 1)
        self.local_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.global_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.GlobelBlock = ContextBlock(inplanes=embed_dim, ratio=2)
        self.local = ConvBranch(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        U = self.conv1_1_1(x)
        b, c, h, w = x.size()
        x_0, x_1, x_2, x_3 = x.chunk(4, dim=1)
        x0 = self.g0(x_0)
        x1 = self.g1(x_1)
        x2 = self.g2(x_2)
        x3 = self.g3(x_3)
        x_ = torch.cat((x0, x1, x2, x3), dim=1)
        SSR = self.tail_conv(x_)
        i1 = self.conv1_1(x)
        xf = torch.cat((SSR, i1), dim=1)
        xf = self.conv_3(self.ca(xf))
        xu = torch.cat((xf, i1), dim=1)
        output = self.conv_3_(xu)
        return output

class ConvBatchNorm1(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super(ConvBatchNorm1, self).__init__()
        #self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=1, padding=1)  # 不改变图片大小
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
def Upsample(x, size, align_corners = False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)
class MSFI(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],qkv_bias=False,
                 qk_scale=None, drop_rate=0.,attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False,aux=True):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.aux =  aux

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)


        for i in range(num_stages-1,0,-1):
            patch_embed_up = OverlapPatchEmbed_up(img_size = img_size // (2 ** (i + 2)),
                                            patch_size = 3,
                                            stride = 2,
                                            in_chans  =  embed_dims[i],
                                            embed_dim = embed_dims[i-1])

            block_up = nn.ModuleList([Block(
                dim=embed_dims[i-1], num_heads=num_heads[i-1], mlp_ratio=mlp_ratios[i-1], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,
                sr_ratio=sr_ratios[i-1], linear=linear)
                for j in range(depths[i-1])])
            norm_up = norm_layer(embed_dims[i-1])
            cur += depths[i-1]

            change_channal = nn.Conv2d(embed_dims[i - 1]*2, embed_dims[i - 1], kernel_size=1, stride=1)

            norm_1 = nn.LayerNorm(embed_dims[i-1])
            setattr(self, f"patch_embed_up{i}", patch_embed_up)
            setattr(self, f"block_up{i}", block_up)
            setattr(self, f"norm_up{i}", norm_up)
            setattr(self, f"norm_1{i}", norm_1)
            setattr(self, f"change_channal{i}", change_channal)


        self.middle = CIT()
        self.middle_CRA = CRA()
        self.MKDA_c4 = MKDA(input_dim=512, embed_dim=32)
        self.MKDA_c3 = MKDA(input_dim=320, embed_dim=32)
        self.MKDA_c2 = MKDA(input_dim=128, embed_dim=32)
        self.MKDA_c1 = MKDA(input_dim=64, embed_dim=32)
        self.L_feature = BasicConv2d(64, 32, 3, 1, 1)
        self.fuse = BasicConv2d(32 * 2, 32, 1)
        self.BDO3 = BDO(input_dim=32)
        self.BDO2 = BDO(input_dim=32)
        self.BDO1 = BDO(input_dim=32)
        self.convtest = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.fuse2 = nn.Sequential(BasicConv2d(32 * 3, 32, 1, 1), nn.Conv2d(32, 1, kernel_size=1, bias=False))
        self.finalconv3 = ConvBatchNorm1(in_channels=32,out_channels=1)
        self.finalconv2 = ConvBatchNorm1(in_channels=32, out_channels=1)
        self.finalconv1 = ConvBatchNorm1(in_channels=32, out_channels=1)
        #self.conv3 = ConvBatchNorm1(in_channels=512,out_channels=32)
        self.Conv_X4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dims[0], embed_dims[0], 3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dims[0], num_classes, 1),

        )


        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head


    def forward_features(self, x):
        B = x.shape[0]
        size = x.size()[2:]

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        encode_feature = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            encode_feature.append(x)

        encode_feature_1 = encode_feature
        encode_feature_1 = self.middle_CRA(encode_feature_1)


        x1 = encode_feature[0]
        x2 = encode_feature[1]
        x3 = encode_feature[2]
        x4 = encode_feature[3]
        x1_1,x2_2,x3_3,x4_4,a = self.middle(x1,x2,x3,x4)
        encode_feature[0] = x1_1
        encode_feature[1] = x2_2
        encode_feature[2] = x3_3
        encode_feature[3] = x4_4
        #
        encode_feature[0] = (x1_1 + encode_feature_1[0])
        encode_feature[1] = (x2_2 + encode_feature_1[1])
        encode_feature[2] = (x3_3 + encode_feature_1[2])
        encode_feature[3] = (x4_4 + encode_feature_1[3])
        x = encode_feature[0]
        _c4 = self.MKDA_c4(encode_feature[3])  # [1, 32, 7, 7]
        _c3 = self.MKDA_c3(encode_feature[2])  # [1, 32, 14, 14]
        _c2 = self.MKDA_c2(encode_feature[1])  # [1, 32, 28, 28]
        _c1 = self.MKDA_c1(encode_feature[0])
        output3 = self.BDO3(_c4, _c3)
        output2 = self.BDO2(output3, _c2)
        output = self.BDO1(output2, _c1)
        output3 = self.finalconv3(output3)
        output2 = self.finalconv2(output2)
        output = self.finalconv1(output)
        output = F.interpolate(output, scale_factor=4, mode='bilinear')
        output2 = F.interpolate(output2, scale_factor=8, mode='bilinear')
        output3 = F.interpolate(output3, scale_factor=16, mode='bilinear')

        # torch.Size([1, 64, 56, 56])
        # torch.Size([1, 128, 28, 28])
        # torch.Size([1, 320, 14, 14])
        # torch.Size([1, 512, 7, 7])

        return output,output2,output3

    def forward(self, x):
        output,output2,output3 = self.forward_features(x)
        return output,output2,output3

def MSFINet(pretrained=False,num_classes=2,**kwargs):
    model = MSFI(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_classes=num_classes,num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 9, 3], sr_ratios=[8, 4, 2, 1],**kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        pretrained_path = "D:/wz/2024-2/pvt_v2_b2.pth"
        # pretrained_path = '/pvt_v2_b2.pth'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "block" in k:
                current_k = "block_up" + k[5:]
                full_dict.update({current_k: v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    del full_dict[k]

        model.load_state_dict(full_dict, strict=False)

    return model



if __name__ == '__main__':

    # input = torch.rand(1, 3, 224, 224)
    input = torch.rand(1, 1, 224, 224)
    net = MSFINet(pretrained=True, num_classes=1)

    output,output2,output3 = net(input)

    Macs, params = profile(net, inputs=(input, ))
    flops, params = clever_format([Macs*2, params], "%.3f")
    print('flops: ', flops)
    print('params: ',params)







