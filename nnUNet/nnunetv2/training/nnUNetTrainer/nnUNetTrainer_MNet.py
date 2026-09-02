

from typing import Tuple

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
import torch._dynamo
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from einops import rearrange, einsum
from functools import partial
from torch import Tensor
import torch.nn.functional as F
from timm.layers import SqueezeExcite

torch._dynamo.config.recompile_limit = 64


class ConvBNAct(nn.Module):
    """Wrapper for Conv + BN + GELU (optional)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bn: bool = True,
        act: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            **kwargs,
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvolutionalStem(nn.Module):
    """Basic convolutional stem without 4x downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels)
        self.conv2 = ConvBNAct(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class OverlapPatchEmbedding(nn.Module):
    """Overlap patch embedding for the first encoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBNAct(in_channels, in_channels)
        self.down = ConvolutionalDownsampling(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.conv(x))


class ConvolutionalUpsampling(nn.Module):
    """
    Upsamle + convolution to prevent checkboard artifacts.
    See: https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBNAct(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class ConvolutionalDownsampling(nn.Module):
    """Downsampling using stride 2 convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = ConvBNAct(in_channels, out_channels, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class RoPE(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        """
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        """
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer("angle", angle)

    def forward(self, slen: Tuple[int]):
        """
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        """
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])  # (h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])  # (w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1)  # (h w d1)
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])  # (h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :])  # (w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1)  # (h w d1)

        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))

        return retention_rel_pos


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 1)
        self.dwconv = nn.Conv2d(
            hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        """
        x: (b h w c)
        """
        x = self.conv1(x)
        x = self.act(x)
        x = x + self.dwconv(x)
        x = self.conv2(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor):
        """
        x: (b c h w)
        """
        x = x.permute(0, 2, 3, 1).contiguous()  # (b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class LinearAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        ffn_ratio: int = 4,
        norm: nn.Module = LayerNorm2d,
        ffn: nn.Module = FeedForwardNetwork,
    ):
        super().__init__()
        self.proj = ConvBNAct(in_channels, out_channels, act=False)
        self.rope = RoPE(out_channels, num_heads)
        self.norm1 = norm(out_channels)
        self.attn = RALA(out_channels, num_heads)
        self.norm2 = norm(out_channels)
        self.ffn = ffn(out_channels, ffn_ratio * out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        _, _, h, w = x.size()
        sin, cos = self.rope((h, w))
        x = x + self.attn(self.norm1(x), sin, cos)
        x = x + self.ffn(self.norm2(x))
        return x


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class RALA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.elu = nn.ELU()
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        """
        x: (b c h w)
        sin: ((h w) d1)
        cos: ((h w) d1)
        """
        B, C, H, W = x.shape
        qkvo = self.qkvo(x)  # (b 3*c h w)
        qkv = qkvo[:, : 3 * self.dim, :, :]
        o = qkvo[:, 3 * self.dim :, :, :]

        q, k, v = rearrange(
            qkv, "b (m n d) h w -> m b n (h w) d", m=3, n=self.num_heads
        )  # (b n (h w) d)

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0  # (b n l d)

        q_mean = q.mean(dim=-2, keepdim=True)  # (b n 1 d)
        eff = self.scale * q_mean @ k.transpose(-1, -2)  # (b n 1 l)
        eff = torch.softmax(eff, dim=-1).transpose(-1, -2)  # (b n l 1)
        k = k * eff * (H * W)

        q_rope = theta_shift(q, sin, cos)
        k_rope = theta_shift(k, sin, cos)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)  # (b n l 1)
        kv = (k_rope.transpose(-2, -1) * ((H * W) ** -0.5)) @ (
            v * ((H * W) ** -0.5)
        )  # (b n d d)

        res = q_rope @ kv * z  # (b n l d)
        res = rearrange(res, "b n (h w) d -> b (n d) h w", h=H, w=W)
        return self.proj(res * o)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = ConvBNAct(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = ConvBNAct(in_channels, out_channels, 1, 1, 0, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn2(x) + identity
        x = self.act(x)
        return x


class ResizingMobileNetBlock(nn.Module):
    """Encoder block with a down/up sampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 2,
        se: bool = True,
        down: bool = True,
    ):
        super().__init__()
        resize = ConvolutionalDownsampling if down else ConvolutionalUpsampling
        self.block = nn.Sequential(
            MobileNetBlock(in_channels, in_channels, expand_ratio, se),
            resize(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileNetBlock(nn.Module):
    """Residual block with dynamic shortcut, used in both encoder and decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 2,
        se: bool = False,
    ):
        super().__init__()
        hidden_channels = out_channels * expand_ratio
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, 1, 0)
        self.dwconv = ConvBNAct(
            hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels
        )
        self.se = SqueezeExcite(hidden_channels, 0.25) if se else nn.Identity()
        self.conv2 = ConvBNAct(hidden_channels, out_channels, 1, 1, 0, act=False)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = ConvBNAct(in_channels, out_channels, 1, 1, 0, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.se(x)
        x = self.conv2(x)
        return x + identity


class SpatialAwareFusionBlock(nn.Module):
    def __init__(self, local_channels, global_channels, expand_ratio=0.25):
        super().__init__()
        channels = local_channels + global_channels

        hidden_channels = int(channels * expand_ratio)
        self.mlp = nn.Sequential(
            ConvBNAct(channels, hidden_channels, 1, 1, 0),
            ConvBNAct(hidden_channels, channels, 1, 1, 0, act=False),
        )
        self.gate = nn.Sigmoid()

    def forward(self, local_feature, global_feature):
        x = torch.cat((local_feature, global_feature), dim=1)
        return x + x * self.gate(self.mlp(x))


class MNet(nn.Module):
    def __init__(
        self,
        in_channel=3,
        num_classes=9,
        deep_supervised=True,
        g_channels=[8, 16, 32, 64, 128],
        l_channels=[16, 32, 64, 128, 256],
        num_heads=[1, 1, 1, 2, 4],
        stem=ConvolutionalStem,
        ope=OverlapPatchEmbedding,
        lb=ResizingMobileNetBlock,
        gb=LinearAttentionBlock,
        fuse=SpatialAwareFusionBlock,
        neck=ResidualBlock,
    ):
        super().__init__()
        self.deep_supervised = deep_supervised
        lg_channels = [x + y for x, y in zip(g_channels, l_channels)]

        self.input_l0 = stem(in_channel, l_channels[0])

        self.encoder1_l1_local = ope(l_channels[0], l_channels[1])
        self.encoder1_l2_local = ope(l_channels[1], l_channels[2])
        self.encoder1_l3_local = ope(l_channels[2], l_channels[3])
        self.encoder1_l4_local = ope(l_channels[3], l_channels[4])

        self.encoder1_l1_global = gb(l_channels[0], g_channels[0], num_heads[0])
        self.encoder1_l2_global = gb(l_channels[1], g_channels[1], num_heads[1])
        self.encoder1_l3_global = gb(l_channels[2], g_channels[2], num_heads[2])
        self.encoder1_l4_global = gb(l_channels[3], g_channels[3], num_heads[3])

        self.decoder1_l4_local = lb(l_channels[4], l_channels[3], down=False)
        self.decoder1_l3_local = lb(lg_channels[3], l_channels[2], down=False)
        self.decoder1_l2_local = lb(lg_channels[2], l_channels[1], down=False)
        self.decoder1_l1_local = lb(lg_channels[1], l_channels[0], down=False)

        self.decoder1_l4_global = gb(l_channels[4], g_channels[4], num_heads[4])
        self.decoder1_l3_global = gb(lg_channels[3], g_channels[3], num_heads[3])
        self.decoder1_l2_global = gb(lg_channels[2], g_channels[2], num_heads[2])
        self.decoder1_l1_global = gb(lg_channels[1], g_channels[1], num_heads[1])

        self.encoder2_l1_local = lb(lg_channels[0], l_channels[1], down=True)
        self.encoder2_l2_local = lb(lg_channels[1], l_channels[2], down=True)
        self.encoder2_l3_local = lb(lg_channels[2], l_channels[3], down=True)
        self.encoder2_l4_local = lb(lg_channels[3], l_channels[4], down=True)

        self.encoder2_l1_global = gb(lg_channels[0], g_channels[0], num_heads[0])
        self.encoder2_l2_global = gb(lg_channels[1], g_channels[1], num_heads[1])
        self.encoder2_l3_global = gb(lg_channels[2], g_channels[2], num_heads[2])
        self.encoder2_l4_global = gb(lg_channels[3], g_channels[3], num_heads[3])

        self.decoder2_l4_local = lb(lg_channels[4], l_channels[3], down=False)
        self.decoder2_l3_local = lb(lg_channels[3], l_channels[2], down=False)
        self.decoder2_l2_local = lb(lg_channels[2], l_channels[1], down=False)
        self.decoder2_l1_local = lb(lg_channels[1], l_channels[0], down=False)

        self.decoder2_l4_local_output = nn.Conv2d(l_channels[4], num_classes, 1, 1, 0)
        self.decoder2_l3_local_output = nn.Conv2d(l_channels[3], num_classes, 1, 1, 0)
        self.decoder2_l2_local_output = nn.Conv2d(l_channels[2], num_classes, 1, 1, 0)
        self.decoder2_l1_local_output = nn.Conv2d(l_channels[1], num_classes, 1, 1, 0)
        self.output_l0 = nn.Sequential(
            neck(lg_channels[0], lg_channels[0]),
            nn.Conv2d(lg_channels[0], num_classes, 1, 1, 0),
        )

        self.x_d1_l3_fuse = fuse(l_channels[3], g_channels[3])
        self.x_d1_l2_fuse = fuse(l_channels[2], g_channels[2])
        self.x_d1_l1_fuse = fuse(l_channels[1], g_channels[1])
        self.x_e2_l0_fuse = fuse(l_channels[0], g_channels[0])
        self.x_e2_l1_fuse = fuse(l_channels[1], g_channels[1])
        self.x_e2_l2_fuse = fuse(l_channels[2], g_channels[2])
        self.x_e2_l3_fuse = fuse(l_channels[3], g_channels[3])
        self.x_e2_l4_fuse = fuse(l_channels[4], g_channels[4])
        self.x_d2_l3_fuse = fuse(l_channels[3], g_channels[3])
        self.x_d2_l2_fuse = fuse(l_channels[2], g_channels[2])
        self.x_d2_l1_fuse = fuse(l_channels[1], g_channels[1])
        self.x_d2_l0_fuse = fuse(l_channels[0], g_channels[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder-decoder 1
        x_e1_l0 = self.input_l0(x)

        x_e1_l1_local = self.encoder1_l1_local(x_e1_l0)
        x_e1_l0_global = self.encoder1_l1_global(x_e1_l0)

        x_e1_l2_local = self.encoder1_l2_local(x_e1_l1_local)
        x_e1_l1_global = self.encoder1_l2_global(x_e1_l1_local)

        x_e1_l3_local = self.encoder1_l3_local(x_e1_l2_local)
        x_e1_l2_global = self.encoder1_l3_global(x_e1_l2_local)

        x_e1_l4_local = self.encoder1_l4_local(x_e1_l3_local)
        x_e1_l3_global = self.encoder1_l4_global(x_e1_l3_local)

        x_d1_l3_local = self.decoder1_l4_local(x_e1_l4_local)
        x_d1_l4_global = self.decoder1_l4_global(x_e1_l4_local)

        x_d1_l3 = self.x_d1_l3_fuse(x_d1_l3_local, x_e1_l3_global)
        x_d1_l2_local = self.decoder1_l3_local(x_d1_l3)
        x_d1_l3_global = self.decoder1_l3_global(x_d1_l3)

        x_d1_l2 = self.x_d1_l2_fuse(x_d1_l2_local, x_e1_l2_global)
        x_d1_l1_local = self.decoder1_l2_local(x_d1_l2)
        x_d1_l2_global = self.decoder1_l2_global(x_d1_l2)

        x_d1_l1 = self.x_d1_l1_fuse(x_d1_l1_local, x_e1_l1_global)
        x_d1_l0_local = self.decoder1_l1_local(x_d1_l1)
        x_d1_l1_global = self.decoder1_l1_global(x_d1_l1)

        # encoder-decoder 2
        x_e2_l0 = self.x_e2_l0_fuse(x_d1_l0_local, x_e1_l0_global)
        x_e2_l1_local = self.encoder2_l1_local(x_e2_l0)
        x_e2_l0_global = self.encoder2_l1_global(x_e2_l0)

        x_e2_l1 = self.x_e2_l1_fuse(x_e2_l1_local, x_d1_l1_global)
        x_e2_l2_local = self.encoder2_l2_local(x_e2_l1)
        x_e2_l1_global = self.encoder2_l2_global(x_e2_l1)

        x_e2_l2 = self.x_e2_l2_fuse(x_e2_l2_local, x_d1_l2_global)
        x_e2_l3_local = self.encoder2_l3_local(x_e2_l2)
        x_e2_l2_global = self.encoder2_l3_global(x_e2_l2)

        x_e2_l3 = self.x_e2_l3_fuse(x_e2_l3_local, x_d1_l3_global)
        x_e2_l4_local = self.encoder2_l4_local(x_e2_l3)
        x_e2_l3_global = self.encoder2_l4_global(x_e2_l3)

        output_l4 = self.decoder2_l4_local_output(x_e2_l4_local)
        x_e2_l4 = self.x_e2_l4_fuse(x_e2_l4_local, x_d1_l4_global)
        x_d2_l3_local = self.decoder2_l4_local(x_e2_l4)

        output_l3 = self.decoder2_l3_local_output(x_d2_l3_local)
        x_d2_l3 = self.x_d2_l3_fuse(x_d2_l3_local, x_e2_l3_global)
        x_d2_l2_local = self.decoder2_l3_local(x_d2_l3)

        output_l2 = self.decoder2_l2_local_output(x_d2_l2_local)
        x_d2_l2 = self.x_d2_l2_fuse(x_d2_l2_local, x_e2_l2_global)
        x_d2_l1_local = self.decoder2_l2_local(x_d2_l2)

        output_l1 = self.decoder2_l1_local_output(x_d2_l1_local)
        x_d2_l1 = self.x_d2_l1_fuse(x_d2_l1_local, x_e2_l1_global)
        x_d2_l0_local = self.decoder2_l1_local(x_d2_l1)

        x_d2_l0 = self.x_d2_l0_fuse(x_d2_l0_local, x_e2_l0_global)
        output_l0 = self.output_l0(x_d2_l0)

        if self.deep_supervised:
            return (output_l0, output_l1, output_l2, output_l3, output_l4)
        else:
            return output_l0


class MNetS(nn.Module):
    def __init__(
        self,
        in_channel=3,
        num_classes=9,
        deep_supervised=True,
        g_channels=[12, 24, 48, 96, 192],
        l_channels=[24, 48, 96, 192, 384],
        num_heads=[1, 1, 1, 2, 4],
        stem=ConvolutionalStem,
        lb=ResizingMobileNetBlock,
        gb=LinearAttentionBlock,
        fuse=SpatialAwareFusionBlock,
        neck=ResidualBlock,
    ):
        super().__init__()
        self.deep_supervised = deep_supervised
        lg_channels = [x + y for x, y in zip(g_channels, l_channels)]

        self.input_l0 = stem(in_channel, l_channels[0])

        self.encoder1_l1_local = lb(l_channels[0], l_channels[1], down=True)
        self.encoder1_l2_local = lb(l_channels[1], l_channels[2], down=True)
        self.encoder1_l3_local = lb(l_channels[2], l_channels[3], down=True)
        self.encoder1_l4_local = lb(l_channels[3], l_channels[4], down=True)

        self.encoder1_l1_global = gb(l_channels[0], g_channels[0], num_heads[0])
        self.encoder1_l2_global = gb(l_channels[1], g_channels[1], num_heads[1])
        self.encoder1_l3_global = gb(l_channels[2], g_channels[2], num_heads[2])
        self.encoder1_l4_global = gb(l_channels[3], g_channels[3], num_heads[3])

        self.decoder1_l4_local = lb(l_channels[4], l_channels[3], down=False)
        self.decoder1_l3_local = lb(lg_channels[3], l_channels[2], down=False)
        self.decoder1_l2_local = lb(lg_channels[2], l_channels[1], down=False)
        self.decoder1_l1_local = lb(lg_channels[1], l_channels[0], down=False)

        self.decoder1_l4_local_output = nn.Conv2d(l_channels[4], num_classes, 1, 1, 0)
        self.decoder1_l3_local_output = nn.Conv2d(l_channels[3], num_classes, 1, 1, 0)
        self.decoder1_l2_local_output = nn.Conv2d(l_channels[2], num_classes, 1, 1, 0)
        self.decoder1_l1_local_output = nn.Conv2d(l_channels[1], num_classes, 1, 1, 0)
        self.output_l0 = nn.Sequential(
            neck(lg_channels[0], lg_channels[0]),
            nn.Conv2d(lg_channels[0], num_classes, 1, 1, 0),
        )

        self.x_d1_l3_fuse = fuse(l_channels[3], g_channels[3])
        self.x_d1_l2_fuse = fuse(l_channels[2], g_channels[2])
        self.x_d1_l1_fuse = fuse(l_channels[1], g_channels[1])
        self.x_e2_l0_fuse = fuse(l_channels[0], g_channels[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_e1_l0 = self.input_l0(x)

        x_e1_l1_local = self.encoder1_l1_local(x_e1_l0)
        x_e1_l0_global = self.encoder1_l1_global(x_e1_l0)

        x_e1_l2_local = self.encoder1_l2_local(x_e1_l1_local)
        x_e1_l1_global = self.encoder1_l2_global(x_e1_l1_local)

        x_e1_l3_local = self.encoder1_l3_local(x_e1_l2_local)
        x_e1_l2_global = self.encoder1_l3_global(x_e1_l2_local)

        x_e1_l4_local = self.encoder1_l4_local(x_e1_l3_local)
        x_e1_l3_global = self.encoder1_l4_global(x_e1_l3_local)

        x_d1_l3_local = self.decoder1_l4_local(x_e1_l4_local)

        x_d1_l3 = self.x_d1_l3_fuse(x_d1_l3_local, x_e1_l3_global)
        x_d1_l2_local = self.decoder1_l3_local(x_d1_l3)

        x_d1_l2 = self.x_d1_l2_fuse(x_d1_l2_local, x_e1_l2_global)
        x_d1_l1_local = self.decoder1_l2_local(x_d1_l2)

        x_d1_l1 = self.x_d1_l1_fuse(x_d1_l1_local, x_e1_l1_global)
        x_d1_l0_local = self.decoder1_l1_local(x_d1_l1)

        x_e2_l0 = self.x_e2_l0_fuse(x_d1_l0_local, x_e1_l0_global)
        output_l0 = self.output_l0(x_e2_l0)
        output_l4 = self.decoder1_l4_local_output(x_e1_l4_local)
        output_l3 = self.decoder1_l3_local_output(x_d1_l3_local)
        output_l2 = self.decoder1_l2_local_output(x_d1_l2_local)
        output_l1 = self.decoder1_l1_local_output(x_d1_l1_local)

        if self.deep_supervised:
            return (output_l0, output_l1, output_l2, output_l3, output_l4)
        else:
            return output_l0


def init_weights(m, a=1e-2):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", a=a)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)


class nnUNetTrainer_MNet(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.initial_lr = 4 * 1e-4
        self.num_epochs = 1000

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    ):
        model = MNet(
            in_channel=num_input_channels,
            num_classes=num_output_channels,
            deep_supervised=enable_deep_supervision,
        )
        model.apply(init_weights)
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.deep_supervised = enabled

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(), self.initial_lr, weight_decay=0.05
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnUNetTrainer_MNetS(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.initial_lr = 4 * 1e-4
        self.num_epochs = 1000

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    ):
        model = MNetS(
            in_channel=num_input_channels,
            num_classes=num_output_channels,
            deep_supervised=enable_deep_supervision,
        )
        model.apply(init_weights)
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.deep_supervised = enabled

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(), self.initial_lr, weight_decay=0.05
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


if __name__ == "__main__":
    input_size = (1, 3, 512, 512)
    # model = MNet().cuda()
    # output = model(torch.rand(*input_size).cuda())
    # print("Input size :", input_size)
    # print("Output size:", [x.shape for x in output])
