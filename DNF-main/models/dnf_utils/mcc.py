# from einops import rearrange
# import torch
# from torch import nn
# import torch.nn.functional as F
#
# from ..utils import LayerNorm
#
# class MCC(nn.Module):
#     """改进版色彩校正模块：引导信息用于控制 attention mask"""
#
#     def __init__(self, f_number, num_heads, padding_mode='reflect', bias=False):
#         super().__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.norm_x = LayerNorm(f_number, eps=1e-6, data_format='channels_first')
#         self.norm_guide = LayerNorm(f_number, eps=1e-6, data_format='channels_first')
#
#         self.guide_conv = nn.Sequential(
#             nn.Conv2d(f_number, f_number, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode),
#             nn.GELU()
#         )
#
#         self.attn_mask_gen = nn.Sequential(
#             nn.Conv2d(f_number, f_number, kernel_size=3, padding=1, bias=bias),
#             nn.Sigmoid()
#         )
#
#         self.qkv_proj = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)
#         self.dwconv = nn.Conv2d(f_number * 3, f_number * 3, kernel_size=3, stride=1, padding=1,
#                                 bias=bias, padding_mode=padding_mode, groups=f_number * 3)
#
#         self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)
#
#         self.feedforward = nn.Sequential(
#             nn.Conv2d(f_number, f_number, kernel_size=1, stride=1, padding=0, bias=bias),
#             nn.GELU(),
#             nn.Conv2d(f_number, f_number, kernel_size=3, stride=1, padding=1, bias=bias, groups=f_number, padding_mode=padding_mode),
#             nn.GELU()
#         )
#
#         self.res_gate = nn.Parameter(torch.tensor(0.5))  # 控制残差权重
#
#     def forward(self, x, guide):
#         B, C, H, W = x.shape
#
#         x = self.norm_x(x)
#         guide_feat = self.norm_guide(self.guide_conv(guide))
#
#         # 生成注意力调控 mask
#         attn_mask = self.attn_mask_gen(guide_feat)  # [B, C, H, W]
#
#         # 用 attn_mask 调控输入特征，稳定融合
#         x_mod = x * (1 + attn_mask)
#
#         # qkv attention
#         qkv = self.dwconv(self.qkv_proj(x_mod))
#         q, k, v = qkv.chunk(3, dim=1)
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         # 对 q/k 做 normalize
#         q = F.normalize(q, dim=-1)
#         k = F.normalize(k, dim=-1)
#
#         # 计算注意力
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         out = attn @ v
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
#
#         out = self.project_out(out)
#
#         # 残差 + FFN
#         out = self.feedforward(out + self.res_gate * x)
#
#         return out

from einops import rearrange
import torch
from torch import nn

from ..utils import LayerNorm

class MCC(nn.Module):
    def __init__(self, f_number, num_heads, padding_mode, bias=False) -> None:
        super().__init__()
        self.norm = LayerNorm(f_number, eps=1e-6, data_format='channels_first')

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.pwconv = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(f_number * 3, f_number * 3, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=f_number * 3)
        self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)
        self.feedforward = nn.Sequential(
            nn.Conv2d(f_number, f_number, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv2d(f_number, f_number, 3, 1, 1, bias=bias, groups=f_number, padding_mode=padding_mode),
            nn.GELU()
        )

    def forward(self, x):

        attn = self.norm(x)

        _, _, h, w = attn.shape

        qkv = self.dwconv(self.pwconv(attn))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.feedforward(out + x)
        # print(f"After out: {out.shape}")
        return out

