# from einops import rearrange
# import torch
# from torch import nn
# import torch.nn.functional as F
#
# from ..utils import LayerNorm
#
# class TEN(nn.Module):
#     """改进版纹理增强模块：guide 生成注意力调控"""
#
#     def __init__(self, f_number, num_heads, padding_mode='reflect', bias=False):
#         super().__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
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
#         self.alpha = nn.Parameter(torch.ones(1, f_number, 1, 1))  # 融合控制
#         self.texture_conv = nn.Conv2d(f_number, f_number, kernel_size=3, padding=1, bias=bias)
#
#         self.feedforward = nn.Sequential(
#             nn.Conv2d(f_number, f_number, kernel_size=1, padding=0, bias=bias),
#             nn.GELU(),
#             nn.Conv2d(f_number, f_number, kernel_size=3, padding=1, bias=bias, groups=f_number, padding_mode=padding_mode),
#             nn.GELU()
#         )
#
#         self.res_gate = nn.Parameter(torch.tensor(0.5))  # 可调节残差比例
#
#     def forward(self, x, guide):
#         B, C, H, W = x.shape
#
#         # print("forward input to norm_x:", x.shape)
#         # print("norm weight shape:", self.norm_x.weight.shape)
#         # x = self.norm_x(x)
#
#         x = self.norm_x(x)
#
#         guide_feat = self.norm_guide(self.guide_conv(guide))
#
#         attn_mask = self.attn_mask_gen(guide_feat)  # [B, C, H, W]
#
#         x_mod = x * (1 + attn_mask)
#
#         # QKV projection
#         qkv = self.dwconv(self.qkv_proj(x_mod))
#         q, k, v = qkv.chunk(3, dim=1)
#
#         # 多头 reshape
#         q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
#         k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
#         v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
#
#         # 注意力计算
#         q = F.normalize(q, dim=-1)
#         k = F.normalize(k, dim=-1)
#         attn = torch.softmax((q @ k.transpose(-2, -1)) * self.temperature, dim=-1)
#
#         out = attn @ v
#         out = rearrange(out, 'b h c (h1 w1) -> b (h c) h1 w1', h=self.num_heads, h1=H, w1=W)
#
#         # 融合注意力特征和纹理特征
#         texture_feat = self.texture_conv(x_mod)
#         alpha = torch.clamp(self.alpha, 0.1, 1.0)
#         out = alpha * out + (1 - alpha) * texture_feat
#
#         # 残差连接 + Feedforward
#         out = self.feedforward(out + self.res_gate * x)
#
#         return out


# from einops import rearrange
# import torch
# from torch import nn
#
# from ..utils import LayerNorm
#
# class TEN(nn.Module):
#     def __init__(self, f_number, num_heads, padding_mode, bias=False) -> None:
#         super().__init__()
#         self.norm = LayerNorm(f_number, eps=1e-6, data_format='channels_first')
#
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.pwconv = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)
#         self.dwconv = nn.Conv2d(f_number * 3, f_number * 3, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=f_number * 3)
#         self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)
#         self.feedforward = nn.Sequential(
#             nn.Conv2d(f_number, f_number, 1, 1, 0, bias=bias),
#             nn.GELU(),
#             nn.Conv2d(f_number, f_number, 3, 1, 1, bias=bias, groups=f_number, padding_mode=padding_mode),
#             nn.GELU()
#         )
#
#     def forward(self, x):
#
#         attn = self.norm(x)
#
#         _, _, h, w = attn.shape
#
#         qkv = self.dwconv(self.pwconv(attn))
#         q, k, v = qkv.chunk(3, dim=1)
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         out = self.project_out(out)
#         out = self.feedforward(out + x)
#         # print(f"After out: {out.shape}")
#         return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..utils import LayerNorm

class TEN(nn.Module):
    """ 仅关注 纹理细节增强 """

    def __init__(self, f_number, num_heads, padding_mode='reflect', bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # **使用 BatchNorm2d 归一化**
        # self.norm = nn.BatchNorm2d(f_number)
        self.norm = LayerNorm(f_number, eps=1e-6, data_format='channels_first')

        # **调整 pwconv 以匹配拼接后的通道数**
        self.pwconv = nn.Conv2d(f_number * 2, f_number * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(f_number * 3, f_number * 3, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode, groups=f_number * 3)

        # **额外的梯度增强模块**
        self.texture_conv = nn.Conv2d(f_number, f_number, kernel_size=3, stride=1, padding=1, bias=bias)

        # **注册 Sobel 卷积核**
        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                       [ 0,  0,  0],
                                       [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_kernel_x", sobel_kernel_x)
        self.register_buffer("sobel_kernel_y", sobel_kernel_y)

        # **纹理自适应增强**
        self.alpha = nn.Parameter(torch.ones(1))  # 纹理增强平衡权重
        self.feedforward = nn.Sequential(
            nn.Conv2d(f_number, f_number, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.GELU(),
            nn.Conv2d(f_number, f_number, kernel_size=3, stride=1, padding=1, bias=bias, groups=f_number, padding_mode=padding_mode),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: 直接输入去噪后的图像（不使用 residual）
        """

        # **对 x 进行归一化**
        # print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{")
        # print(x.shape)
        x = self.norm(x)

        # **计算梯度信息（直接基于 x）**
        sobel_x = F.conv2d(x, self.sobel_kernel_x.expand(x.shape[1], -1, -1, -1), padding=1, groups=x.shape[1])
        sobel_y = F.conv2d(x, self.sobel_kernel_y.expand(x.shape[1], -1, -1, -1), padding=1, groups=x.shape[1])
        texture_info = torch.sqrt(sobel_x ** 2 + sobel_y ** 2 + 1e-6)

        # **对梯度进行归一化**
        texture_info = texture_info / (texture_info.max(dim=-1, keepdim=True)[0] + 1e-6)  # 避免梯度值过大

        # **拼接去噪后的图像和梯度增强信息**
        combined_input = torch.cat([x, texture_info], dim=1)  # [batch, f_number * 2, H, W]

        # **通过 pwconv 处理拼接后的数据**
        qkv = self.dwconv(self.pwconv(combined_input))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=x.shape[2], w=x.shape[3])

        # **纹理增强**
        texture_feat = self.texture_conv(texture_info)
        alpha = torch.clamp(self.alpha, 0.1, 1.0)  # **确保 alpha 不会太小或太大**
        out = alpha * out + (1 - alpha) * texture_feat

        # **最终优化**
        out = self.feedforward(out + x)
        return out
