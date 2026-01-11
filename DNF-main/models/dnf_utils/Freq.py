
# import torch
# import torch.nn as nn
#
#
# class AdaptiveFreqSplit(nn.Module):
#     def __init__(self, in_channels):
#         super(AdaptiveFreqSplit, self).__init__()
#         # 低频卷积：可学习的深度可分离卷积，初始化为平滑卷积核
#         self.low_conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)
#         # 高频卷积：初始化为拉普拉斯卷积核
#         self.high_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
#
#         # 门控机制：1x1卷积，生成权重图
#         self.gate_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#
#         self._init_kernels()
#
#     def _init_kernels(self):
#         # 初始化低频卷积核为均值滤波器（平滑）
#         low_kernel = torch.ones(1, 1, 5, 5) / 25.0
#         self.low_conv.weight.data.copy_(low_kernel.repeat(self.low_conv.out_channels, 1, 1, 1))
#
#         # 初始化高频卷积核为拉普拉斯算子
#         laplacian_kernel = torch.tensor([[0, -1, 0],
#                                          [-1, 4, -1],
#                                          [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         self.high_conv.weight.data.copy_(laplacian_kernel.repeat(self.high_conv.out_channels, 1, 1, 1))
#
#     def forward(self, x):
#         # 计算低频特征
#         low_feat = self.low_conv(x)
#         # 计算高频特征
#         high_feat = self.high_conv(x)
#
#         # 计算门控权重（0~1）
#         gate = self.sigmoid(self.gate_conv(x))
#
#         # 低频和高频特征自适应融合
#         x_low = low_feat * gate
#         x_high = high_feat * (1 - gate)
#
#         return x_low, x_high

import torch
import torch.nn as nn


class DualGateFreqSplit(nn.Module):
    """
    双门控频域分离模块（单输入版本）
    - 输入: x
    - 输出: x_low（低频）、x_high（高频）
    - 特点: 分别为低频和高频设计独立的门控通道
    """
    def __init__(self, in_channels):
        super(DualGateFreqSplit, self).__init__()

        # 低频卷积：平滑卷积（Depthwise）
        self.low_conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)
        # 高频卷积：拉普拉斯卷积（Depthwise）
        self.high_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

        # 双门控：低频门和高频门
        self.gate_low = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.gate_high = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # 初始化卷积核
        self._init_kernels()

    def _init_kernels(self):
        # 初始化低频卷积核为均值滤波器
        low_kernel = torch.ones(1, 1, 5, 5) / 25.0
        self.low_conv.weight.data.copy_(low_kernel.repeat(self.low_conv.out_channels, 1, 1, 1))

        # 初始化高频卷积核为拉普拉斯算子
        laplacian_kernel = torch.tensor([[0, -1, 0],
                                         [-1, 4, -1],
                                         [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.high_conv.weight.data.copy_(laplacian_kernel.repeat(self.high_conv.out_channels, 1, 1, 1))

    def forward(self, x):
        """
        输入：
            x: 特征图 [B, C, H, W]
        输出：
            x_low: 加权后的低频分量
            x_high: 加权后的高频分量
        """
        # 提取低频与高频特征
        low_feat = self.low_conv(x)
        high_feat = self.high_conv(x)

        # 双门控机制（独立学习低频与高频的权重）
        gate_low = self.sigmoid(self.gate_low(x))     # 控制低频信息保留
        gate_high = self.sigmoid(self.gate_high(x))   # 控制高频信息保留

        # 分别加权融合
        x_low = low_feat * gate_low
        x_high = high_feat * gate_high

        return x_low, x_high





