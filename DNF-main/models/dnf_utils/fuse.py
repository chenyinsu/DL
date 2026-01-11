import torch
from torch import nn
from torch.nn import functional as F

class PDConvFuse(nn.Module):
    def __init__(self, in_channels=None, f_number=None, feature_num=2, bias=True, **kwargs) -> None:
        super().__init__()
        if in_channels is None:
            assert f_number is not None
            in_channels = f_number
        self.feature_num = feature_num
        self.act = nn.GELU()
        self.pwconv = nn.Conv2d(feature_num * in_channels, in_channels, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias, groups=in_channels, padding_mode='reflect')

    def forward(self, *inp_feats):
        assert len(inp_feats) == self.feature_num
        return self.dwconv(self.act(self.pwconv(torch.cat(inp_feats, dim=1))))

class GFM(nn.Module):
    def __init__(self, in_channels, feature_num=2, bias=True, padding_mode='reflect', **kwargs) -> None:
        super().__init__()
        self.feature_num = feature_num

        hidden_features = in_channels * feature_num
        self.pwconv = nn.Conv2d(hidden_features, hidden_features * 2, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=hidden_features * 2)
        self.project_out = nn.Conv2d(hidden_features, in_channels, kernel_size=1, bias=bias)
        self.mlp = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)

    def forward(self, *inp_feats):
        assert len(inp_feats) == self.feature_num
        shortcut = inp_feats[0]
        x = torch.cat(inp_feats, dim=1)
        x = self.pwconv(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return self.mlp(x + shortcut)


# class GFM(nn.Module):
#     """ 两个完全独立的门控（局部+全局） """
#     def __init__(self, in_channels, feature_num=2, bias=True, reduction=8):
#         super().__init__()
#         self.feature_num = feature_num
#         self.in_channels = in_channels
#         self.reduction = reduction
#         self.bias = bias
#
#         hidden_features = in_channels * feature_num
#
#         # 共享卷积特征提取
#         self.pwconv = nn.Conv2d(hidden_features, hidden_features * 2, 1, bias=bias)
#         self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, padding=1,
#                                 groups=hidden_features * 2, bias=bias)
#         self.project_out = nn.Conv2d(hidden_features, in_channels, 1, bias=bias)
#         self.mlp = nn.Conv2d(in_channels, in_channels, 1, bias=True)
#
#         # 局部门控
#         hidden_gate = max(4, in_channels // reduction)
#         self.local_gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, hidden_gate, 1, bias=bias),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_gate, in_channels, 1, bias=bias),
#             nn.Sigmoid()
#         )
#
#         # 全局门控
#         self.global_gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, hidden_gate, 1, bias=bias),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_gate, in_channels, 1, bias=bias),
#             nn.Sigmoid()
#         )
#
#     def forward(self, *inp_feats):
#         assert len(inp_feats) == self.feature_num
#         shortcut = inp_feats[0]
#
#         # 拼接输入
#         x = torch.cat(inp_feats, dim=1)
#
#         # 共享卷积特征
#         x = self.pwconv(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         F_s = self.mlp(x + shortcut)
#
#         # 两个独立门控
#         F_local = F_s * self.local_gate(F_s)
#         F_global = F_s * self.global_gate(F_s)
#
#         return F_local, F_global


# class GFM(nn.Module):
#     """
#     升级版 GFM：融合局部卷积和全局注意力
#     - 局部信息：卷积（pwconv + dwconv）
#     - 全局信息：通道注意力
#     """
#     def __init__(self, in_channels, feature_num=2, reduction=8, bias=True, padding_mode='reflect'):
#         super().__init__()
#         self.feature_num = feature_num
#
#         hidden_channels = in_channels * feature_num
#
#         # 局部卷积融合
#         self.pwconv = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=1, bias=bias)
#         self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1,
#                                 bias=bias, padding_mode=padding_mode, groups=hidden_channels * 2)
#         self.project_out = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=bias)
#
#         # 通道注意力（全局信息）
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.attn_fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True)
#         self.attn_fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True)
#         self.sigmoid = nn.Sigmoid()
#
#         # 最终融合残差
#         self.mlp = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
#
#     def forward(self, *inp_feats):
#         assert len(inp_feats) == self.feature_num
#         shortcut = inp_feats[0]
#
#         # 拼接输入
#         x = torch.cat(inp_feats, dim=1)
#
#         # 局部卷积融合
#         x = self.pwconv(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#
#         # 全局通道注意力
#         attn = self.global_pool(x)
#         attn = F.gelu(self.attn_fc1(attn))
#         attn = self.attn_fc2(attn)
#         attn = self.sigmoid(attn)
#
#         x = x + x * attn  # 局部+全局融合
#
#         # 残差连接
#         out = self.mlp(x + shortcut)
#         return out