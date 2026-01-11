import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class GAF(nn.Module):
    def __init__(self, in_channels, feature_num=2, bias=True, padding_mode='reflect', **kwargs):
        super().__init__()
        self.feature_num = feature_num
        hidden_features = in_channels * feature_num
        reduction = max(4, in_channels // 16)  # 防止过度压缩

        self.pwconv = nn.Conv2d(hidden_features, hidden_features * 2, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, bias=bias, padding_mode=padding_mode,
                                groups=hidden_features * 2)
        self.project_out = nn.Conv2d(hidden_features, in_channels, kernel_size=1, bias=bias)

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_texture, x_color):
        shortcut_texture = x_texture
        shortcut_color = x_color  # 保留原始信息

        # **1. 先施加通道注意力**
        x_texture = x_texture * self.channel_attention(x_texture)
        x_color = x_color * self.channel_attention(x_color)

        # **2. 特征提取**
        x = torch.cat([x_texture, x_color], dim=1)
        x = self.pwconv(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        # **3. 施加空间注意力**
        spatial_input = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x = x * spatial_att

        # **4. 残差连接，确保融合**
        return x + shortcut_texture + shortcut_color  # 保持纹理+色彩信息

