import torch
import torch.nn as nn
import torch.nn.functional as F

class Inte(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super(Inte, self).__init__()
        d = max(in_channels // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局池化

        # 通道压缩 + 激活
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, d, 1, bias=bias),
            nn.LeakyReLU(0.2)
        )

        # 两个分支各自的注意力生成器
        self.fc_local = nn.Conv2d(d, in_channels, 1, bias=bias)
        self.fc_global = nn.Conv2d(d, in_channels, 1, bias=bias)

        # self.softmax = nn.Softmax(dim=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self,local_feat, global_feat):
        # 输入特征 [B, C, H, W]
        batch, C, H, W = local_feat.size()

        # 拼接两个分支，形成 [B, 2*C, H, W]
        feats = torch.cat([local_feat, global_feat], dim=1)

        # 重塑为 [B, 2, C, H, W]
        feats = feats.view(batch, 2, C, H, W)

        # 融合两个分支的特征（求和）
        feats_sum = torch.sum(feats, dim=1)  # [B, C, H, W]

        feats_pool = self.avg_pool(feats_sum)  # [B, C, 1, 1]

        feats_compress = self.conv_du(feats_pool)  # [B, d, 1, 1]

        attn_local = self.sigmoid(self.fc_local(feats_compress))  # [B, C, 1, 1]
        attn_global = self.sigmoid(self.fc_global(feats_compress))  # [B, C, 1, 1]

        out = local_feat * attn_local + global_feat * attn_global

        return out
