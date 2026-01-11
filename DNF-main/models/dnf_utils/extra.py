import torch
import torch.nn as nn


class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
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
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x


##########################################################################
### --------- Residual Context Block (RCB) ----------
class Global(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(Global, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = act

        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res += x
        return res

# class Local(nn.Module):
#     def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
#         super(Local, self).__init__()
#
#         act = nn.LeakyReLU(0.2)
#
#         self.body = nn.Sequential(
#             nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
#             act,
#             nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
#         )
#
#         self.act = act
#
#     def forward(self, x):
#         res1 = self.body(x)
#         res1 = self.act(res1)
#         res1 += x
#         return res1

class Local(nn.Module):
    """
    多尺度局部信息增强模块
    - 支持通道数变化 32-512
    - 包含局部卷积特征、全局信息引导、通道注意力、空间注意力
    - 自动调整 reduction 保证中间通道维度合理
    """
    def __init__(self, in_channels, base_reduction=8):
        super(Local, self).__init__()

        # 动态计算 reduction，保证中间通道维度合理
        reduction = max(4, min(base_reduction, in_channels // 16))
        mid_channels = max(4, in_channels // reduction)

        # 局部卷积特征提取
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )

        # 全局信息引导
        self.global_guidance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 通道注意力（自适应 reduction）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # 输出融合卷积
        self.fuse = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 局部卷积特征
        local_feat = self.local_conv(x)

        # 全局信息引导
        global_weight = self.global_guidance(x)
        guided_feat = local_feat * global_weight

        # 通道注意力
        ca = self.channel_att(guided_feat)
        feat_ca = guided_feat * ca

        # 空间注意力
        avg_out = torch.mean(feat_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(feat_ca, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        feat_sa = feat_ca * sa

        # 融合残差输出
        out = self.fuse(feat_sa + x)

        return out

