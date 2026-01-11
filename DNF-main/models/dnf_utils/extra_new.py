from einops import rearrange
import torch
from torch import nn

from ..utils import LayerNorm


class EXtra(nn.Module):
    def __init__(self, f_number, num_heads, padding_mode, bias=False) -> None:
        super().__init__()
        self.norm = LayerNorm(f_number, eps=1e-6, data_format='channels_first')
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.pwconv = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(f_number * 3, f_number * 3, 3, 1, 1, bias=bias, padding_mode=padding_mode,
                                groups=f_number * 3)
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
        return out

class Local(nn.Module):
    def __init__(self, f_number, kernel_size=3, reduction=8, bias=False, groups=1):
        super(Local, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(f_number, f_number, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(f_number, f_number, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = act

    def forward(self, x):
        res1 = self.body(x)
        res1 = self.act(res1)
        res1 += x
        return res1

class AdaptiveFreqSplit(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveFreqSplit, self).__init__()
        self.low_conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)
        self.high_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.gate_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self._init_kernels()

    def _init_kernels(self):
        low_kernel = torch.ones(1, 1, 5, 5) / 25.0
        self.low_conv.weight.data.copy_(low_kernel.repeat(self.low_conv.out_channels, 1, 1, 1))
        laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)
        self.high_conv.weight.data.copy_(laplacian_kernel.repeat(self.high_conv.out_channels, 1, 1, 1))

    def forward(self, x):
        low_feat = self.low_conv(x)
        high_feat = self.high_conv(x)
        gate = self.sigmoid(self.gate_conv(x))
        x_low = low_feat * gate
        x_high = high_feat * (1 - gate)
        return x_low, x_high


class FrequencyAwareInte(nn.Module):
    def __init__(self, f_number, num_heads, padding_mode, reduction=8, bias=False):
        super().__init__()
        d = max(f_number // reduction, 4)

        # 1. 特征提取：局部（高频主导）和全局（低频主导）
        self.local_conv = Local(f_number)  # 局部特征
        self.extr = EXtra(f_number, num_heads, padding_mode)  # 全局特征

        # 2. 分别为局部和全局特征创建分频模块（输入通道均为f_number，解决通道匹配问题）
        self.freq_local = AdaptiveFreqSplit(in_channels=f_number)  # 局部特征分频
        self.freq_global = AdaptiveFreqSplit(in_channels=f_number)  # 全局特征分频

        # 3. 同频率自适应融合门控（核心修改：分别融合低频和高频）
        # 低频融合门控（局部低频 + 全局低频）
        self.low_fusion_gate = nn.Sequential(
            nn.Conv2d(f_number * 2, d, kernel_size=5, padding=2, bias=bias),  # 大核捕捉低频关联
            nn.GELU(),
            nn.Conv2d(d, f_number, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )

        # 高频融合门控（局部高频 + 全局高频）
        self.high_fusion_gate = nn.Sequential(
            nn.Conv2d(f_number * 2, d, kernel_size=3, padding=1, bias=bias),  # 小核捕捉高频细节
            nn.GELU(),
            nn.Conv2d(d, f_number, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 步骤1：提取局部和全局特征
        local_feat = self.local_conv(x)  # [B, f_number, H, W]（局部特征）
        global_feat = self.extr(x)  # [B, f_number, H, W]（全局特征）

        # 步骤2：局部和全局特征分别分频（解决原代码通道不匹配问题）
        local_low, local_high = self.freq_local(local_feat)  # 局部低频、局部高频
        global_low, global_high = self.freq_global(global_feat)  # 全局低频、全局高频

        # 步骤3：同频率特征自适应融合
        # 低频融合：局部低频（小范围平滑）与全局低频（大范围趋势）融合
        low_cat = torch.cat([local_low, global_low], dim=1)  # [B, 2*f_number, H, W]
        low_weight = self.low_fusion_gate(low_cat)  # [B, f_number, H, W]（低频融合权重）
        fused_low = local_low * low_weight + global_low * (1 - low_weight)  # 自适应平衡

        # 高频融合：局部高频（细纹理）与全局高频（大边缘）融合
        high_cat = torch.cat([local_high, global_high], dim=1)  # [B, 2*f_number, H, W]
        high_weight = self.high_fusion_gate(high_cat)  # [B, f_number, H, W]（高频融合权重）
        fused_high = local_high * high_weight + global_high * (1 - high_weight)  # 自适应平衡

        return fused_low, fused_high  # 输出融合后的高频和低频特征