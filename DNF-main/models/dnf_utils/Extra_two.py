import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbeddingFast2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0, "维度必须能被4整除 (x/y 各占一半)"
        self.dim = dim
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, q, k, H, W):
        B, heads, N, dim_h = q.shape
        half_dim = self.dim // 2
        freq_seq = torch.arange(0, half_dim, 2, device=q.device).float()
        inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))

        x_pos = torch.arange(W, device=q.device).float()
        y_pos = torch.arange(H, device=q.device).float()
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing="ij")
        x_grid = x_grid.reshape(-1, 1)  # [H*W, 1]
        y_grid = y_grid.reshape(-1, 1)  # [H*W, 1]

        x_grid = x_grid * self.scale
        y_grid = y_grid * self.scale

        sin_x = torch.sin(x_grid * inv_freq)
        cos_x = torch.cos(x_grid * inv_freq)
        sin_y = torch.sin(y_grid * inv_freq)
        cos_y = torch.cos(y_grid * inv_freq)

        sin = torch.cat([sin_x, sin_y], dim=-1).unsqueeze(0)  # [1, H*W, dim//2]
        cos = torch.cat([cos_x, cos_y], dim=-1).unsqueeze(0)  # [1, H*W, dim//2]

        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]

        q_rotated = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rotated = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

        return q_rotated, k_rotated


class BiGatedLinearAttention(nn.Module):
    def __init__(self, f_number, num_heads, dropout=0.0):
        super().__init__()
        self.f_number = f_number
        self.num_heads = num_heads
        self.head_dim = f_number // num_heads
        self.scale = self.head_dim ** -0.5

        self.attn_norm = nn.LayerNorm(f_number)  # Transformer部分层归一化
        self.qkv = nn.Linear(f_number, f_number * 3, bias=False)
        nn.init.xavier_uniform_(self.qkv.weight)

        self.proj = nn.Linear(f_number, f_number)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.full((1, 1, f_number), 0.1))
        self.rope = RotaryEmbeddingFast2D(self.head_dim)
        self.attn_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, H, W):  # 新增H和W参数，接收真实特征图尺寸
        B, N, C = x.shape

        x = self.attn_norm(x)  # 序列特征归一化

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = qkv

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 传入真实H和W计算位置编码
        q, k = self.rope(q, k, H, W)
        q = q * self.scale

        k_sum = k.sum(dim=2, keepdim=True)  # [B, heads, head_dim, 1]
        kv = torch.einsum('b h n d, b h n e -> b h d e', k, v)  # [B, heads, head_dim, head_dim]

        numerator = torch.einsum('b h n d, b h d e -> b h n e', q, kv)
        denominator = torch.einsum('b h n d, b h d k -> b h n k', q, k_sum) + 1e-4
        out = torch.clamp(numerator / denominator, min=-1e4, max=1e4)

        out = out * self.attn_scale
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        out = out * torch.sigmoid(self.gate)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class LocalGlobalFusion(nn.Module):
    def __init__(self, f_number, num_heads, dropout=0.0, use_group_norm=False):
        super().__init__()
        self.f_number = f_number

        # CNN部分归一化（根据批次大小选择）
        if use_group_norm:
            self.cnn_norm = nn.GroupNorm(num_groups=8, num_channels=f_number)
        else:
            self.cnn_norm = nn.BatchNorm2d(f_number, eps=1e-5, momentum=0.1)

        self.local_conv = nn.Conv2d(f_number, f_number, kernel_size=3, padding=1, groups=f_number, bias=False)
        self.global_attn = BiGatedLinearAttention(f_number, num_heads, dropout)
        self.gate_conv = nn.Conv2d(f_number, f_number, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, C, H, W = x.shape  # 获取真实的特征图高和宽
        x = self.cnn_norm(x)  # CNN部分归一化

        # 局部特征提取
        O_local = self.local_conv(x)

        # 全局特征提取：展平为序列并传入真实H和W
        x_flat = O_local.flatten(2).transpose(1, 2)  # [B, H*W, C]，此时N=H*W
        O_global = self.global_attn(x_flat, H, W)  # 传递真实H和W，解决形状不匹配

        # 全局特征重塑为特征图形状
        O_global = O_global.transpose(1, 2).reshape(B, C, H, W)

        # 自适应门控融合
        G2D = torch.sigmoid(self.gate_conv(O_local))
        out = G2D * O_local + (1 - G2D) * O_global
        # 残差连接
        out = out * self.residual_scale + x
        return out
