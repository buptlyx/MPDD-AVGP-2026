from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 对时间维度做注意力池化
# 把[N, T, H]变成[N, H]
# N = batch_size, T = 时间步数量, H = hidden_dim
# 自动判断那些时间步是重要的
class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # 给每一个时间步计算一个分数
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 得到每一个时间步的注意力分数
        scores = self.score(x).squeeze(-1)
        # 如果提供了有效时间步标记，则对无效时间步进行掩码处理
        if seq_mask is not None:
            seq_mask = seq_mask.bool()
            # 掩码必须是二维的
            if seq_mask.ndim != 2:
                raise ValueError(f"Expected seq_mask shape [N, T], got {tuple(seq_mask.shape)}")
            # 如果所有时间步都被mask，则用一个全True的mask代替
            all_masked = ~seq_mask.any(dim=1)
            # 如果存在全mask的样本
            if all_masked.any():
                seq_mask = seq_mask.clone()
                seq_mask[all_masked] = True
            # 无效帧用极小值代替
            scores = scores.masked_fill(~seq_mask, -1e9)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(attn * x, dim=1)


class HybridTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.3,
        pre_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even, got {hidden_dim}")
        # 预降维处理
        if pre_dim is not None and input_dim > pre_dim:
            self.pre_proj = nn.Linear(input_dim, pre_dim)
            conv_in = pre_dim
        else:
            self.pre_proj = None
            conv_in = input_dim
        # 两层卷积提取局部特征
        self.conv1 = nn.Conv1d(conv_in, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        # 双向LSTM捕捉前后文时序依赖
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pool = TemporalAttentionPool(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        seq_mask = torch.any(torch.abs(x) > 0, dim=-1)

        if self.pre_proj is not None:
            x = self.pre_proj(x)

        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = self.dropout(x)
        x = F.gelu(self.conv2(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        x = self.pool(x, seq_mask=seq_mask)
        return self.norm(x)
