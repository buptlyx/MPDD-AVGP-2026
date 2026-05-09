from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_temporal_encoder import HybridTemporalEncoder
# test
# 模态编码器
# 用处：把每一个模态的时间序列编码成一个向量
# 输入：[N, T, H]，输出:[N, H]
# 多模态信息需要对输入进行降维处理
class ModalityEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.5,
        pre_dim: int | None = None,
    ) -> None:
        super().__init__()
        if pre_dim is not None and input_dim > pre_dim:
            self.pre_proj = nn.Linear(input_dim, pre_dim)
            lstm_in = pre_dim
        else:
            self.pre_proj = None
            lstm_in = input_dim
        self.proj = nn.Linear(lstm_in, hidden_dim)
        # 定义双向LSTM
        self.lstm = nn.LSTM(
            # input_size
            hidden_dim,
            # hidden_size
            hidden_dim // 2,
            # layer
            num_layers=1,
            # batch是否放在第一维
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_proj is not None:
            x = F.relu(self.pre_proj(x))
        x = F.relu(self.proj(x))
        x = self.dropout(x)
        x, _ = self.lstm(x)
        # 平均池化
        return self.norm(x.mean(dim=1))

# 人格特质编码器
# 用处：把人格特质的1024维向量编码成一个64维向量
class PersonalityEncoder(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Baseline模型
class TorchcatBaseline(nn.Module):
    # 预定义每一个子赛道的模态组合
    SUBTRACKS = {
        "A-V-P": ["audio", "video", "personality"],
        "A-V-G-P": ["audio", "video", "gait", "personality"],
        "G-P": ["gait", "personality"],
    }
    # 支持两种编码器：普通的双向LSTM和混合注意力编码器
    ENCODER_TYPES = {"bilstm_mean", "hybrid_attn"}

    def __init__(
        self,
        # 赛道选择
        subtrack: str = "A-V-G-P",
        # 分类任务数
        num_classes: int = 3,
        # 是否是回归任务（是否使用回归头）
        is_regression: bool = False,
        use_regression_head: bool = False,
        # 每个模态的输入维度和编码器的隐藏层维度
        audio_dim: int = 64,
        video_dim: int = 1000,
        gait_dim: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        # 编码器的类型选择
        encoder_type: str = "bilstm_mean",
    ) -> None:
        super().__init__()
        if subtrack not in self.SUBTRACKS:
            raise ValueError(f"Unknown subtrack: {subtrack}")
        if encoder_type not in self.ENCODER_TYPES:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.subtrack = subtrack
        self.modalities = self.SUBTRACKS[subtrack]
        self.encoder_type = encoder_type
        self.is_regression = is_regression
        self.use_regression_head = use_regression_head

        if "audio" in self.modalities:
            pre_audio = 128 if audio_dim > 128 else None
            self.audio_enc = (
                HybridTemporalEncoder(audio_dim, hidden_dim, dropout, pre_dim=pre_audio)
                if encoder_type == "hybrid_attn"
                else ModalityEncoder(audio_dim, hidden_dim, dropout, pre_dim=pre_audio)
            )
        if "video" in self.modalities:
            pre_video = 128 if video_dim > 128 else None
            self.video_enc = (
                HybridTemporalEncoder(video_dim, hidden_dim, dropout, pre_dim=pre_video)
                if encoder_type == "hybrid_attn"
                else ModalityEncoder(video_dim, hidden_dim, dropout, pre_dim=pre_video)
            )
        if "gait" in self.modalities:
            # 步态维度较小，使用结构简单的编码器
            self.gait_enc = ModalityEncoder(gait_dim, hidden_dim, dropout)
        if "personality" in self.modalities:
            # personality不是时序数据，直接使用全连接神经网络
            self.pers_enc = PersonalityEncoder(1024, hidden_dim, dropout)
        # 融合层：把所有模态的编码结果拼接起来进行分类
        fused_dim = hidden_dim * len(self.modalities)
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1 if is_regression else num_classes),
        )
        # 可选回归头
        if use_regression_head:
            self.regressor = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    @staticmethod
    def _masked_average_sequences(x: torch.Tensor, pair_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pair_mask is None:
            return x.mean(dim=1)
        weights = pair_mask.unsqueeze(-1).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    @staticmethod
    def _masked_average_features(x: torch.Tensor, pair_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pair_mask is None:
            return x.mean(dim=1)
        weights = pair_mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def _encode_pairwise_sequences(
        self,
        x: torch.Tensor,
        encoder: nn.Module,
        pair_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, pair_count, seq_len, feat_dim = x.shape
        encoded = encoder(x.reshape(batch_size * pair_count, seq_len, feat_dim))
        encoded = encoded.reshape(batch_size, pair_count, -1)
        return self._masked_average_features(encoded, pair_mask)

    def forward(
        self,
        audio: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        gait: torch.Tensor | None = None,
        personality: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = []

        if "audio" in self.modalities:
            if self.encoder_type == "hybrid_attn":
                features.append(self._encode_pairwise_sequences(audio, self.audio_enc, pair_mask))
            else:
                features.append(self.audio_enc(self._masked_average_sequences(audio, pair_mask)))

        if "video" in self.modalities:
            if self.encoder_type == "hybrid_attn":
                features.append(self._encode_pairwise_sequences(video, self.video_enc, pair_mask))
            else:
                features.append(self.video_enc(self._masked_average_sequences(video, pair_mask)))

        if "gait" in self.modalities:
            features.append(self.gait_enc(gait))

        if "personality" in self.modalities:
            features.append(self.pers_enc(personality))

        fused = torch.cat(features, dim=-1)
        logits = self.classifier(fused)
        if self.use_regression_head:
            return logits, self.regressor(fused).squeeze(-1)
        return logits
