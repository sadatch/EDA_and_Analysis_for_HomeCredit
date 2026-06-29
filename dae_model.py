"""
Denoising Autoencoder (DAE) with Swap Noise.
ikiri_DS (2位解法) 構成を参考にしたPyTorch実装。

- 各カラムを独立に、一定確率で「同じカラムの別行の値」に入れ替えるswap noise
  （ガウスノイズと違いカテゴリ変数・スケールの異なる数値変数にも自然に効く）
- Encoder 3層（デフォルト各1024、元実装は4096）→ bottleneck → Decoderで再構成
- 学習後、Encoder各層の出力をconcatして特徴量として抜き出す
"""
import numpy as np
import torch
import torch.nn as nn


def swap_noise(x: torch.Tensor, swap_rate: float, generator=None) -> torch.Tensor:
    """
    各列について、確率swap_rateでバッチ内の別の行の値に入れ替える。
    x: (batch, n_features)
    """
    batch_size, n_features = x.shape
    x_noisy = x.clone()
    mask = torch.rand(batch_size, n_features, generator=generator, device=x.device) < swap_rate
    # 各列ごとにバッチ内の行をシャッフルした版を用意し、maskが立っている箇所だけ採用
    shuffled_idx = torch.stack([
        torch.randperm(batch_size, generator=generator, device=x.device)
        for _ in range(n_features)
    ], dim=1)  # (batch, n_features)
    shuffled_x = torch.gather(x, 0, shuffled_idx)
    x_noisy = torch.where(mask, shuffled_x, x_noisy)
    return x_noisy


class StackedDAE(nn.Module):
    """
    3層スタックEncoder + 対称Decoder。
    各Encoder層の出力（活性化後）をconcatしたものを最終特徴量として使う。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 1024, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        enc_dims = [input_dim] + [hidden_dim] * n_layers
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(enc_dims[i], enc_dims[i + 1]),
                nn.BatchNorm1d(enc_dims[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            for i in range(n_layers)
        ])

        dec_dims = list(reversed(enc_dims))
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dec_dims[i], dec_dims[i + 1]),
                nn.BatchNorm1d(dec_dims[i + 1]) if i + 1 < len(dec_dims) - 1 else nn.Identity(),
                nn.ReLU(inplace=True) if i + 1 < len(dec_dims) - 1 else nn.Identity(),
            )
            for i in range(len(dec_dims) - 1)
        ])

    def encode(self, x: torch.Tensor):
        """各層の出力をリストで返す（最後がbottleneck）。"""
        hs = []
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
            hs.append(h)
        return hs

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder_layers:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor):
        hs = self.encode(x)
        recon = self.decode(hs[-1])
        return recon, hs

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """学習済みモデルでconcat特徴量を抽出（推論時はノイズなし）。"""
        self.eval()
        hs = self.encode(x)
        return torch.cat(hs, dim=1)
