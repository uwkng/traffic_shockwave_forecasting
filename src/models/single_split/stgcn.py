"""STGCN — Spatio-Temporal Graph Convolutional Network (Yu et al., IJCAI 2018).

Extracted from notebooks/train_stgcn.ipynb. The architecture is unchanged;
the only difference is that c_in is a parameter so the same model serves
every ablation rung.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    d = adj.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    lap = np.eye(n) - np.diag(d_inv_sqrt) @ adj @ np.diag(d_inv_sqrt)
    lambda_max = eigsh(lap, k=1, which="LM", return_eigenvectors=False)[0]
    return (2.0 / lambda_max) * lap - np.eye(n)


def cheb_polynomials(scaled_lap: np.ndarray, K: int) -> list[torch.Tensor]:
    n = scaled_lap.shape[0]
    polys = [np.eye(n, dtype=np.float32)]
    if K > 1:
        polys.append(scaled_lap.astype(np.float32))
    for _ in range(2, K):
        polys.append((2.0 * scaled_lap @ polys[-1] - polys[-2]).astype(np.float32))
    return [torch.from_numpy(p) for p in polys]


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class ChebConv(nn.Module):
    def __init__(self, K: int, c_in: int, c_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(K, c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, cheb_polys: list[torch.Tensor]) -> torch.Tensor:
        B, T, N, _ = x.shape
        x_flat = x.reshape(B * T, N, -1)
        out = torch.zeros(B * T, N, self.weight.shape[2], device=x.device, dtype=x.dtype)
        for k, poly in enumerate(cheb_polys):
            transformed = torch.matmul(poly.to(x.device), x_flat)
            out = out + torch.matmul(transformed, self.weight[k])
        return (out + self.bias).reshape(B, T, N, -1)


class TemporalConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(kernel_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        out = self.conv(x)
        p, q = out.chunk(2, dim=1)
        return (p * torch.sigmoid(q)).permute(0, 2, 3, 1)


class STConvBlock(nn.Module):
    def __init__(self, Ks: int, Kt: int, c_in: int, c_mid: int, c_out: int,
                 dropout: float = 0.0):
        super().__init__()
        self.temp1 = TemporalConv(c_in, c_mid, Kt)
        self.graph = ChebConv(Ks, c_mid, c_mid)
        self.temp2 = TemporalConv(c_mid, c_out, Kt)
        self.norm = nn.LayerNorm(c_out)
        self.dropout = nn.Dropout(dropout)
        self.residual_conv = (nn.Conv2d(c_in, c_out, kernel_size=1)
                              if c_in != c_out else None)
        self.Kt = Kt

    def forward(self, x: torch.Tensor, cheb_polys: list[torch.Tensor]) -> torch.Tensor:
        residual = x
        out = self.temp1(x)
        out = self.graph(out, cheb_polys)
        out = self.temp2(out)
        out = self.norm(out)
        out = self.dropout(out)
        trim = self.Kt - 1
        if trim > 0:
            residual = residual[:, trim:-trim, :, :]
        if self.residual_conv is not None:
            residual = self.residual_conv(
                residual.permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)
        return torch.relu(out + residual)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class STGCN(nn.Module):
    """STGCN with configurable input channels.

    Parameters
    ----------
    cheb_polys : list[Tensor]
        Chebyshev polynomial matrices from ``cheb_polynomials()``.
    c_in : int
        Number of input channels — set from ``loaders["c_in"]``.
    input_window, horizon : int
        From ``contract.INPUT_WINDOW`` / ``contract.HORIZON``.
    Ks, Kt : int
        Chebyshev order and temporal kernel size.
    blocks : list of [c_in, c_mid, c_out]
        Channel sizes per ST-Conv block. The first block's ``c_in`` is
        overridden by the ``c_in`` argument so the config doesn't need to
        know the rung.
    dropout : float
    """

    def __init__(self, cheb_polys: list[torch.Tensor], c_in: int,
                 input_window: int, horizon: int, Ks: int, Kt: int,
                 blocks: list[list[int]], dropout: float):
        super().__init__()
        self.n_cheb = len(cheb_polys)
        for k, poly in enumerate(cheb_polys):
            self.register_buffer(f"_cheb_{k}", poly)

        blocks = [list(b) for b in blocks]
        blocks[0][0] = c_in

        self.st_blocks = nn.ModuleList()
        for channels in blocks:
            self.st_blocks.append(
                STConvBlock(Ks, Kt, channels[0], channels[1], channels[2], dropout)
            )

        t_remaining = input_window - len(blocks) * 2 * (Kt - 1)
        assert t_remaining > 0, (
            f"input_window={input_window} too short for {len(blocks)} blocks "
            f"with Kt={Kt}: need > {len(blocks) * 2 * (Kt - 1)} steps"
        )
        c_last = blocks[-1][-1]
        self.output_conv = nn.Conv2d(c_last, c_last, kernel_size=(t_remaining, 1))
        self.fc = nn.Linear(c_last, horizon)

    @property
    def _cheb_polys(self) -> list[torch.Tensor]:
        return [getattr(self, f"_cheb_{k}") for k in range(self.n_cheb)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        polys = self._cheb_polys
        for block in self.st_blocks:
            x = block(x, polys)
        x = x.permute(0, 3, 1, 2)          # [B, C, T', N]
        x = self.output_conv(x).squeeze(2)  # [B, C, N]
        x = x.permute(0, 2, 1)             # [B, N, C]
        x = self.fc(x)                      # [B, N, horizon]
        return x.permute(0, 2, 1)           # [B, horizon, N]
