"""
Stage 9 - STGCN (Yu et al., IJCAI 2018), lifted out of notebooks/train_stgcn.ipynb.

The architecture is the notebook's cell 7 unchanged. One thing differs, and it
is the reason this file exists: `c_in` is a parameter instead of a hardcoded 1,
so the same model serves every rung of the ablation ladder (c_in 1, 2, 4, 5 or
11). The graph convolution was always general in c_in - Yu et al. define it as
    y_j = sum_{i=1..C_i} Theta_{i,j}(L) x_i,   Theta in R^{K x C_i x C_o}
and only used C_i = 1 in their experiments. Nothing here is a new architecture.

Where this deviates from the paper, on purpose, and why:

  * ST-Conv block channels are 64/64/64, not the paper's 64/16/64. The notebook
    made that choice for the stage-8 reproduction and matching it keeps rung 0
    comparable to the number we already have.
  * The model emits all HORIZON steps from one forward pass (`fc` maps the last
    feature vector to HORIZON), where the paper predicts a single step. This is
    the convention DCRNN and Graph WaveNet use on PEMS-BAY, which is what our
    baseline numbers come from.

The paper's own numbers are on BJER4 and PeMSD7, not PEMS-BAY; the PEMS-BAY
STGCN baseline everyone quotes is from Wu et al. (2019), Table 2.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# graph support

def scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    """Rescale the normalised Laplacian to [-1, 1] so Chebyshev is stable.

    The adjacency is SYMMETRISED first, and that is not cosmetic. adj_mx_bay is
    directed - 1,818 of its 2,694 edges have no reverse - because it encodes
    driving distance along the road network. A Chebyshev graph convolution is
    defined on an undirected graph: it needs a symmetric Laplacian with a real
    spectrum. Feeding the raw directed matrix to `eigsh`, which assumes symmetry
    and does not check, returned lambda_max = 1.2670 against a true largest
    eigenvalue of 1.0013 - the rescaling was off by 27%. `max(A, A.T)` gives a
    valid symmetric Laplacian with lambda_max = 1.2408, so the numbers move by
    about 2% and the operator becomes well defined.

    Keep the ORIGINAL directed matrix for anything that needs travel direction -
    the upstream-propagation analysis in src/eval/metrics.py reads
    data/processed/adj_mx.npy, not this.
    """
    n = adj.shape[0]
    adj = np.maximum(adj, adj.T)           # Chebyshev needs an undirected graph
    d = adj.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    lap = np.eye(n) - np.diag(d_inv_sqrt) @ adj @ np.diag(d_inv_sqrt)
    assert np.allclose(lap, lap.T, atol=1e-6), "Laplacian is not symmetric"
    lambda_max = eigsh(lap, k=1, which="LM", return_eigenvectors=False)[0]
    return (2.0 / lambda_max) * lap - np.eye(n)


def cheb_polynomials(scaled_lap: np.ndarray, K: int) -> list[torch.Tensor]:
    """T_0..T_{K-1} of the rescaled Laplacian."""
    n = scaled_lap.shape[0]
    polys = [np.eye(n, dtype=np.float32)]
    if K > 1:
        polys.append(scaled_lap.astype(np.float32))
    for _ in range(2, K):
        polys.append((2.0 * scaled_lap @ polys[-1] - polys[-2]).astype(np.float32))
    return [torch.from_numpy(p) for p in polys]


# ---------------------------------------------------------------------------
# layers

class ChebConv(nn.Module):
    """Graph convolution, Chebyshev approximation of order K."""

    def __init__(self, K: int, c_in: int, c_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(K, c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, cheb_polys):
        B, T, N, _ = x.shape
        x_flat = x.reshape(B * T, N, -1)
        out = torch.zeros(B * T, N, self.weight.shape[2],
                          device=x.device, dtype=x.dtype)
        for k, poly in enumerate(cheb_polys):
            out = out + torch.matmul(torch.matmul(poly.to(x.device), x_flat),
                                     self.weight[k])
        return (out + self.bias).reshape(B, T, N, -1)


class TemporalConv(nn.Module):
    """Gated 1-D convolution along time. Shortens the sequence by Kt-1."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(kernel_size, 1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        p, q = self.conv(x).chunk(2, dim=1)
        return (p * torch.sigmoid(q)).permute(0, 2, 3, 1)


class STConvBlock(nn.Module):
    """temporal -> graph -> temporal, with a residual connection."""

    def __init__(self, Ks, Kt, c_in, c_mid, c_out, dropout=0.0):
        super().__init__()
        self.temp1 = TemporalConv(c_in, c_mid, Kt)
        self.graph = ChebConv(Ks, c_mid, c_mid)
        self.temp2 = TemporalConv(c_mid, c_out, Kt)
        self.norm = nn.LayerNorm(c_out)
        self.dropout = nn.Dropout(dropout)
        self.residual_conv = (nn.Conv2d(c_in, c_out, kernel_size=1)
                              if c_in != c_out else None)
        self.Kt = Kt

    def forward(self, x, cheb_polys):
        residual = x
        out = self.dropout(self.norm(self.temp2(self.graph(self.temp1(x),
                                                           cheb_polys))))
        trim = self.Kt - 1
        if trim > 0:
            residual = residual[:, trim:-trim, :, :]
        if self.residual_conv is not None:
            residual = self.residual_conv(
                residual.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return torch.relu(out + residual)


# ---------------------------------------------------------------------------

class STGCN(nn.Module):
    """X [B, INPUT_WINDOW, N, c_in] -> Y_hat [B, HORIZON, N].

    `c_in` comes from `loader.build_loaders(...)["c_in"]`; do not hardcode it.
    """

    def __init__(self, cheb_polys, c_in, input_window, horizon, Ks, Kt,
                 blocks, dropout):
        super().__init__()
        blocks = [list(b) for b in blocks]
        blocks[0][0] = c_in            # the whole point of this file
        self.n_cheb = len(cheb_polys)
        for k, poly in enumerate(cheb_polys):
            self.register_buffer(f"_cheb_{k}", poly)
        self.st_blocks = nn.ModuleList(
            STConvBlock(Ks, Kt, ci, cm, co, dropout) for ci, cm, co in blocks)
        t_remaining = input_window - len(blocks) * 2 * (Kt - 1)
        if t_remaining <= 0:
            raise ValueError(
                f"input_window {input_window} is too short for {len(blocks)} "
                f"blocks at Kt={Kt}: each block consumes 2*(Kt-1)="
                f"{2 * (Kt - 1)} steps, leaving {t_remaining}.")
        c_last = blocks[-1][-1]
        self.output_conv = nn.Conv2d(c_last, c_last, kernel_size=(t_remaining, 1))
        self.fc = nn.Linear(c_last, horizon)

    @property
    def _cheb_polys(self):
        return [getattr(self, f"_cheb_{k}") for k in range(self.n_cheb)]

    def forward(self, x):
        polys = self._cheb_polys
        for block in self.st_blocks:
            x = block(x, polys)
        x = x.permute(0, 3, 1, 2)
        x = self.output_conv(x).squeeze(2).permute(0, 2, 1)
        return self.fc(x).permute(0, 2, 1)          # [B, horizon, N]


def build_model(adj: np.ndarray, c_in: int, cfg: dict,
                input_window: int, horizon: int, device="cpu") -> STGCN:
    """Assemble the model from an adjacency matrix and `cfg["model"]["stgcn"]`."""
    m = cfg["model"]["stgcn"]
    polys = cheb_polynomials(scaled_laplacian(adj), m["Ks"])
    model = STGCN(polys, c_in, input_window, horizon,
                  m["Ks"], m["Kt"], m["blocks"], m["dropout"])
    return model.to(device)
