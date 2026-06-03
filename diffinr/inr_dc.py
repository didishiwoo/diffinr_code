"""
INR-based Data Consistency Module (DiffINR paper §3.3).

Architecture:
  - Shared HashEncoding for coordinate embedding
  - Two separate INR_MLP instances for real and imaginary components
  - Two-stage learning:
      Stage 1: Prior embedding — L2 loss, lr=1e-3, n_iter (default 250)
      Stage 2: DC refinement  — L1 loss (ACS-calibrated), lr=1e-5, n_iter (default 250)

Key detail:
  - ACS scale calibration in dc_refinement prevents energy drift when the
    diffusion prior and physical measurements have different scales.
"""

import torch
import torch.nn as nn
from .hash_encoding import HashEncoding
from .inr_mlp import INR_MLP


class INRDCModule(nn.Module):
    """INR-based Data Consistency Module.

    Args:
        img_size: Image spatial size (H, W). Default 320.
        hash_num_levels: Hash encoding levels (L). Default 4 (P0).
        hash_feat_dim: Feature dim per level (F). Default 2.
        hash_log2_size: Log2 of hash table size (T). Default 10.
        mlp_hidden_dim: MLP hidden layer width. Default 64.
        mlp_num_layers: MLP hidden layer count. Default 2.
    """

    def __init__(
        self,
        img_size: int = 320,
        hash_num_levels: int = 4,
        hash_feat_dim: int = 2,
        hash_log2_size: int = 10,
        mlp_hidden_dim: int = 64,
        mlp_num_layers: int = 2,
    ):
        super().__init__()
        self.img_size = img_size
        self.hash_encoding = HashEncoding(
            num_levels=hash_num_levels,
            feat_dim=hash_feat_dim,
            log2_hashmap_size=hash_log2_size,
        )
        in_dim = self.hash_encoding.out_dim
        self.mlp_real = INR_MLP(in_dim, mlp_hidden_dim, mlp_num_layers)
        self.mlp_imag = INR_MLP(in_dim, mlp_hidden_dim, mlp_num_layers)

        self.register_buffer(
            "_coords",
            self._build_coord_grid(img_size),
        )

    @staticmethod
    def _build_coord_grid(img_size: int) -> torch.Tensor:
        xs = torch.linspace(0, 1 - 1.0 / img_size, img_size, dtype=torch.float32)
        ys = torch.linspace(0, 1 - 1.0 / img_size, img_size, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1).reshape(-1, 2)

    def forward(self, coords: torch.Tensor) -> tuple:
        enc = self.hash_encoding(coords)
        real = self.mlp_real(enc)
        imag = self.mlp_imag(enc)
        return real, imag

    def to_complex(self, coords: torch.Tensor = None) -> torch.Tensor:
        if coords is None:
            coords = self._coords
        real, imag = self.forward(coords)
        H = W = self.img_size
        return torch.complex(real.reshape(H, W), imag.reshape(H, W))

    def _image_from_mlp(self) -> torch.Tensor:
        return self.to_complex()

    # ---- Stage 1: Prior Embedding ----

    def prior_embedding(
        self,
        x0_pred: torch.Tensor,
        lr: float = 1e-3,
        n_iter: int = 250,
    ) -> "INRDCModule":
        """Stage 1: Embed diffusion prior into INR (L2 loss).

        Args:
            x0_pred: (H, W) complex tensor — Tweedie denoised image.
            lr: Learning rate (paper: 1e-3).
            n_iter: Number of optimization iterations (paper: 250).

        Returns:
            self, for method chaining.
        """
        with torch.enable_grad():
            coords = self._coords
            target_real = x0_pred.real.reshape(-1, 1).detach()
            target_imag = x0_pred.imag.reshape(-1, 1).detach()

            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            for _ in range(n_iter):
                pred_real, pred_imag = self.forward(coords)
                loss = loss_fn(pred_real, target_real) + loss_fn(pred_imag, target_imag)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    # ---- Stage 2: Data Consistency Refinement ----

    def dc_refinement(
        self,
        y: torch.Tensor,
        forward_op,
        lr: float = 1e-5,
        n_iter: int = 250,
    ) -> tuple:
        """Stage 2: Data consistency refinement (L1 loss in k-space).

        Optimizes INR so that A(INR(d)) matches measured k-space y.

        Args:
            y: Measured k-space (complex tensor).
            forward_op: MRIForwardOperator (A = MFS).
            lr: Learning rate (paper: 1e-5).
            n_iter: Number of iterations (paper: 250).

        Returns:
            (image, final_loss): (H, W) complex image and the final L1 loss value.
        """
        with torch.enable_grad():
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            final_loss = None

            for _ in range(n_iter):
                inr_img = self._image_from_mlp()
                kspace_pred = forward_op(inr_img)

                # L1 loss in k-space
                loss = torch.mean(torch.abs(kspace_pred - y))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                final_loss = loss.item()

        return self._image_from_mlp(), final_loss
