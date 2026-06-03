"""
INR-based Data Consistency Module (DiffINR paper §3.3).

Architecture:
  - Shared HashEncoding for coordinate embedding
  - Two separate INR_MLP instances for real and imaginary components
  - Two-stage learning:
      Stage 1: Prior embedding — L2 loss, lr=1e-3, 250 iterations
      Stage 2: DC refinement  — L1 loss, lr=1e-5, 250 iterations

References:
  - DiffINR (Medical Image Analysis, 2025)
  - Instant-NGP / Müller et al. SIGGRAPH 2022 (hash encoding)
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
        in_dim = self.hash_encoding.out_dim  # num_levels * feat_dim
        self.mlp_real = INR_MLP(in_dim, mlp_hidden_dim, mlp_num_layers)
        self.mlp_imag = INR_MLP(in_dim, mlp_hidden_dim, mlp_num_layers)

        # Pre-compute normalized coordinate grid for given image size
        self.register_buffer(
            "_coords",
            self._build_coord_grid(img_size),  # (H*W, 2)
        )

    @staticmethod
    def _build_coord_grid(img_size: int) -> torch.Tensor:
        """Build normalized coordinate grid [0, 1).

        Returns:
            (H*W, 2) tensor of (x, y) coordinates.
        """
        # Use linspace to [0, 1 - 1/img_size) to keep exactly img_size points
        # while ensuring all values are strictly < 1.0
        xs = torch.linspace(0, 1 - 1.0 / img_size, img_size, dtype=torch.float32)
        ys = torch.linspace(0, 1 - 1.0 / img_size, img_size, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1).reshape(-1, 2)

    def forward(self, coords: torch.Tensor) -> tuple:
        """Evaluate INR at given coordinates.

        Args:
            coords: (N, 2) normalized coordinates in [0, 1).
        Returns:
            (real, imag): tuple of (N, 1) tensors.
        """
        enc = self.hash_encoding(coords)          # (N, L*F)
        real = self.mlp_real(enc)                 # (N, 1)
        imag = self.mlp_imag(enc)                 # (N, 1)
        return real, imag

    def to_complex(self, coords: torch.Tensor = None) -> torch.Tensor:
        """Evaluate INR and return as complex image.

        Args:
            coords: (N, 2) or None (uses pre-computed grid).
        Returns:
            (H, W) complex tensor.
        """
        if coords is None:
            coords = self._coords
        real, imag = self.forward(coords)
        H = W = self.img_size
        real_img = real.reshape(H, W)
        imag_img = imag.reshape(H, W)
        return torch.complex(real_img, imag_img)

    def _image_from_mlp(self) -> torch.Tensor:
        """Render full image from current MLP parameters.

        Returns:
            (H, W) complex image tensor.
        """
        return self.to_complex()

    # ---- Stage 1: Prior Embedding ----

    def prior_embedding(
        self,
        x0_pred: torch.Tensor,
        lr: float = 1e-3,
        n_iter: int = 100,
    ) -> None:
        """Stage 1: Embed diffusion prior into INR (L2 loss).

        Fits INR to the Tweedie-denoised image x0_pred.

        Args:
            x0_pred: (H, W) complex tensor — Tweedie denoised image.
            lr: Learning rate (paper: 1e-3).
            n_iter: Number of optimization iterations (paper: 250).
        """
        with torch.enable_grad():
            coords = self._coords
            target_real = x0_pred.real.reshape(-1, 1).detach()  # (N, 1)
            target_imag = x0_pred.imag.reshape(-1, 1).detach()

            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            loss_fn = nn.MSELoss()  # L2 loss

            for _ in range(n_iter):
                pred_real, pred_imag = self.forward(coords)
                loss = loss_fn(pred_real, target_real) + loss_fn(pred_imag, target_imag)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # ---- Stage 2: Data Consistency Refinement ----

    def dc_refinement(
        self,
        y: torch.Tensor,
        forward_op,
        lr: float = 1e-5,
        n_iter: int = 100,
    ) -> torch.Tensor:
        """Stage 2: Data consistency refinement (L1 loss).

        Optimizes INR so that A(INR(d)) matches measured k-space y.

        Args:
            y: Measured k-space (complex tensor).
            forward_op: MRIForwardOperator instance (A = MFS).
            lr: Learning rate (paper: 1e-5).
            n_iter: Number of iterations (paper: 250).

        Returns:
            (H, W) complex image after DC refinement.
        """
        with torch.enable_grad():
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

            for _ in range(n_iter):
                # Render current INR image
                inr_img = self._image_from_mlp()  # (H, W) complex

                # Forward operator: A(INR(d))
                kspace_pred = forward_op(inr_img)  # complex k-space

                # L1 loss in k-space
                loss = torch.mean(torch.abs(kspace_pred - y))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self._image_from_mlp()
