"""
Multi-resolution hash grid encoding.

Reference:
  - Müller et al. "Instant Neural Graphics Primitives with a
    Multiresolution Hash Encoding." SIGGRAPH 2022.
  - Shi et al. 2022 (as cited in DiffINR paper).

Implements a pure-PyTorch version of the multiresolution hash encoding.
For P0 we use L=4 levels; the full paper uses L=16.
"""

import torch
import torch.nn as nn
import numpy as np


class HashEncoding(nn.Module):
    """Multi-resolution hash grid encoding.

    Args:
        num_levels: Number of resolution levels (L in paper).
        feat_dim: Feature dimension per level (F in paper).
        log2_hashmap_size: Log2 of hash table size (T in paper).
        base_resolution: Base grid resolution.
        finest_resolution: Finest grid resolution.
    """

    def __init__(
        self,
        num_levels: int = 4,
        feat_dim: int = 2,
        log2_hashmap_size: int = 10,
        base_resolution: int = 16,
        finest_resolution: int = 128,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.feat_dim = feat_dim
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.out_dim = num_levels * feat_dim

        # Per-level resolution: grows exponentially from base to finest
        growth_factor = (finest_resolution / base_resolution) ** (1.0 / (num_levels - 1)) \
            if num_levels > 1 else 1.0
        self.register_buffer(
            "resolutions",
            torch.tensor(
                [base_resolution * (growth_factor ** i) for i in range(num_levels)],
                dtype=torch.float32,
            ),
        )

        # Hash tables: one per level, shape (hashmap_size, feat_dim)
        self.hash_tables = nn.ParameterList([
            nn.Parameter(
                torch.randn(self.hashmap_size, feat_dim) * 0.01
            )
            for _ in range(num_levels)
        ])

        # Primes for hash function (standard from Instant-NGP)
        self.register_buffer(
            "_primes",
            torch.tensor([1, 2654435761, 805459861], dtype=torch.int64),
        )

    def _hash(self, indices: torch.Tensor) -> torch.Tensor:
        """Hash function using primes.

        Args:
            indices: (..., 3) integer indices (ix, iy, level_offset)
        Returns:
            (...,) hashed indices in [0, hashmap_size)
        """
        indices = indices.to(torch.int64)
        # h = (ix * p1) ^ (iy * p2) ^ (level * p3)
        h = (
            indices[..., 0] * self._primes[0]
            ^ indices[..., 1] * self._primes[1]
            ^ indices[..., 2] * self._primes[2]
        )
        return h % self.hashmap_size

    def _lookup(self, level: int, indices: torch.Tensor) -> torch.Tensor:
        """Look up features from a single level's hash table.

        Args:
            level: Resolution level index.
            indices: (..., 2) grid coordinates (ix, iy).
        Returns:
            (..., feat_dim) feature vectors.
        """
        # Add level offset for hashing
        level_offset = torch.full_like(indices[..., :1], level)
        idx_with_level = torch.cat([indices, level_offset], dim=-1)  # (..., 3)
        hash_idx = self._hash(idx_with_level)  # (...,)
        return self.hash_tables[level][hash_idx]  # (..., feat_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates.

        Args:
            coords: (..., 2) normalized coordinates in [0, 1).
        Returns:
            (..., num_levels * feat_dim) concatenated features.
        """
        orig_shape = coords.shape[:-1]
        flat_coords = coords.reshape(-1, 2)  # (N, 2)
        N = flat_coords.shape[0]

        level_features = []
        for level in range(self.num_levels):
            res = self.resolutions[level].item()

            # Scale coordinates to grid resolution
            scaled = flat_coords * res  # (N, 2)

            # Integer grid cell coordinates (lower-left corner)
            icoord = scaled.to(torch.int64)  # (N, 2)
            # Keep within [0, res-2] to ensure we have the upper-right corner
            icoord = torch.clamp(icoord, 0, int(res) - 2)

            # Fractional part for interpolation
            frac = scaled - icoord.float()  # (N, 2)

            # Four corners of the cell
            corners = torch.stack([
                icoord,                                       # (0, 0)
                icoord + torch.tensor([1, 0], device=coords.device),  # (1, 0)
                icoord + torch.tensor([0, 1], device=coords.device),  # (0, 1)
                icoord + torch.tensor([1, 1], device=coords.device),  # (1, 1)
            ], dim=0)  # (4, N, 2)

            # Look up features for all 4 corners
            feats_00 = self._lookup(level, corners[0])  # (N, F)
            feats_10 = self._lookup(level, corners[1])
            feats_01 = self._lookup(level, corners[2])
            feats_11 = self._lookup(level, corners[3])

            # Bilinear interpolation
            w_x = frac[:, 0:1]   # (N, 1)
            w_y = frac[:, 1:2]
            feats = (
                feats_00 * (1 - w_x) * (1 - w_y)
                + feats_10 * w_x * (1 - w_y)
                + feats_01 * (1 - w_x) * w_y
                + feats_11 * w_x * w_y
            )  # (N, F)

            level_features.append(feats)

        # Concatenate all levels: (N, L * F)
        out = torch.cat(level_features, dim=-1)
        return out.reshape(*orig_shape, self.out_dim)
