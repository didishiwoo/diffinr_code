"""
Tiny MLP for implicit neural representation.

Architecture (matching DiffINR paper):
  - Input: hash encoding features (dim = num_levels * feat_dim)
  - Hidden: 2 layers × 64 neurons, ReLU activation
  - Output: 1 scalar (pixel intensity for one component)

The paper uses two separate MLPs for real and imaginary components.
Each MLP is identical in architecture but has independent weights.
"""

import torch
import torch.nn as nn


class INR_MLP(nn.Module):
    """Tiny MLP with 2 hidden layers of 64 neurons and ReLU.

    Args:
        in_dim: Input feature dimension (from hash encoding).
        hidden_dim: Hidden layer width (default 64, per paper).
        num_layers: Number of hidden layers (default 2, per paper).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (..., in_dim) hash encoded features.
        Returns:
            (..., 1) scalar pixel intensity.
        """
        return self.net(x)
