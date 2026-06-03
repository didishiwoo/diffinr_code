"""
MRI forward operator A = MFS for DiffINR.

  A(x) = M · F · S(x)

  where:
    M = under-sampling mask (diagonal)
    F = Fourier transform (FFT) — centered (matches HFS-SDE's fft2c_2d)
    S = coil sensitivity maps (multi-coil) or identity (single-coil)

Paper §2.1 Eq.(2): A = M F S
Paper Stage 2: optimizes ||A(INR(d)) - y||₁ — only forward is needed.
"""

import torch


def _fft2c_centered(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT, matching HFS-SDE's fft2c_2d convention.

    Args:
        x: (..., H, W) complex tensor.
    Returns:
        (..., H, W) complex k-space with centered DC.
    """
    orig_ndim = x.ndim
    # Ensure 4D: (1, 1, H, W)
    while x.ndim < 4:
        x = x.unsqueeze(0)
    # ifftshift on spatial dims, then fft2
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    # Restore original dimensions
    while x.ndim > orig_ndim:
        x = x.squeeze(0)
    return x


class MRIForwardOperator:
    """MRI forward acquisition operator A = MFS.

    Uses centered FFT (ifftshift → fft2 → fftshift) to match the
    HFS-SDE convention used in checkpoint training and data loading.

    Args:
        mask: Under-sampling mask (H, W), 1=sampled, 0=not sampled.
        sens_maps: Coil sensitivity maps (C, H, W) complex, or None for single-channel.
    """

    def __init__(self, mask: torch.Tensor, sens_maps: torch.Tensor = None):
        self.mask = mask  # (H, W)
        self.sens_maps = sens_maps  # (C, H, W) or None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply A = MFS.

        Correct order:
          S: coil sensitivity maps applied in image domain (x * sens_maps)
          F: Centered Fourier transform
          M: under-sampling mask

        Args:
            x: (H, W) complex image.
        Returns:
            y: (C, H, W) complex k-space (multi-coil) or (H, W) (single-coil).
        """
        if self.sens_maps is not None:
            # S: Apply coil sensitivity maps in image domain
            x = x * self.sens_maps  # (C, H, W)
        # F: Centered Fourier transform (matches HFS-SDE convention)
        kspace = _fft2c_centered(x)  # (C, H, W) or (H, W)
        # M: Apply under-sampling mask
        kspace = kspace * self.mask

        return kspace

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
