"""
MRI forward operator A = MFS for DiffINR.

  A(x) = M · F · S(x)

  where:
    M = under-sampling mask (diagonal)
    F = Fourier transform (FFT) — centered (matches HFS-SDE's fft2c_2d)
    S = coil sensitivity maps (multi-coil) or identity (single-coil)

Paper §2.1 Eq.(2): A = M F S
"""

import torch


def _fft2c_centered(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT, matching HFS-SDE's fft2c_2d convention."""
    orig_ndim = x.ndim
    while x.ndim < 4:
        x = x.unsqueeze(0)
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    while x.ndim > orig_ndim:
        x = x.squeeze(0)
    return x


def _ifft2c_centered(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D IFFT (inverse of _fft2c_centered)."""
    orig_ndim = x.ndim
    while x.ndim < 4:
        x = x.unsqueeze(0)
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    while x.ndim > orig_ndim:
        x = x.squeeze(0)
    return x


class MRIForwardOperator:
    """MRI forward acquisition operator A = MFS.

    Args:
        mask: Under-sampling mask (H, W), 1=sampled, 0=not sampled.
        sens_maps: Coil sensitivity maps (C, H, W) complex, or None for single-channel.
        acs_width: Number of center k-space lines treated as ACS (default 24).
    """

    def __init__(self, mask: torch.Tensor, sens_maps: torch.Tensor = None,
                 acs_width: int = 24):
        self.mask = mask  # (H, W)
        self.sens_maps = sens_maps  # (C, H, W) or None
        self.acs_width = acs_width

    # ---- ACS mask ----

    def get_acs_mask(self) -> torch.Tensor:
        """Binary mask for ACS (auto-calibration) center k-space lines.

        The ACS region is a vertical band of width `acs_width` at center,
        matching the HFS-SDE uniform mask convention.
        """
        H, W = self.mask.shape
        acs = torch.zeros_like(self.mask, dtype=torch.float32)
        center = W // 2
        half = self.acs_width // 2
        acs[:, center - half:center + half] = 1.0
        return acs

    # ---- forward ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply A = MFS.

        Args:
            x: (H, W) complex image.
        Returns:
            y: (C, H, W) or (H, W) complex k-space.
        """
        if self.sens_maps is not None:
            x = x * self.sens_maps
        kspace = _fft2c_centered(x)
        kspace = kspace * self.mask
        return kspace

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    # ---- adjoint ----

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply A^H = S^H · F^{-1} · M.

        Args:
            y: (C, H, W) or (H, W) complex k-space.
        Returns:
            x: (H, W) complex image.
        """
        # M: apply mask (already applied to y, but ensure consistency)
        y_masked = y * self.mask
        # F^{-1}: IFFT
        img = _ifft2c_centered(y_masked)
        # S^H: sum over coils with conjugate sensitivity weights
        if self.sens_maps is not None:
            img = torch.sum(img * self.sens_maps.conj(), dim=0)
        return img
