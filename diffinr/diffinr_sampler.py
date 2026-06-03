"""
Core DiffINR sampler — strictly implements Algorithm 1 from the paper.

Algorithm overview:
  for t = T down to 1:
    ① Reverse diffusion step (Eq.11):
         x_{t-1} = (1 - 0.5·β_t/T)·x_t - (β_t/T)·s_θ(x_t, t) + √(β_t/T)·ε
    ② Tweedie denoising (Eq.6):
         x_{0|t-1} = (x_{t-1} + std²·s_θ) / √ᾱ(t-1)
    ③ INR-DC (conditional, when t > t* and (t-1) % k == 0):
         Stage 1: prior embedding (L2 lr=1e-3 250it)
         Stage 2: DC refinement  (L1 lr=1e-5 250it)
         x_{t-1} = √ᾱ(t-1)·x̂₀ + √(1-ᾱ(t-1))·ε  (noise remap)

This sampler is completely independent of HFS-SDE's PC sampler.
It only reuses:
  - HFS-SDE's score_model (loaded from checkpoint)
  - HFS-SDE's VPSDE instance (via sde_lib) for marginal_prob
  - HFS-SDE's utility functions from utils.utils (fft2c_2d, etc.)
"""

import torch
import torch.nn as nn


class DiffINRSampler(nn.Module):
    """Core DiffINR sampler (Algorithm 1).

    Args:
        score_model: Pre-trained score network (from HFS-SDE checkpoint).
        sde: VPSDE instance (provides marginal_prob, sde methods).
        config: Configuration object containing sampling parameters.
    """

    def __init__(self, score_model, sde, config):
        super().__init__()
        self.score_model = score_model
        self.sde = sde
        self.config = config

        # Sampling hyper-parameters from config
        self.T = config.sampling.T          # 2000
        self.t_star = config.sampling.t_star  # 1200
        self.k = config.sampling.k           # 50
        self.beta_min = config.model.beta_min  # 0.1
        self.beta_max = config.model.beta_max  # 20.0

        # For score model: handle eval mode
        self.score_model.eval()

        # Will be set by caller
        self.inr_dc_module = None

    # ---- Score function wrapper ----

    def _score_fn(self, x: torch.Tensor, t_cont: torch.Tensor) -> torch.Tensor:
        """Compute score with correct label convention.

        During training, get_score_fn(sde, model, continuous=True) wraps
        the model so that labels = std (from marginal_prob), NOT t_cont.
        This method replicates that wrapper for inference.

        Args:
            x: (B, 2, H, W) current state.
            t_cont: () continuous time scalar in [0, 1).
        Returns:
            (B, 2, H, W) score estimate.
        """
        t_1d = t_cont.unsqueeze(0)  # () → (1,)
        # Get std as labels — matches get_score_fn for continuous=True
        _, std = self.sde.marginal_prob(torch.zeros_like(x[:1]), t_1d)
        labels = std  # (1,) scalar — same for entire batch
        return self.score_model(x, labels)

    # ---- Helper: batch time ----

    @staticmethod
    def _make_t_batch(t_cont: torch.Tensor, n: int) -> torch.Tensor:
        """Expand continuous time scalar to batch.

        Args:
            t_cont: scalar tensor ().
            n: batch size.
        Returns:
            (n,) tensor.
        """
        return t_cont.expand(n)

    # ---- Helper: image format conversions ----

    @staticmethod
    def _complex_to_r2c(x: torch.Tensor) -> torch.Tensor:
        """Convert real/imag stacked to complex.

        Args:
            x: (B, 2, H, W) or (2, H, W).
        Returns:
            complex tensor (B, H, W) or (H, W).
        """
        if x.dim() == 4:
            return torch.complex(x[:, 0], x[:, 1])
        else:
            return torch.complex(x[0], x[1])

    @staticmethod
    def _complex_to_c2r(x: torch.Tensor) -> torch.Tensor:
        """Convert complex to real/imag stacked.

        Args:
            x: complex tensor (B, H, W) or (H, W).
        Returns:
            (B, 2, H, W) or (2, H, W).
        """
        if x.dim() == 3:
            return torch.stack([x.real, x.imag], dim=1)
        else:
            return torch.stack([x.real, x.imag], dim=0)

    # ---- Helpers: alpha_bar ----

    def _get_sqrt_alpha_bar(self, t_cont: torch.Tensor) -> torch.Tensor:
        """Compute √ᾱ(t) from continuous time t.

        Uses the identity: std² = 1 - ᾱ, so √ᾱ = √(1 - std²).
        This avoids the numerically unstable mean/x division.

        Args:
            t_cont: () continuous time scalar.
        Returns:
            () scalar tensor = √ᾱ(t).
        """
        t_1d = t_cont.unsqueeze(0)  # () → (1,)
        # dummy x with batch dim matching t_1d
        dummy = torch.zeros(1, device=t_cont.device)
        _, std = self.sde.marginal_prob(dummy, t_1d)  # both (1,)
        # ᾱ = 1 - std²
        sqrt_alpha_bar = torch.sqrt(1.0 - std**2 + 1e-8)  # (1,)
        return sqrt_alpha_bar.squeeze(0)  # ()

    # ---- Core Algorithm Steps ----

    @torch.no_grad()
    def reverse_step(
        self, x: torch.Tensor, t_cont: torch.Tensor, epsilon_zero: bool = False
    ) -> torch.Tensor:
        """① Reverse diffusion step — paper Eq.(11).

        Reverse VP-SDE (Anderson's theorem, dt_reverse = -dt):
          x_{t-1} = x_t + [-½β(t)·x_t - β(t)·score]·(-Δt) + √β(t)·√Δt·ε
                  = (1 + ½β(t)·Δt)·x_t + β(t)·Δt·score + √β(t)·Δt·ε

        With Δt = 1/T → paper Eq.(11):
          x_{t-1} = (1 + ½·β_t/T)·x_t + (β_t/T)·s_θ(x_t, t) + √(β_t/T)·ε

        where s_θ is the score function from the pretrained model.

        Args:
            x: (B, 2, H, W) current state (real/imag stacked).
            t_cont: () continuous time tensor.
            epsilon_zero: If True, set ε=0 (for the last step t=1).
        Returns:
            (B, 2, H, W) next state x_{t-1}.
        """
        # Paper Algorithm 1 line 3: ε ~ N(0,I) if t > 1 else 0
        epsilon = torch.zeros_like(x) if epsilon_zero else torch.randn_like(x)

        # Continuous β(t) = β_min + t·(β_max - β_min)
        beta = self.beta_min + t_cont * (self.beta_max - self.beta_min)  # scalar

        # Discretized step: Δt = 1/T
        beta_disc = beta / self.T

        # Predict score: s_θ(x_t, t)  (using correct label convention)
        score = self._score_fn(x, t_cont)  # (B, 2, H, W)

        # Paper Eq.(11): reverse step with + signs (from reverse-time SDE derivation)
        x_next = (
            (1 + 0.5 * beta_disc) * x
            + beta_disc * score
            + torch.sqrt(beta_disc) * epsilon
        )

        return x_next

    @torch.no_grad()
    def tweedie(self, x: torch.Tensor, t_cont: torch.Tensor) -> torch.Tensor:
        """② Tweedie denoising — corrected Eq.(6).

        x_{0|t} = (x_t + std² · s_θ) / √ᾱ(t)

        Derivation:
          score = ∇_x log p_t(x) = -ε / std
          x₀ = (x - std·ε) / √ᾱ  = (x + std²·score) / √ᾱ

        Args:
            x: (B, 2, H, W) noisy state x_{t-1}.
            t_cont: () continuous time scalar tensor.
        Returns:
            (B, 2, H, W) denoised x_{0|t-1}.
        """
        score = self._score_fn(x, t_cont)

        t_1d = t_cont.unsqueeze(0)  # () → (1,)
        _, std = self.sde.marginal_prob(x, t_1d)
        # std shape: (1,) — broadcastable to (B,2,H,W) via [:,None,None,None]

        # √ᾱ(t) = √(1 - std²)
        sqrt_alpha_bar = torch.sqrt(1.0 - std**2 + 1e-8)  # (1,)

        # Corrected Tweedie: x₀ = (x + std² · score) / √ᾱ
        x0_pred = (x + std[:, None, None, None]**2 * score) / sqrt_alpha_bar[:, None, None, None]
        return x0_pred

    @torch.no_grad()
    def noise_remap(
        self, x_clean: torch.Tensor, t_cont: torch.Tensor
    ) -> torch.Tensor:
        """③ Noise remapping — Algorithm 1 line 10-11.

        x_{t-1} = √ᾱ(t) · x̂₀ + √(1-ᾱ(t)) · ε

        Args:
            x_clean: (B, 2, H, W) INR output (clean image).
            t_cont: () continuous time scalar tensor.
        Returns:
            (B, 2, H, W) noisy x_{t-1} for continued diffusion.
        """
        t_1d = t_cont.unsqueeze(0)  # () → (1,)
        mean, std = self.sde.marginal_prob(x_clean, t_1d)
        epsilon = torch.randn_like(x_clean)
        return mean + std[:, None, None, None] * epsilon

    # ---- Main Sampling Loop ----

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        forward_op,
        img_shape: tuple,
        inr_dc_module=None,
    ) -> torch.Tensor:
        """Full DiffINR sampling loop — Algorithm 1.

        Args:
            y: Measured k-space (complex) shape depends on coils.
            forward_op: MRIForwardOperator instance.
            img_shape: (B, 2, H, W) shape for sampling.
            inr_dc_module: INRDCModule instance (set here or via attr).

        Returns:
            (B, 2, H, W) reconstructed image.
        """
        if inr_dc_module is not None:
            self.inr_dc_module = inr_dc_module

        device = next(self.score_model.parameters()).device
        dtype = next(self.score_model.parameters()).dtype

        # Initialize: x_T ~ N(0, I)
        x = torch.randn(img_shape, device=device, dtype=dtype)

        # Logging interval
        log_interval = max(1, self.T // 20)  # ~ every 5%
        _inr_phase_logged = False

        for step in range(self.T, 0, -1):
            # Continuous time: (t-1)/T so that t=1 → 0, t=T → (T-1)/T
            t_cont = torch.tensor((step - 1) / self.T, device=device)

            # ① Reverse diffusion step (Eq.11)
            # Paper Algorithm 1 line 3: ε ~ N(0,I) if t > 1 else 0
            x = self.reverse_step(x, t_cont, epsilon_zero=(step == 1))

            # ② Tweedie denoising (Eq.6)
            x0_pred = self.tweedie(x, t_cont)

            # Progress log (for non-INR steps)
            if step % log_interval == 0 and not (step > self.t_star and (step - 1) % self.k == 0):
                print(f"  [t={step}/{self.T}] reverse step + tweedie")

            # ③ INR-DC (conditional)
            # Condition: t <= t* AND (t-1) % k == 0
            # Paper: INR-DC只在图像已经较为干净的后半程激活（t ≤ t*）
            if step <= self.t_star and (step - 1) % self.k == 0:
                if not _inr_phase_logged:
                    _inr_phase_logged = True
                    print(f"  [t={step}/{self.T}] Entered INR-DC phase (t ≤ t*={self.t_star})")
                if step == self.t_star or (step - 1) % (self.k * 4) == 0:
                    print(f"  [t={step}/{self.T}] INR-DC start (prior + DC refinement)")
                # Convert from (B,2,H,W) stacked to (H,W) complex for INR
                # Currently support batch_size=1
                x0_complex = self._complex_to_r2c(x0_pred[0])  # (H, W)

                # Stage 1: Prior Embedding
                self.inr_dc_module.prior_embedding(
                    x0_complex, lr=1e-3, n_iter=250
                )

                # Stage 2: DC Refinement
                x_dc = self.inr_dc_module.dc_refinement(
                    y, forward_op, lr=1e-5, n_iter=250
                )  # (H, W) complex

                # Convert back to (B,2,H,W)
                x_dc_stacked = self._complex_to_c2r(x_dc).unsqueeze(0)  # (1,2,H,W)

                # Noise remapping (Algorithm 1 line 10-11)
                x = self.noise_remap(x_dc_stacked, t_cont)

                if (step - 1) % (self.k * 4) == 0 or step == self.T:
                    print(f"  [t={step}/{self.T}] INR-DC done")

        # Final step log
        print(f"  Sampling completed ({self.T} steps)")

        return x
