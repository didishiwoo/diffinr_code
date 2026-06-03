"""
DiffINR sampler — posterior sampling strictly following Algorithm 1.

Supports two reverse-step modes:
  - "sde":   Continuous-time SDE (Eq.11), Euler-Maruyama discretization.
             x_{t-1} = (1 + 0.5·β_t/T)·x_t + (β_t/T)·s_θ + √(β_t/T)·ε
  - "ddpm":  Discrete DDPM update matching ablation.py.
             Maps step → discrete index, converts score → ε, uses DDPM formula.

Algorithm 1 structure (paper §3, Algorithm 1):
  for t = T down to 1:
    ① Reverse diffusion step → x_{t-1}
    ② Tweedie denoising      → x_{0|t-1}
    ③ INR-DC (conditional):
         prior_embedding → dc_refinement → noise_remap → x_{t-1}

Key design:
  - HFS-SDE pretrained model outputs ∇log p (Score), NOT ε.
    Tweedie uses PLUS sign:  x₀ = (x + std²·score) / √ᾱ
  - ᾱ(t) computed via VPSDE.marginal_prob() for consistency with pretrained weights.
"""

import torch
import torch.nn as nn


class DiffINRSampler(nn.Module):
    def __init__(self, score_model, sde, config):
        super().__init__()
        self.score_model = score_model
        self.sde = sde
        self.config = config
        self.T = config.sampling.T
        self.t_star = config.sampling.t_star
        self.k = config.sampling.k
        self.mode = config.sampling.mode  # "sde" or "ddpm"
        self.beta_min = config.model.beta_min
        self.beta_max = config.model.beta_max
        self.score_model.eval()
        self.inr_dc_module = None

        # Discrete DDPM parameters (for mode="ddpm")
        if self.mode == "ddpm":
            self.N = config.model.num_scales  # 1000 (training steps)
            self.register_buffer("_betas", sde.discrete_betas.clone())
            self.register_buffer("_alphas", sde.alphas.clone())
            self.register_buffer("_alphas_cumprod", sde.alphas_cumprod.clone())

    # ---- helpers ----

    def _tensor(self, val, device):
        return torch.tensor(val, device=device, dtype=torch.float32)

    def _beta_t(self, t_cont):
        return self.beta_min + t_cont * (self.beta_max - self.beta_min)

    def _alpha_bar(self, t_cont, device):
        """Compute √ᾱ(t) and √(1-ᾱ(t)) via VPSDE.marginal_prob()."""
        t_1d = self._tensor([t_cont], device)
        _, std = self.sde.marginal_prob(torch.zeros(1, 1, device=device), t_1d)
        std = std.float()
        sqrt_ab = torch.sqrt(1.0 - std**2 + 1e-8)
        return sqrt_ab, std

    def _score(self, x, step):
        t_cont = (step - 1) / self.T
        _, std = self._alpha_bar(t_cont, x.device)
        return self.score_model(x.float(), std)

    @staticmethod
    def r2c(x):
        return torch.complex(x[:, 0], x[:, 1]) if x.dim() == 4 else torch.complex(x[0], x[1])

    @staticmethod
    def c2r(x):
        return torch.stack([x.real, x.imag], dim=1) if x.dim() == 3 else torch.stack([x.real, x.imag], dim=0)

    # ---- reverse step — dual mode ----

    @torch.no_grad()
    def reverse_step(self, x, step):
        if self.mode == "sde":
            return self._reverse_step_sde(x, step)
        else:
            return self._reverse_step_ddpm(x, step)

    def _reverse_step_sde(self, x, step):
        t_cont = (step - 1) / self.T
        beta_val = self._beta_t(t_cont)
        beta_disc = beta_val / self.T
        beta_tensor = self._tensor(beta_disc, x.device)
        score = self._score(x, step)
        epsilon = torch.randn_like(x) if step > 1 else torch.zeros_like(x)
        return (1.0 + 0.5 * beta_disc) * x + beta_disc * score + torch.sqrt(beta_tensor) * epsilon

    def _reverse_step_ddpm(self, x, step):
        t_cont = (step - 1) / self.T
        idx = min(int(t_cont * self.N), self.N - 1)
        score = self._score(x, step)
        _, std = self._alpha_bar(t_cont, x.device)
        alpha = self._alphas[idx]
        beta = self._betas[idx]
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_beta = torch.sqrt(beta)
        sqrt_1m_alpha_bar = torch.sqrt(1.0 - self._alphas_cumprod[idx])
        eps_pred = -std[:, None, None, None] * score
        epsilon = torch.randn_like(x) if step > 1 else torch.zeros_like(x)
        return (x - (beta / sqrt_1m_alpha_bar) * eps_pred) / sqrt_alpha + sqrt_beta * epsilon

    # ---- Tweedie & noise_remap ----

    @torch.no_grad()
    def tweedie(self, x, step):
        t_cont = (step - 1) / self.T
        sqrt_ab, std = self._alpha_bar(t_cont, x.device)
        score = self._score(x, step)
        return (x + std[:, None, None, None]**2 * score) / sqrt_ab[:, None, None, None]

    @torch.no_grad()
    def noise_remap(self, x_clean, step):
        if step <= 1:
            return x_clean
        t_cont = (step - 1) / self.T
        sqrt_ab, std = self._alpha_bar(t_cont, x_clean.device)
        return sqrt_ab[:, None, None, None] * x_clean \
               + std[:, None, None, None] * torch.randn_like(x_clean)

    # ---- main loop ----

    def sample(self, y, forward_op, img_shape, inr_dc_module=None):
        self.inr_dc_module = inr_dc_module

        x = torch.randn(img_shape, device=next(self.score_model.parameters()).device)
        total = self.T
        log_intv = max(1, total // 20)
        _inr_losses = []

        # Log initial
        print(f"  [init t={total}/{total}] noise |x|_max={x.abs().max().item():.3f}")

        for step in range(total, 0, -1):
            do_inr = (self.inr_dc_module is not None
                      and step <= self.t_star
                      and (step - 1) % self.k == 0)
            log_this = (step % log_intv == 0 or step == total or do_inr)

            if do_inr:
                print(f"  [t={step}/{total}] INR-DC start (prior + DC refinement)")

            with torch.no_grad():
                # ① Reverse diffusion: x_t → x_{t-1}
                x_prev = self.reverse_step(x, step)

                # ② Tweedie denoising: x_{t-1} → x_{0|t-1}
                tweedie_step = step - 1 if step > 1 else 1
                x0 = self.tweedie(x_prev, tweedie_step)

            # ③ INR-DC (only when image is clean enough: step ≤ t*)
            if do_inr:
                with torch.enable_grad():
                    self.inr_dc_module.zero_grad()
                    x0_c = self.r2c(x0[0])

                    # Stage 1: Prior embedding
                    self.inr_dc_module.prior_embedding(x0_c, lr=1e-3, n_iter=250)

                    # Stage 2: DC refinement
                    x_inr, final_loss = self.inr_dc_module.dc_refinement(
                        y, forward_op, lr=1e-5, n_iter=250
                    )
                    _inr_losses.append(final_loss)

                with torch.no_grad():
                    x0_refined = self.c2r(x_inr).unsqueeze(0)
                    x = self.noise_remap(x0_refined, tweedie_step)

                n_inr = len(_inr_losses)
                print(f"  [t={step}/{total}] INR-DC done  ({n_inr})  loss={final_loss:.4e}")
            else:
                x = x_prev
                if log_this:
                    x_abs = x.abs()
                    print(f"  [t={step}/{total}] reverse step + tweedie  "
                          f"|x|_max={x_abs.max().item():.3f}")

        # Summary
        print(f"  ── Done ({total} steps) ──")
        if _inr_losses:
            print(f"  INR-DC triggered {len(_inr_losses)} times  "
                  f"final_loss={_inr_losses[-1]:.4e}  "
                  f"best_loss={min(_inr_losses):.4e}")
        print(f"  |x|_max={x.abs().max().item():.3f}  |x|_mean={x.abs().mean().item():.4f}  "
              f"x_real∈[{x.real.min().item():.3f},{x.real.max().item():.3f}]")
        return x
