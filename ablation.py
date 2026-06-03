"""
Ablation: Pure DDPM with standard discrete DDPM update.
Uses VPSDE's discrete betas/alphas to avoid SDE discretization drift.
"""
import torch, sys, os
os.chdir('/root/diffinr_code/diffinr_code')
sys.path.insert(0, '.'); sys.path.insert(0, 'hfs_sde')

import sde_lib
from models import ddpm
from config import get_config
import scipy.io as scio

config = get_config()
device = torch.device('cuda:0')

model = ddpm.DDPM(config).to(device)
ckpt = torch.load('checkpoint/checkpoint_190.pth', map_location='cpu')
model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model'].items()}, strict=False)
model.eval()
print('Model loaded')

sde = sde_lib.VPSDE(config)

# Use the pre-computed discrete parameters from VPSDE (N=1000)
N = config.model.num_scales  # 1000
betas = sde.discrete_betas.to(device)      # (1000,)  β_0,...,β_{N-1}
alphas = sde.alphas.to(device)             # (1000,)  α_i = 1 - β_i
alphas_cumprod = sde.alphas_cumprod.to(device)  # (1000,)  ᾱ_i

# Sampling with T steps, mapping to discrete grid
T = 500
x = torch.randn(1, 2, 320, 320, device=device)
print(f'Starting T={T}')

for step in range(T, 0, -1):
    if step % 200 == 0 or step == T:
        print(f'  step {step}/{T}', end='')

    # Map continuous time to discrete index [0, N-1]
    t_cont = (step - 1) / T  # [0, 1)
    idx = min(int(t_cont * N), N - 1)  # discrete index

    # Get labels = std for the score model (continuous training convention)
    t_1d = torch.tensor([t_cont], device=device)
    _, labels = sde.marginal_prob(torch.zeros(1, device=device), t_1d)

    with torch.no_grad():
        score = model(x, labels)  # score = ∇log p

    # Standard DDPM update:
    # ε_pred = -std · score  (convert score to noise prediction)
    # x₀ = (x - std · ε_pred) / √ᾱ  (= (x + std²·score) / √ᾱ)
    # But we don't explicitly compute x₀; use the single-step formula:
    # x_{i-1} = (x - β_i/√(1-ᾱ_i) · ε_pred) / √α_i  +  √β_i · z
    
    alpha_i = alphas[idx]
    beta_i = betas[idx]
    sqrt_alpha_i = torch.sqrt(alpha_i)
    sqrt_beta_i = torch.sqrt(beta_i)
    sqrt_1m_alpha_bar = torch.sqrt(1.0 - alphas_cumprod[idx])
    
    eps_pred = -labels * score  # convert score to noise
    
    eps = torch.zeros_like(x) if step == 1 else torch.randn_like(x)
    
    # DDPM update: x_{i-1} = (x - β_i/√(1-ᾱ_i) · ε_pred) / √α_i + √β_i · z
    x = (x - (beta_i / sqrt_1m_alpha_bar) * eps_pred) / sqrt_alpha_i + sqrt_beta_i * eps

    if step % 200 == 0 or step == T:
        print(f'  |x|={x.abs().max().item():.2f}  idx={idx}')

recon_abs = torch.sqrt(x[:,0]**2 + x[:,1]**2).squeeze().cpu()
print(f'\nResult: [{recon_abs.min():.4f}, {recon_abs.max():.4f}] mean={recon_abs.mean():.4f}')
scio.savemat('results/pure_ddpm_v3.mat', {'recon': recon_abs.numpy()})
print('Saved.')
