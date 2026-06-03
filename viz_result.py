"""Visualize DiffINR result vs zero-filled for a specific run."""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'hfs_sde')
import torch, numpy as np, scipy.io as scio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run_id = '20260603_200457_fake_multi'
results_dir = 'results'

# Load recon and zf
recon = scio.loadmat(os.path.join(results_dir, f'recon_{run_id}.mat'))['recon']
zf = scio.loadmat(os.path.join(results_dir, f'zf_{run_id}.mat'))['zf']

# Normalize to [0,1]
recon_norm = recon / recon.max()
zf_norm = zf / zf.max()

# Error map
error = np.abs(zf_norm - recon_norm)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

im1 = axes[0,0].imshow(zf_norm, cmap='gray', vmin=0, vmax=1)
axes[0,0].set_title('Zero-Filled')
axes[0,0].axis('off'); plt.colorbar(im1, ax=axes[0,0], fraction=0.046)

im2 = axes[0,1].imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
axes[0,1].set_title('DiffINR (fake_multi)')
axes[0,1].axis('off'); plt.colorbar(im2, ax=axes[0,1], fraction=0.046)

im3 = axes[0,2].imshow(error, cmap='hot', vmin=0, vmax=1)
axes[0,2].set_title('Error (|ZF - DiffINR|)')
axes[0,2].axis('off'); plt.colorbar(im3, ax=axes[0,2], fraction=0.046)

# Profiles
h, w = zf_norm.shape
axes[1,0].plot(zf_norm[h//2,:], 'b-', label='ZF', alpha=0.8)
axes[1,0].plot(recon_norm[h//2,:], 'r--', label='DiffINR', alpha=0.8)
axes[1,0].set_title(f'Center Row Profile (y={h//2})')
axes[1,0].legend(); axes[1,0].set_xlabel('x')

axes[1,1].plot(zf_norm[:,w//2], 'b-', label='ZF', alpha=0.8)
axes[1,1].plot(recon_norm[:,w//2], 'r--', label='DiffINR', alpha=0.8)
axes[1,1].set_title(f'Center Col Profile (x={w//2})')
axes[1,1].legend(); axes[1,1].set_xlabel('y')

axes[1,2].hist(zf_norm.ravel(), bins=100, alpha=0.5, label='ZF', color='blue')
axes[1,2].hist(recon_norm.ravel(), bins=100, alpha=0.5, label='DiffINR', color='red')
axes[1,2].set_title('Histogram')
axes[1,2].legend()

plt.suptitle(f'DiffINR vs Zero-Filled — fake_multi mode\nrun_id={run_id}', fontsize=14)
plt.tight_layout()
out = os.path.join(results_dir, f'comparison_{run_id}.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
