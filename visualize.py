"""
Visualize DiffINR reconstruction — auto-called by main.py after each run.
Saves timestamped comparison PNG, never overwrites previous results.
"""
import os
import numpy as np
import scipy.io as scio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


def visualize_run(results_dir: str, run_id: str):
    """Generate comparison figure for a given run_id.

    Args:
        results_dir: Directory containing the .mat files.
        run_id:  e.g. '20260603_161210'
    """
    # Find files matching *{run_id}.mat
    import glob
    gt_pattern = os.path.join(results_dir, f'*gt*{run_id}.mat')
    recon_pattern = os.path.join(results_dir, f'*recon*{run_id}.mat')
    gt_files = sorted(glob.glob(gt_pattern))
    recon_files = sorted(glob.glob(recon_pattern))

    if not gt_files or not recon_files:
        print(f'  [viz] No files found for run_id={run_id}')
        print(f'  [viz]   Searched: {gt_pattern}')
        print(f'  [viz]   Searched: {recon_pattern}')
        return

    # Use the first pair (sample 0)
    gt_path, recon_path = gt_files[0], recon_files[0]

    gt_data = scio.loadmat(gt_path)
    recon_data = scio.loadmat(recon_path)

    gt = gt_data[[k for k in gt_data.keys() if not k.startswith('__')][0]]
    recon = recon_data[[k for k in recon_data.keys() if not k.startswith('__')][0]]

    gt_abs = np.abs(gt)
    recon_abs = np.abs(recon)

    # Normalize to [0,1]
    gt_norm = gt_abs / gt_abs.max()
    recon_norm = recon_abs / recon_abs.max() if recon_abs.max() > 0 else recon_abs

    # PSNR (normalized amplitude)
    mse = np.mean((gt_norm - recon_norm) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))

    error = np.abs(gt_norm - recon_norm)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Images
    im1 = axes[0, 0].imshow(gt_norm, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    im2 = axes[0, 1].imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Recon (PSNR: {psnr:.2f} dB)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    im3 = axes[0, 2].imshow(error, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Error Map')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # Row 2: Profiles & histogram
    h, w = gt_norm.shape
    axes[1, 0].plot(gt_norm[h // 2, :], 'b-', label='GT', alpha=0.8)
    axes[1, 0].plot(recon_norm[h // 2, :], 'r--', label='Recon', alpha=0.8)
    axes[1, 0].set_title(f'Center Row Profile (y={h // 2})')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('intensity')

    axes[1, 1].plot(gt_norm[:, w // 2], 'b-', label='GT', alpha=0.8)
    axes[1, 1].plot(recon_norm[:, w // 2], 'r--', label='Recon', alpha=0.8)
    axes[1, 1].set_title(f'Center Col Profile (x={w // 2})')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('y')
    axes[1, 1].set_ylabel('intensity')

    axes[1, 2].hist(gt_norm.ravel(), bins=100, alpha=0.5, label='GT', color='blue')
    axes[1, 2].hist(recon_norm.ravel(), bins=100, alpha=0.5, label='Recon', color='red')
    axes[1, 2].set_title('Pixel Intensity Histogram')
    axes[1, 2].legend()

    plt.suptitle(f'DiffINR Reconstruction — run {run_id}', fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(results_dir, f'comparison_{run_id}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [viz] Saved: {out_path}  (PSNR: {psnr:.2f} dB)')


def save_mat(results_dir: str, run_id: str, tensor, name: str = 'recon'):
    """Save a torch/complex tensor as .mat with run_id prefix."""
    arr = tensor.cpu().detach().numpy() if hasattr(tensor, 'detach') else tensor
    path = os.path.join(results_dir, f'{name}_{run_id}.mat')
    scio.savemat(path, {name: np.squeeze(arr)})
    print(f'  [save] {path}')


if __name__ == '__main__':
    # CLI usage: python3 visualize.py <run_id>
    import sys
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    run_id = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y%m%d_%H%M%S')
    visualize_run(results_dir, run_id)
