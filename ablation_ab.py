"""
Ablation: Run WITHOUT INR-DC, then compare with the WITH-INR-DC result (20260603_184332).
Same DDPM mode, same data, for fair ablation.
"""
import os, sys
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "hfs_sde"))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import torch
import numpy as np
import h5py
import sde_lib
import losses
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
from models import ddpm
from utils.utils import get_mask, restore_checkpoint, r2c, ifft2c_2d, fft2c_2d, IFFT2c, FFT2c, normalize_complex
from config import get_config
from diffinr.diffinr_sampler import DiffINRSampler
from diffinr.forward_operator import MRIForwardOperator
import scipy.io as scio
from datetime import datetime


def load_and_normalize(h5_path, slice_idx):
    import utils.utils as uu
    with h5py.File(h5_path, 'r') as f:
        ksp = f['kspace'][slice_idx].copy()
    ksp_np = np.expand_dims(ksp, (0, 1))
    ksp_np = IFFT2c(ksp_np)
    ksp_np = uu.crop(ksp_np, 320, 320)
    ksp_np = FFT2c(ksp_np)
    ksp_t = torch.from_numpy(ksp_np)
    img = ifft2c_2d(ksp_t)
    img_norm = normalize_complex(img)
    ksp_norm = fft2c_2d(img_norm)
    return ksp_norm.squeeze()


def main():
    config = get_config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = str(device)
    logging.info(f"Device: {device}")

    # Load model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    ckpt_path = os.path.join(_PROJECT_DIR, "checkpoint", f"checkpoint_{config.sampling.ckpt}.pth")
    state = restore_checkpoint(ckpt_path, state, device=device)
    score_model.eval()
    logging.info("Checkpoint loaded")

    sde = sde_lib.VPSDE(config)

    # Load data (same slice as previous with-INR run)
    data_dir = os.path.join(_PROJECT_DIR, "data", "knee_singlecoil_test", "singlecoil_test")
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    h5_path = os.path.join(data_dir, h5_files[0])

    kspace_full = load_and_normalize(h5_path, slice_idx=18).to(device)
    atb_mask = get_mask(config, "sample").to(device)
    mask_real = atb_mask[0, 0].real if atb_mask[0, 0].is_complex() else atb_mask[0, 0]
    y = kspace_full * mask_real
    forward_op = MRIForwardOperator(mask_real)

    # Run WITHOUT INR-DC
    torch.manual_seed(42)  # fixed seed for reproducibility
    sampler = DiffINRSampler(score_model, sde, config).to(device)
    logging.info("Running WITHOUT INR-DC (DDPM mode)...")
    recon = sampler.sample(
        y=y, forward_op=forward_op,
        img_shape=(1, 2, 320, 320),
        inr_dc_module=None,
    )
    recon_abs = r2c(recon).abs().squeeze().cpu()
    logging.info(f"No-INR: [{recon_abs.min():.4f}, {recon_abs.max():.4f}] "
                 f"mean={recon_abs.mean():.4f}, std={recon_abs.std():.4f}")

    # Save
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(_PROJECT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    scio.savemat(os.path.join(out_dir, f"recon_no_inr_{run_id}.mat"),
                 {"recon": recon_abs.numpy()})

    # Load WITH-INR result (run 184332) for comparison
    try:
        d_inr = scio.loadmat(os.path.join(out_dir, "recon_20260603_184332.mat"))
        k_inr = [k for k in d_inr.keys() if not k.startswith('__')][0]
        r_inr = np.squeeze(np.abs(d_inr[k_inr]))
        logging.info(f"Loaded with-INR from recon_20260603_184332.mat: [{r_inr.min():.4f}, {r_inr.max():.4f}]")

        # Normalize both to [0,1] for comparison
        no_inr_n = recon_abs.numpy() / recon_abs.numpy().max()
        inr_n = r_inr / r_inr.max()
        diff = np.abs(no_inr_n - inr_n)
        psnr = 20 * np.log10(1.0 / np.sqrt(np.mean(diff**2) + 1e-10))

        logging.info(f"Comparison: |No-INR - INR| max={diff.max():.4f} mean={diff.mean():.4f} PSNR={psnr:.2f}dB")

        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        im = axes[0,0].imshow(no_inr_n, cmap='gray', vmin=0, vmax=1); axes[0,0].set_title('No INR-DC'); axes[0,0].axis('off'); plt.colorbar(im, ax=axes[0,0], fraction=0.046)
        im = axes[0,1].imshow(inr_n, cmap='gray', vmin=0, vmax=1); axes[0,1].set_title('With INR-DC'); axes[0,1].axis('off'); plt.colorbar(im, ax=axes[0,1], fraction=0.046)
        im = axes[0,2].imshow(diff, cmap='hot', vmin=0, vmax=1); axes[0,2].set_title(f'|Diff| (max={diff.max():.3f})'); axes[0,2].axis('off'); plt.colorbar(im, ax=axes[0,2], fraction=0.046)

        h = no_inr_n.shape[0]
        axes[1,0].plot(no_inr_n[h//2,:], 'b-', label='No INR', alpha=0.8)
        axes[1,0].plot(inr_n[h//2,:], 'r--', label='With INR', alpha=0.8)
        axes[1,0].set_title('Center Row'); axes[1,0].legend()

        axes[1,1].hist(no_inr_n.ravel(), bins=100, alpha=0.5, label='No INR', color='blue')
        axes[1,1].hist(inr_n.ravel(), bins=100, alpha=0.5, label='With INR', color='red')
        axes[1,1].set_title('Histogram'); axes[1,1].legend()

        axes[1,2].text(0.3, 0.5,
            f'DDPM T={config.sampling.T}\nt*={config.sampling.t_star}\nk={config.sampling.k}\n\nDiff max={diff.max():.3f}\nDiff mean={diff.mean():.4f}\nPSNR={psnr:.2f}dB',
            fontsize=14, transform=axes[1,2].transAxes)
        axes[1,2].axis('off')

        plt.suptitle('DiffINR Ablation: INR-DC Effect', fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'ablation_{run_id}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved: {out_path}")
    except Exception as e:
        logging.warning(f"Could not load with-INR result for comparison: {e}")
        logging.info("Saved no-INR result only; run with-INR separately to compare.")

    logging.info("Done!")


if __name__ == "__main__":
    main()
