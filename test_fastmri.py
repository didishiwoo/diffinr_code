"""
DiffINR sampling on real fastMRI knee data (singlecoil_test).

Supports two modes:
  - singlecoil:  forward operator without sens maps (A = MF)
  - fake_multi:  replicate single-coil across 15 coils with unit sens maps,
                 enabling the multi-coil code path for pipeline validation.

Usage:
    cd /root/diffinr_code/diffinr_code
    python test_fastmri.py                          # single-coil (default)
    python test_fastmri.py --mode fake_multi        # fake multi-coil
"""

import os, sys, argparse
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
from utils.utils import (
    get_mask,
    restore_checkpoint,
    IFFT2c, FFT2c,
)
from config import get_config
from diffinr.diffinr_sampler import DiffINRSampler
from diffinr.inr_dc import INRDCModule
from diffinr.forward_operator import MRIForwardOperator
import scipy.io as scio
from datetime import datetime


def load_and_normalize(h5_path, slice_idx, config):
    """Load a slice with EXACT HFS-SDE "std" normalization (training config).

    Pipeline (matches FastMRIKneeDataSet.__getitem__ with normalize_type="std"):
      raw kspace (640, 368) → expand → IFFT2c → center crop 320×320 → FFT2c
      → kspace /= (normalize_coeff * std(kspace))

    Returns:
        kspace_norm: (320, 320) complex — normalized k-space in HFS-SDE convention.
    """
    import utils.utils as uu

    with h5py.File(h5_path, 'r') as f:
        ksp = f['kspace'][slice_idx].copy()  # (640, 368) complex64

    # IFFT2c → crop 320×320 → FFT2c (numpy pipeline, same as dataset)
    ksp_np = np.expand_dims(ksp, (0, 1))    # (1, 1, 640, 368)
    ksp_np = IFFT2c(ksp_np)                 # image domain
    ksp_np = uu.crop(ksp_np, 320, 320)      # (1, 1, 320, 320)
    ksp_np = FFT2c(ksp_np)                  # back to k-space
    ksp_t = torch.from_numpy(ksp_np)        # (1, 1, 320, 320) complex

    # "std" normalization: kspace /= (normalize_coeff * std(kspace))
    ksp_std = ksp_t.abs().std()
    if ksp_std > 1e-8:
        ksp_t = ksp_t / (config.data.normalize_coeff * ksp_std)

    return ksp_t.squeeze()                  # (320, 320) complex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='singlecoil',
                        choices=['singlecoil', 'fake_multi'],
                        help="'singlecoil' (A=MF) or 'fake_multi' (A=MFS, 15 coils, unit sens)")
    parser.add_argument('--ncoils', type=int, default=15,
                        help='Number of coils for fake_multi mode')
    args = parser.parse_args()

    config = get_config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = str(device)
    img_sz = config.data.image_size  # 320
    logging.info(f"Device: {device}")
    logging.info(f"Mode: {args.mode}" + (f"  ncoils={args.ncoils}" if args.mode == 'fake_multi' else ""))

    # --- Load score model ---
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    ckpt_path = os.path.join(_PROJECT_DIR, "checkpoint", f"checkpoint_{config.sampling.ckpt}.pth")
    state = restore_checkpoint(ckpt_path, state, device=device)
    score_model.eval()
    logging.info(f"Checkpoint loaded")

    # --- Setup SDE ---
    sde = sde_lib.VPSDE(config)

    # --- Load test data ---
    data_dir = os.path.join(_PROJECT_DIR, "data", "knee_singlecoil_test", "singlecoil_test")
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    logging.info(f"Found {len(h5_files)} test files, using: {h5_files[0]}")

    h5_path = os.path.join(data_dir, h5_files[0])
    with h5py.File(h5_path, 'r') as f:
        n_slices = f['kspace'].shape[0]
    slice_idx = n_slices // 2  # middle slice
    logging.info(f"File: {h5_files[0]}, {n_slices} slices, using slice {slice_idx}")

    kspace_full = load_and_normalize(h5_path, slice_idx, config)  # (320, 320) complex
    kspace_full = kspace_full.to(device)

    # --- Generate standard HFS-SDE mask (8x uniform, 24 ACS) ---
    atb_mask = get_mask(config, "sample").to(device)   # (1, 1, 320, 320) complex
    mask_2d = atb_mask[0, 0]                           # (320, 320) complex
    mask_real = mask_2d.real if mask_2d.is_complex() else mask_2d

    n_sampled = mask_real.sum().item()
    acc = mask_real.numel() / (n_sampled + 1e-8)
    logging.info(f"Mask: sampled={int(n_sampled)}/{mask_real.numel()}, acc={acc:.1f}x")

    # --- Build y (under-sampled k-space) and forward operator ---
    if args.mode == 'fake_multi':
        ncoils = args.ncoils
        # Replicate single-coil k-space across coils
        kspace_multi = kspace_full.unsqueeze(0).expand(ncoils, -1, -1).contiguous()  # (C, H, W)
        # Apply mask
        y = kspace_multi * mask_real.unsqueeze(0)  # (C, H, W)
        # Unit sensitivity maps: all 1s → each coil sees the same image
        sens_maps = torch.ones(ncoils, img_sz, img_sz, dtype=torch.complex64, device=device)
        forward_op = MRIForwardOperator(mask_real, sens_maps)

        logging.info(f"Fake multi-coil: y={tuple(y.shape)}, sens_maps={tuple(sens_maps.shape)}")

        # Zero-filled reference via adjoint
        zf_adj = forward_op.adjoint(y)
        zf_abs = zf_adj.abs()
    else:
        # Single-coil: no sens maps
        y = kspace_full * mask_real  # (320, 320) complex
        forward_op = MRIForwardOperator(mask_real)

        # Zero-filled reference via adjoint
        zf_adj = forward_op.adjoint(y)
        zf_abs = zf_adj.abs()

    # --- Create sampler and INR module ---
    sampler = DiffINRSampler(score_model, sde, config).to(device)
    inr_module = INRDCModule(img_size=img_sz).to(device)

    img_shape = (1, 2, img_sz, img_sz)

    # --- Run DiffINR sampling ---
    logging.info(f"Starting DiffINR sampling (T={config.sampling.T})...")
    recon_stacked = sampler.sample(
        y=y,
        forward_op=forward_op,
        img_shape=img_shape,
        inr_dc_module=inr_module,
    )

    recon = torch.complex(recon_stacked[:, 0], recon_stacked[:, 1])  # (1, H, W)
    recon_abs = recon.abs().squeeze()  # (H, W)

    # --- Stats ---
    logging.info(f"Recon: [{recon_abs.min():.4f}, {recon_abs.max():.4f}], "
                 f"mean={recon_abs.mean():.4f}, std={recon_abs.std():.4f}")

    # Quick PSNR vs zero-filled
    zf_abs_norm = zf_abs / (zf_abs.max() + 1e-12)
    recon_abs_norm = recon_abs / (recon_abs.max() + 1e-12)
    mse = torch.mean((zf_abs_norm - recon_abs_norm) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-10))
    logging.info(f"PSNR (vs zero-filled): {psnr.item():.2f} dB")

    # --- Save ---
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + f"_{args.mode}"
    out_dir = os.path.join(_PROJECT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    scio.savemat(os.path.join(out_dir, f"recon_{run_id}.mat"),
                 {"recon": recon_abs.cpu().numpy()})
    scio.savemat(os.path.join(out_dir, f"zf_{run_id}.mat"),
                 {"zf": zf_abs.cpu().numpy()})
    logging.info(f"Saved to {out_dir} (run_id={run_id})")
    logging.info("Done!")


if __name__ == "__main__":
    main()
