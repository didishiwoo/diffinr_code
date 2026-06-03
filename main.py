"""
DiffINR: INR-based posterior sampling for diffusion models.

Standalone inference entry point — fully self-contained.
Upload the entire `diffinr_code/` folder to cloud GPU and run.

Usage:
    cd diffinr_code
    pip install -r requirements.txt
    python main.py
"""

import os
import sys
import logging
from datetime import datetime

# All HFS-SDE dependencies are vendored in hfs_sde/
# Phantom test data is in data/photom/
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "hfs_sde"))

import torch
import numpy as np

import sde_lib
import losses
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
from models import ddpm  # register DDPM model
from utils.utils import (
    get_mask,
    Emat_xyt_complex,
    r2c,
    restore_checkpoint,
)
import utils.datasets as datasets

from config import get_config
from diffinr.diffinr_sampler import DiffINRSampler
from diffinr.inr_dc import INRDCModule
from diffinr.forward_operator import MRIForwardOperator
import visualize  # our timestamped visualization module


def main():
    # Generate unique run ID (timestamp)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = get_config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = str(device)
    logging.info(f"Using device: {device}")
    logging.info(f"Run ID: {run_id}")
    logging.info(f"Sampling config: T={config.sampling.T}, t*={config.sampling.t_star}, "
                 f"k={config.sampling.k}, mask={config.sampling.mask_type}_"
                 f"acc{config.sampling.acc}_acs{config.sampling.acs}, "
                 f"checkpoint={config.sampling.ckpt}")
    logging.info(f"Image size: {config.data.image_size}x{config.data.image_size}")

    # 1. Load VP-SDE pretrained score network
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    ckpt_path = os.path.join(_PROJECT_DIR, "checkpoint", f"checkpoint_{config.sampling.ckpt}.pth")
    state = restore_checkpoint(ckpt_path, state, device=device)
    score_model.eval()
    logging.info(f"Loaded checkpoint: {ckpt_path}")

    # 2. Setup SDE
    sde = sde_lib.VPSDE(config)

    # 3. Load under-sampling mask
    atb_mask = get_mask(config, "sample").to(device)

    # 4. Load phantom test dataset
    test_dl = datasets.get_dataset(config, "photom")

    # 5. Create DiffINR sampler + INR module
    sampler = DiffINRSampler(score_model, sde, config).to(device)
    inr_module = INRDCModule(img_size=config.data.image_size).to(device)

    img_shape = (
        config.sampling.batch_size,      # 1
        config.data.num_channels,         # 2 (real/imag)
        config.data.image_size,           # 320
        config.data.image_size,           # 320
    )

    # 6. Run on each test sample
    for index, point in enumerate(test_dl):
        logging.info(f"--- Sample {index} ---")

        k0, csm = point
        k0 = k0.to(device)
        csm = csm.to(device)

        # Ground truth image
        gt = Emat_xyt_complex(k0, True, csm, 1.0)
        gt = gt.squeeze(1)

        # Under-sampled k-space + forward operator
        y = k0 * atb_mask

        if csm is not None and csm.shape[1] > 1:
            forward_op = MRIForwardOperator(atb_mask[0, 0], csm[0])
            y_single = y[0]
        else:
            forward_op = MRIForwardOperator(atb_mask[0, 0])
            y_single = y[0, 0]

        # DiffINR sampling
        with torch.no_grad():
            recon_stacked = sampler.sample(
                y=y_single,
                forward_op=forward_op,
                img_shape=img_shape,
                inr_dc_module=inr_module,
            )

        recon = r2c(recon_stacked)

        # Save with run_id (never overwrites)
        out_dir = os.path.join(_PROJECT_DIR, "results")
        os.makedirs(out_dir, exist_ok=True)

        visualize.save_mat(out_dir, run_id, recon, f"recon_{index}")
        visualize.save_mat(out_dir, run_id, gt, f"gt_{index}")

        # Quick PSNR
        gt_abs, recon_abs = gt.abs(), recon.abs()
        mse = torch.mean((gt_abs - recon_abs) ** 2)
        psnr = 20 * torch.log10(gt_abs.max() / torch.sqrt(mse))
        logging.info(f"Sample {index}: PSNR ≈ {psnr.item():.2f} dB")

    logging.info("DiffINR sampling completed.")

    # Auto-visualize
    logging.info(f"Generating comparison figure for run_id={run_id} ...")
    visualize.visualize_run(out_dir, run_id)
    logging.info("Done.")


if __name__ == "__main__":
    main()
