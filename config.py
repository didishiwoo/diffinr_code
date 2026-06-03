"""
DiffINR standalone configuration.

References:
  - DiffINR (Medical Image Analysis, 2025)
  - HFS-SDE (TMI, 2024) — used for VP-SDE pretrained weights
  - fastMRI (NYU)
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # training (only used for model/score network loading)
    config.training = training = ml_collections.ConfigDict()
    training.sde = "vpsde"
    training.continuous = True
    training.reduce_mean = True
    training.batch_size = 1
    training.epochs = 1
    training.snapshot_freq = 50000
    training.log_freq = 50
    training.eval_freq = 100
    training.likelihood_weighting = False

    # sampling — DiffINR specific (Algorithm 1)
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.batch_size = 1
    sampling.ckpt = 190
    sampling.mask_type = "uniform"
    sampling.acc = "8"
    sampling.acs = "24"
    sampling.datashift = "photom"

    # paper Algorithm 1 hyper-parameters
    sampling.mode = "sde"                  # "sde" (continuous Eq.11) or "ddpm" (discrete DDPM update)
    sampling.T = 2000                    # total reverse steps (paper: 2000)
    sampling.t_star = 1200               # INR start timestep (paper: 1200)
    sampling.k = 50                      # INR interval (paper: 50)

    # data
    config.data = data = ml_collections.ConfigDict()
    data.centered = False
    data.dataset_name = "fastMRI_knee"
    data.image_size = 320
    data.num_channels = 2
    data.random_flip = True
    data.uniform_dequantization = False
    data.normalize_type = "std"      # HFS-SDE training uses "std", not "minmax"
    data.normalize_coeff = 1.5       # kspace / (1.5 * std(kspace))

    # model (matched with HFS-SDE VP-SDE pretrained weights)
    config.model = model = ml_collections.ConfigDict()
    model.name = "ddpm"
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.0
    model.dropout = 0.1
    model.sigma_min = 0.01
    model.sigma_max = 348
    model.matrix = True
    model.embedding_type = "fourier"

    # optimization (unused during inference)
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    config.seed = 1000
    config.device = "cuda:0"

    return config
