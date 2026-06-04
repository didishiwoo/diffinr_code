# Phantom DiffINR — 主实验结果

> 生成时间: 2026-06-04 08:12

## 说明

DiffINR 完整管线的主要实验结果（phantom 数据）。包含多次运行、对应 zerofilled 重建、
消融实验（无 INR-DC）以及各种可视化对比图。

## 文件列表

**MAT 文件 (9)**

| 文件 | 大小 | 类型 |
|------|------|------|
| `recon_20260603_181641.mat` | 401.4 KB | DiffINR |
| `recon_20260603_182407.mat` | 401.4 KB | DiffINR |
| `recon_20260603_183548.mat` | 401.4 KB | DiffINR |
| `recon_20260603_184332.mat` | 401.4 KB | DiffINR |
| `recon_no_inr_20260603_184814.mat` | 401.4 KB | Ablation(w/o INR) |
| `zf_20260603_181641.mat` | 1604.9 KB | Zero-filled |
| `zf_20260603_182407.mat` | 802.5 KB | Zero-filled |
| `zf_20260603_183548.mat` | 802.5 KB | Zero-filled |
| `zf_20260603_184332.mat` | 802.5 KB | Zero-filled |

**PNG 文件 (11)**

| 文件 | 大小 | 内容 |
|------|------|------|
| `ablation_20260603_184814.png` | 2083.7 KB | 消融实验 |
| `ablation_comparison.png` | 871.9 KB | 消融对比 |
| `ablation_summary.png` | 1921.8 KB | 消融汇总 |
| `compare_183548.png` | 843.5 KB | 重建对比图 |
| `comparison_ddpm_184332.png` | 2141.4 KB | DDPM 基线对比 |
| `comparison_run_181641.png` | 817.8 KB | 多轮运行对比 |
| `comparison_run_182407.png` | 917.7 KB | 多轮运行对比 |
| `recon_viz_183548.png` | 457.3 KB | 重建可视化 |
| `zf_20260603_181641.png` | 398.1 KB | Zero-filled 可视化 |
| `zf_visualization_20260603_181641.png` | 339.8 KB | ZF 可视化 |
| `zf_visualization_20260603_182407.png` | 325.8 KB | ZF 可视化 |

## 运行时间线

| 运行 (timestamp) | 方法 | 范围 | 均值 | 说明 |
|-----------------|------|------|------|------|
| 181641 | DiffINR | [0.08, 11.73] | 3.15 | (1,1,320,320) 未 squeeze |
| 182407 | DiffINR | [0.00, 9.38] | 2.50 |  |
| 183548 | DiffINR | [0.01, 8.89] | 2.88 |  |
| 184332 | DiffINR | [0.00, 4.19] | 0.95 |  |
| 184814 | Ablation w/o INR | [0.00, 3.99] | 0.95 |  |
