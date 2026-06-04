# DiffINR 复现项目 — 实验结果总览

> 更新日期: 2026-06-04

---

## 目录结构

```
results/
├── README.md                      # 📄 本文档（整体分析与总结）
│
├── ground_truth/                  # Phantom 完全采样真值
│   ├── comparison_all.png         #   可视化对比图
│   └── README.md                  #   分析文档
│
├── v0_phantom/                    # v0 早期复数域重建实验
│   ├── comparison_all.png
│   └── README.md
│
├── phantom_diffinr/               # DiffINR 主实验（phantom 数据）
│   ├── comparison_all.png
│   └── README.md
│
├── improved/                      # 后续改进版本（缩放修复后）
│   ├── comparison_all.png
│   └── README.md
│
├── pure_ddpm/                     # 纯 DDPM 基线（无 INR-DC）
│   ├── comparison_all.png
│   └── README.md
│
├── result/                        # 展平的总览对比图
│   ├── ground_truth.png
│   ├── v0_phantom.png
│   ├── phantom_diffinr.png
│   ├── improved.png
│   └── pure_ddpm.png
│
└── back_up/                       # 原始数据 + 历史图片归档
    ├── ground_truth/
    ├── v0_phantom/
    ├── phantom_diffinr/
    ├── improved/
    └── pure_ddpm/
```

---

## 1. 实验流水线概览

```
论文 Algorithm 1 实现流程:

  x_T ~ N(0, I)
  for step = T(2000) → 1:
    ① Reverse diffusion (Eq.11)
    ② Tweedie denoising (Eq.6) → x_{0|t-1}
    ③ INR-DC (如果 t > t* 且 (t-1) % k == 0):
         Stage 1: Prior embedding (L2, lr=1e-3, 250it)
         Stage 2: DC refinement   (L1, lr=1e-5, 250it)
         Noise remap → x_{t-1}
```

- **预训练模型**: HFS-SDE (TMI 2024) VP-SDE, β_min=0.1, β_max=20.0, N=1000
- **采样参数**: T=2000, t\*=1200, k=50
- **INR 架构**: HashEncoding (L=4, F=2) + 2×MLP (64×2, 实部/虚部分离)
- **数据**: Phantom 320×320, 单线圈

---

## 2. 各组实验结果汇总

### 2.1 Ground Truth（基线真值）

| 指标 | 值 |
|------|------|
| 文件 | `gt_0_1.mat` |
| 形状 | (320, 320) complex128 |
| 幅度范围 | [0.0000, 1.0000] |
| 幅度均值 | 0.0933 |
| 说明 | 全采样重建，作为所有重建的质量参考 |

> **注意**: GT 幅度范围约 [0, 1]，所有重建结果的数值范围应与此对比判断缩放正确性。

---

### 2.2 v0 Phantom（早期复数域实验）

| 运行 | 方法 | 幅度范围 | 幅度均值 | 说明 |
|------|------|---------|---------|------|
| 161343 | DDPM reverse diffusion | [0.0009, 0.7055] | 0.1205 | |
| 172055 | DDPM reverse diffusion | [0.0048, 5.4103] | 1.0960 | 部分像素偏高 |
| 185329 | DDPM reverse diffusion | [0.0009, 3.5741] | 0.9714 | |
| 193016 | DDPM reverse diffusion | [0.3544, 2.9800] | 1.1620 | |
| — | Zero-filled (zf.mat) | [0.0000, 1.0943] | 0.1016 | |

**特点**: 复数域重建，结果参差不齐，范围不稳定。该阶段主要验证 diffusion pipeline 可以跑通。

---

### 2.3 Phantom DiffINR（主实验）

| 运行 | 方法 | 范围 | 均值 | 说明 |
|------|------|------|------|------|
| 181641 | DDPM + INR-DC | [0.08, 11.73] | 3.15 | (1,1,320,320) 未 squeeze |
| 182407 | DDPM + INR-DC | [0.003, 9.38] | 2.50 | |
| 183548 | DDPM + INR-DC | [0.007, 8.89] | 2.88 | **数值偏高** |
| 184332 | DDPM + INR-DC | [0.002, 4.19] | 0.95 | ✅ 范围趋于合理 |
| 184814 | DDPM only (ablation) | [0.003, 3.99] | 0.95 | ✅ Ablation 对照 |

**趋势**: 从 181641 → 184332，数值范围逐步收敛（max 从 11.73 降至 4.19）。
184332 与 ablation 的均值相同 (0.95)，但 INR-DC 版本的上限稍高 (4.19 vs 3.99)。

---

### 2.4 Improved（改进版本）

| 运行 | 方法 | 范围 | 均值 | 说明 |
|------|------|------|------|------|
| 190229 | DDPM + INR-DC | [0.003, 3.75] | 0.95 | 进一步稳定 |
| 190633 | DDPM + INR-DC | [0.005, 3.67] | 0.97 | 与 190229 一致 |
| 200457 | DDPM + INR-DC (fake multi) | [0.247, 1.05] | 0.58 | fake multi-coil 测试 |

**特点**: 数值范围收敛到 [0, 3.7] 左右，对比主实验版明显改善。200457 的 fake multi-coil 测试范围较窄。

---

### 2.5 Pure DDPM（无 INR-DC 基线）

| 版本 | 范围 | 均值 | 说明 |
|------|------|------|------|
| v1 (pure_ddpm.mat) | [0.008, 8.15] | 2.20 | 初版，偏亮 |
| v2 (pure_ddpm_v2.mat) | [7.474, 50.29] | 44.33 | ⚠️ **异常！需排查** |
| v3 (pure_ddpm_v3.mat) | [0.005, 3.48] | 0.96 | ✅ 可用基线 |

---

## 3. 关键对比

### 3.1 DiffINR vs Pure DDPM（184332 vs pure_ddpm_v3）

| 方法 | 范围 | 均值 |
|------|------|------|
| Pure DDPM (v3) | [0.005, 3.48] | 0.96 |
| DiffINR (184332) | [0.002, 4.19] | 0.95 |
| Ablation (184814) | [0.003, 3.99] | 0.95 |

两者很接近，DiffINR max 略高 (4.19 vs 3.48)，说明 INR-DC 可能保留了更多高频细节。

### 3.2 数值范围演进

```
Early (v0)  ── 范围不稳定 [0, 5.4]  — 复数域重建
      │
      ▼
Main (phantom_diffinr)
      181641: max=11.73 ⚠️ 偏高
      182407: max=9.38
      183548: max=8.89
      184332: max=4.19  ✅ 收敛
      │
      ▼
Improved:  max≈3.7       ✅ 稳定
Pure DDPM: max≈3.5       ✅ 稳定
```

### 3.3 Zero-filled 重建对比

| 组 | ZF 范围 | ZF 均值 | Recon 均值 | 提升 |
|----|---------|---------|-----------|------|
| v0_phantom | [0, 1.09] | 0.10 | 0.12~1.16 | 不稳定 |
| phantom_diffinr | [0.001, 0.94] | 0.41 | 0.95~3.15 | 稳定 |
| improved | [0.001, 1.40] | 0.61 | 0.95~0.97 | 稳定 |

---

## 4. 已知问题

1. **v2 DDPM 范围异常** (`pure_ddpm_v2.mat`): mean=44.33, max=50.29, 远超出正常范围，可能由于归一化/缩放错误导致。
2. **主实验早期版本偏亮** (181641~183548): max 在 8~12 之间，与 GT (max≈1) 不匹配，后期通过修复采样逻辑收敛。
3. **复数域处理**: 所有 v0 结果为 complex-valued，取幅度可视化时需注意范围差异。
4. **fake multi-coil** (`200457_fake_multi`): 目前仅做功能性测试，数值范围窄 (0.25~1.05)，需真实多线圈数据验证。

---

## 5. 结论

- ✅ **DiffINR pipeline 跑通** — 从 noise → 2000 步反向扩散 → INR-DC 重构，整体流程无报错
- ✅ **INR-DC 功能正常** — ablation 对比显示有/无 INR-DC 均可运行
- ✅ **数值范围已收敛** — 后期版本均值稳定在 0.95 左右
- ⚠️ **重建质量待提升** — 当前结果虽可见解剖轮廓，但与 GT 差距较大（均值差异 ~0.85），需进一步优化采样逻辑
- 📌 **下一步**: 检查 score network 输出缩放、验证加噪回映射数值正确性、过渡到真实 k-space 数据测试

---

## 6. 可视化文件索引

| 文件 | 路径 |
|------|------|
| 总览对比图 (展平) | `result/groups/*.png` |
| Ground Truth 对比 | `ground_truth/comparison_all.png` |
| v0 实验对比 | `v0_phantom/comparison_all.png` |
| 主实验对比 | `phantom_diffinr/comparison_all.png` |
| 改进版本对比 | `improved/comparison_all.png` |
| DDPM 基线对比 | `pure_ddpm/comparison_all.png` |
| 原始数据归档 | `back_up/groups/` |
