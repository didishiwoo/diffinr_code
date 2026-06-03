# DiffINR 复现项目 — 需求边界文档 v0.2

> v0.2 更新说明：技术方案确定后，修正与设计方案不一致的内容。
> 核心变更：不复用 HFS-SDE 的 PC Sampler，改为自实现论文 Algorithm 1（DDPM-style 反向扩散）。

---

## 1. 项目目标

以复现 DiffINR 论文实验为目标，基于 **HFS-SDE（TMI 2024）** 的 VP-SDE 预训练权重，
自实现论文 Algorithm 1 的 DDPM-style 反向扩散采样流程，并集成 INR 数据一致性模块，验证 INR-DC 的效果。

## 2. 技术选型

| 模块 | 选择 | 理由 |
|------|------|------|
| **预训练权重** | HFS-SDE 的 VP-SDE checkpoint | 提供预训练 score network，无需从头训练 |
| **采样框架** | **自实现**论文 Eq.(11) DDPM-style 反向扩散 | HFS-SDE 的 PC Sampler（Predictor+Corrector）与论文不匹配，论文无 PC 框架 |
| **SDE 类型** | VP-SDE（β_min=0.1, β_max=20.0, N=1000） | 与论文兼容（论文虽以 VE-SDE 举例，但方法无 SDE 限制） |
| **INR 架构** | 2 个独立 MLP（实部/虚部）× 2 隐层 × 64 神经元 + Hash Encoding + ReLU | 按 DiffINR 论文描述实现 |
| **Hash Encoding** | tiny-cuda-nn（云端 GPU 运行） | 与论文引用的 Shi et al. 2022 一致 |
| **INR 参考** | 自实现（参考 tiny-cuda-nn / instant-ngp） | NeRP 的 SIREN 风格不匹配 |
| **数据集（初期）** | HFS-SDE 自带的 phantom 数据（`data/photom/`） | 零下载成本，先验证 pipeline 正确性 |
| **硬件** | 云端 GPU（NVIDIA） | 本地 Mac（MPS）只做轻量验证 |

## 3. 范围边界

### 包含（In Scope）

- [x] 加载 HFS-SDE 预训练权重（VP-SDE score network）
- [x] 自实现论文 Algorithm 1 采样循环（Eq.11 反向扩散 + Eq.6 Tweedie Denoising）
- [ ] INR 模块实现（Hash Encoding + tiny MLP × 2）
- [ ] INR-DC 两阶段流程（Stage 1 prior embedding + Stage 2 DC refinement）
- [ ] 加噪回映射（保持扩散链连续性）
- [ ] P0 可运行（phantom 数据上输出可视图像）
- [ ] 对比实验：无 INR-DC vs 有 INR-DC（同一采样链）

### 不包含（Out of Scope）

- [ ] 扩散模型训练（直接使用 HFS-SDE 预训练权重）
- [ ] HFS-SDE 的 PC Sampler / Predictor-Corrector / Langevin MCMC
- [ ] 大规模数据集评测（fastMRI 全量下载和评估）
- [ ] P1/P2 级别的指标追赶（PSNR/SSIM 对标论文）
- [ ] 多线圈并行成像支持（先做单通道）
- [ ] 显式梯度 DC 对比（论文无此设计）
- [ ] 训练脚本、分布式支持

## 4. 系统架构

```
     HFS-SDE 预训练权重 + VPSDE 实例 (只用于加载模型和提供 marginal_prob)
                             │
                     ┌───────▼──────────────────────────┐
                     │     自实现 DiffINR Sampler        │
                     │     (不调用 HFS-SDE sampling.py)   │
                     │                                   │
                     │  x_T ~ N(0, I)                    │
                     │                                   │
                     │  for t = T(2000) down to 1:      │
                     │                                   │
                     │    ① 反向扩散 Eq.11               │
                     │      x_{t-1} = (1+0.5β_t/T)·x_t   │
                     │              + (β_t/T)·s_θ + √(β_t/T)·ε │
                     │                                   │
                     │    ② Tweedie Denoising Eq.6      │
                     │      x₀|ₜ₋₁ = (xₜ₋₁ - √(1-ᾱ)·s_θ)/√ᾱ │
                     │       (复用 VPSDE.marginal_prob)   │
                     │                                   │
                     │    ③ INR-DC (条件触发)            │
                     │      if t <= t*(1200) and (t-1)%k(50)==0:   # 后半程触发 │
                     │        Stage 1: prior_embedding   │
                     │        Stage 2: dc_refinement     │
                     │        → 加噪回映射              │
                     │  end for                          │
                     └───────────────┬───────────────────┘
                                     │ x₀
                                     ▼
                             输出重建图像
```

## 5. 交付标准（广义 P0）

1. **能跑通**：从 noise → 完整采样循环 → 输出图像，无报错
2. **图像可见**：输出图像在视觉上不是纯噪声，有解剖结构轮廓
3. **INR-DC 生效**：对比"不启用 INR-DC"的结果，INR-DC 版本重建质量肉眼可辨更好

## 6. 明确不做的决策

- 不修改 HFS-SDE 的 score network 架构和训练流程
- 不调用 HFS-SDE 的 PC Sampler / Predictor / Corrector（完全自实现采样逻辑）
- 不实现显式梯度 DC（论文只用 INR-DC）
- 不实现 DiffINR 论文中完整的全部实验（R=12, R=18 等）
- 不在 HFS-SDE 原本的数据集准备流程上增加新依赖
- 不做 ablation study（除非 P0 完成后有要求）

---

> 文档版本：v0.3
> 创建日期：2026-06-03
> v0.3 修正：INR-DC 条件改为 t <= t*（后半程触发），代码中 step > t_star → step <= t_star
> 审批状态：✅ 已审批
