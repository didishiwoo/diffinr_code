# DiffINR 复现项目 — 技术方案文档 v0.2

> 修正说明：v0.1 使用了 HFS-SDE 的 PC Sampler（Predictor-Corrector），
> 与论文匹配。论文使用 **DDPM-style 反向扩散**（Eq.11），无 Predictor/Corrector/Langevin MCMC。
> v0.2 已全部修正。

---

## 1. 架构总结

```
论文采样 = 简单的 DDPM 反向扩散 (Eq.11) + Tweedie Denoising (Eq.6) + INR-DC
          (无 Predictor-Corrector, 无 Langevin MCMC, 无任何 PC 框架)
```

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DiffINR Pipeline                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  预训练权重: HFS-SDE 的 VP-SDE checkpoint (N=1000, continuous t)    │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │             自实现采样器 (按论文 Algorithm 1)                   │   │
│  │                                                               │   │
│  │  x_T ~ N(0, I)                                                │   │
│  │                                                               │   │
│  │  for t = T(2000) down to 1:                                  │   │
│  │                                                               │   │
│  │    ═══ ① 反向扩散步 (Eq.11) ═══                             │   │
│  │    ε ~ N(0, I) if t > 1 else 0                               │   │
│  │    x_{t-1} = (1 + 0.5β_t)·x_t + β_t·s_θ(x_t, t) + √β_t·ε   │   │
│  │                                                               │   │
│  │    ═══ ② Tweedie Denoising (Eq.6, Score形式: 加号) ═══                       │
│    x₀|ₜ₋₁ = (xₜ₋₁ + (1-ᾱₜ₋₁)·s_θ(xₜ₋₁, t)) / √ᾱₜ₋₁      │   │
│  │                                                               │   │
│  │    ═══ ③ INR-DC (条件触发: 后半程 t ≤ t*) ═══                              │   │
│  │    if t <= t*(1200) and (t-1) % k(50) == 0:                   │   │
│  │      Stage 1: prior_embedding(INR, x₀|ₜ₋₁)                   │   │
│  │      Stage 2: dc_refinement(INR, y, A)                       │   │
│  │      → x̂₀|ₜ₋₁ = INR(d)                                       │   │
│  │      → xₜ₋₁ = √ᾱₜ₋₁·x̂₀|ₜ₋₁ + √(1-ᾱₜ₋₁)·εₜ₋₁ (加噪回映射)  │   │
│  │  end for                                                      │   │
│  │                                                               │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                              │ x₀                                     │
│                              ▼                                        │
│                      输出重建图像                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. 修正清单

| 项目 | v0.1 (错误的) | v0.2 (正确的) | 原因 |
|------|--------------|--------------|------|
| 采样框架 | HFS-SDE PC Sampler (Predictor + Corrector + Langevin) | **自实现论文 Eq.(11) DDPM-style 反向扩散** | HFS-SDE 的 PC 框架与论文不匹配 |
| T (采样步数) | 1000 | **2000** | 论文明确 T=2000 |
| t* (INR 起始) | 1200 | **1200** (T=2000 时合理) | 论文明确 t*=1200 |
| 梯度 DC | 复用 HFS-SDE 梯度 DC | **论文没有"梯度 DC"**，已去除 | 论文只有 INR-DC |
| 代码定位 | 修改 HFS-SDE 的 sampling.py | **自实现 sampler，HFS-SDE 只作为权重和模型加载器** | 避免被 HFS-SDE 框架污染 |

---

## 3. 核心数学推导

### 3.1 β_t 和 ᾱ_t 的定义 (连续时间 VP-SDE)

从 HFS-SDE 的 `VPSDE` 类 (β_min=0.1, β_max=20.0, N=1000):

**β(t)** 连续时间 t ∈ [0, 1]:
```
β(t) = β_min + t · (β_max - β_min)
```

**ᾱ(t)** 累积信号保持系数:
```
√ᾱ(t) = exp(-0.25·t²·(β_max-β_min) - 0.5·t·β_min)
```

### 3.2 反向扩散步 (论文 Eq.11)

对于 T=2000 步离散化，时间步 t (从 T 到 1)：
```
t_cont = (t-1) / T            # 连续时间 [0, 1]
β_t = β(t_cont)                # 来自 VPSDE.sde(x, t) 的 beta

x_{t-1} = (1 + 0.5·β_t/T)·x_t
        + (β_t/T) · s_θ(x_t, t_cont)
        + √(β_t/T) · ε
```

> **重要**: ᾱ(t) 的计算直接复用 `sde_lib.VPSDE.marginal_prob()` 方法，
> 不自实现。该函数提供 `mean` 和 `std`，其中：
> - `mean = √ᾱ(t) · x`  →  `sqrt_alpha_bar = mean / x`
> - `std = √(1-ᾱ(t))`  →  可直接用于 Tweedie 公式
>
> 这避免了手算导致的数值误差，确保与预训练权重一致。

### 3.3 Tweedie Denoising (论文 Eq.6) — ⚠️ 符号修正

> **重要**: 论文 Eq.(6) 使用 DDPM 离散符号惯例，写为 `x_D = (x_t - √(1-ᾱ)·ε_θ) / √ᾱ`。
> 但 HFS-SDE 预训练权重输出的是 **Score**（即 `∇log p_t(x) = -ε / √(1-ᾱ)`），
> 因此代码实现必须使用 **加号**：`x₀ = (x + std² · score) / √ᾱ`。

复用 `VPSDE.marginal_prob`:
```
mean, std = sde.marginal_prob(x, t_cont)
# std = √(1 - ᾱ(t))
# √ᾱ(t) = mean / x = √(1 - std²)

# ✅ 正确公式 (Score 形式，加号):
#   x₀_pred = (x + std² · s_θ(x, t)) / √ᾱ

# ❌ 错误公式 (照抄论文 Eq.6 的减号会导致数值爆炸):
#   x₀_pred = (x - std · s_θ(x, t)) / √ᾱ   ← 这是 DDPM ε-形式，不适用于 Score 模型
```

得:
```python
def tweedie(self, x, t_cont):
    score = self.score_model(x, t_cont * torch.ones(x.shape[0]))
    mean, std = self.sde.marginal_prob(x, t_cont)
    sqrt_alpha_bar = mean / x  # √ᾱ(t)
    # Score 模型输出 ∇log p，必须用加号
    return (x + std * std * score) / sqrt_alpha_bar
```

### 3.4 加噪回映射 (论文 Algorithm 1, line 10-11)

INR 输出干净图像后，加回当前时间步的噪声以保持扩散链：
```
xₜ₋₁ = √ᾱ(t) · x̂₀|ₜ₋₁ + √(1-ᾱ(t)) · εₜ₋₁
```

---

## 4. 模块划分

### 4.1 `diffinr_sampler.py` (核心采样器)

```python
class DiffINRSampler:
    """
    自实现采样器，严格按论文 Algorithm 1。
    不使用 HFS-SDE 的 PC Sampler 任何部分。

    关键设计:
      - ᾱ(t) 计算直接复用 sde_lib.VPSDE.marginal_prob()，不自实现
      - β(t) 计算复用 sde_lib.VPSDE.sde() 中的 beta_t
    """
    def __init__(self, score_model, sde, config):
        self.score_model = score_model  # HFS-SDE 预训练 score network
        self.sde = sde                  # VPSDE 实例，提供 marginal_prob / sde
        self.config = config
        self.T = config.sampling.T         # 2000
        self.t_star = config.sampling.t_star  # 1200
        self.k = config.sampling.k         # 50
        self.beta_min = config.model.beta_min  # 0.1
        self.beta_max = config.model.beta_max  # 20.0

    def _batch_t(self, t_cont, n):
        """将标量 t_cont 扩展为 batch 维度"""
        return t_cont.expand(n)

    def tweedie(self, x, t_cont):
        """Tweedie denoising via sde.marginal_prob (Eq.6, Score-form)

        ⚠️ HFS-SDE 预训练模型输出 Score (∇log p)，而非 ε。
        因此必须使用加号：
          x₀_pred = (x + std² · s_θ(x, t)) / √ᾱ

        复用 VPSDE.marginal_prob:
          mean = √ᾱ(t) · x
          std  = √(1 - ᾱ(t))
        """
        score = self.score_model(x, self._batch_t(t_cont, x.shape[0]))
        mean, std = self.sde.marginal_prob(x, t_cont)
        sqrt_alpha_bar = mean / x  # √ᾱ(t)
        return (x + std**2 * score) / sqrt_alpha_bar

    def reverse_step(self, x, t_cont):
        """Eq.11: x_{t-1} = (1 + 0.5·β_t/T)·x_t + (β_t/T)·s_θ + √(β_t/T)·ε

        β(t) = β_min + t_cont · (β_max - β_min)，连续 t ∈ [0,1]
        离散化: β_discrete = β(t) / T
        """
        beta = (self.beta_min + t_cont * (self.beta_max - self.beta_min)) / self.T
        score = self.score_model(x, self._batch_t(t_cont, x.shape[0]))
        eps = torch.randn_like(x)
        return (1 + 0.5 * beta) * x + beta * score + torch.sqrt(beta) * eps

    def noise_remap(self, x_clean, t_cont):
        """加噪回映射 (Algorithm 1, line 10-11)

        将 INR 输出的干净图像加噪到 t-1 时间步
        xₜ₋₁ = √ᾱ(t) · x̂₀|ₜ₋₁ + √(1-ᾱ(t)) · εₜ₋₁
        """
        mean, std = self.sde.marginal_prob(x_clean, t_cont)
        epsilon = torch.randn_like(x_clean)
        return mean + std * epsilon

    def sample(self, y, forward_op, img_shape):
        """主采样循环 (Algorithm 1)"""
        x = torch.randn(img_shape)  # x_T ~ N(0, I)

        for step in range(self.T, 0, -1):
            # ① 反向扩散步 (Eq.11): x_t → x_{t-1}
            x_prev = self.reverse_step(x, (step - 1) / self.T)

            # ② Tweedie Denoising (Eq.6, Score 形式: 加号)
            x0_pred = self.tweedie(x_prev, step - 1 if step > 1 else 1)

            # ③ INR-DC (条件触发: t > t* 且 (t-1) % k == 0)
            if step > self.t_star and (step - 1) % self.k == 0:
                inr_out = self.inr_dc_module.prior_embedding(x0_pred).dc_refinement(y, forward_op)
                x = self.noise_remap(inr_out, step - 1)  # 加噪回映射
            else:
                x = x_prev  # 直接用反向扩散输出

        return x
```

### 4.2 `inr_dc.py` (INR-DC 模块)

```
inr_dc.py
├── HashEncoding (多分辨率哈希网格)
│   ├── L=4 级分辨率, F=2 维特征
│   ├── 基础分辨率 16, 最大分辨率 512
│   ├── 每级哈希表 T=2¹⁰=1024
│   └── 双线性插值
│
├── INR_MLP (单分量)
│   ├── HashEncoding → Linear(8→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1)
│   └── 参数约 8.5K
│
├── INRDCModule
│   ├── mlp_real: INR_MLP  # 实部
│   ├── mlp_imag: INR_MLP  # 虚部
│   │
│   ├── prior_embedding(x0_pred, lr=1e-3, n_iter=250)
│   │   └── L2 loss ||INR(d) - x0_pred||²
│   │
│   ├── dc_refinement(y, forward_op, lr=1e-5, n_iter=250)
│   │   └── L1 loss ||A·INR(d) - y||₁
│   │
│   ├── forward(coords) → (real, imag)          # 返回实/虚部分量
│   └── to_complex(coords) → complex(real, imag) # 合并为复数，便于调试
```

### 4.3 `forward_operator.py` (MRI 前向算子)

```python
class MRIForwardOperator:
    """A = M·F·S (采样模板 × FFT × 灵敏度图)

    论文 Stage 2 仅需 forward 计算 ||A·INR(d) - y||₁，不需要 adjoint。
    """
    def __init__(self, mask, sens_maps=None):
        self.mask = mask        # 欠采样模板
        self.sens_maps = sens_maps  # 线圈灵敏度图 (多通道)

    def forward(self, x):
        """x: (B, H, W) complex → y: (B, C, H, W) k-space"""
        kspace = torch.fft.fft2(x)          # F(x)
        if self.sens_maps is not None:
            kspace = kspace * self.sens_maps  # S·F(x)
        return kspace * self.mask             # M·S·F(x)
```

---

## 5. 文件结构

```
HFS-SDE/
├── diffinr/                       # ⭐ 新建：DiffINR 模块目录
│   ├── __init__.py
│   ├── diffinr_sampler.py         # 自实现采样器 (Algorithm 1)
│   ├── inr_dc.py                  # INR-DC 模块 (HashEncoding + MLP + 两阶段)
│   ├── hash_encoding.py           # 多分辨率哈希编码
│   ├── inr_mlp.py                 # Tiny MLP
│   └── forward_operator.py        # MRI 前向/反向算子 (FFT + Mask)
│
├── run_lib.py                     # 修改: 新增 diffinr_sample() 入口
├── main.py                        # 修改: 新增 --mode diffinr
├── configs/
│   └── vp/
│       └── diffinr.py             # 新增: DiffINR 专属配置 (T=2000, t*=1200, k=50)
│
├── sampling.py                    # ⚠️ 不做修改，只作为参考
├── sde_lib.py                     # ⚠️ 不做修改，但复用 VPSDE.marginal_prob
└── models/                        # 复用: 模型加载
```

与 HFS-SDE 的集成方式：
- **模型加载**: 复用 `models/model_utils.create_model()` + `run_lib.py` 中的 checkpoint 加载逻辑
- **SDE 数学**: 复用 `sde_lib.VPSDE.marginal_prob()` 计算 √ᾱ(t)
- **工具函数**: 复用 `utils/utils.py` 中的 `Emat_xyt`, `fft2c_2d`, `ifft2c_2d` 等
- **采样器**: **完全自实现**，不调用 `sampling.py` 的任何函数

---

## 6. 参数配置 (论文 vs 实际)

| 参数 | 论文 | HFS-SDE VP-SDE 预训练 | 本实现 |
|------|------|----------------------|--------|
| β_min | — | 0.1 | 0.1 |
| β_max | — | 20.0 | 20.0 |
| 训练步数 | — | 1000 | 1000 (预训练) |
| 采样步数 T | 2000 | — | **2000** |
| t* (INR 起始) | 1200 | — | **1200** |
| k (INR 间隔) | 50 | — | **50** |
| 图像尺寸 | 320×320 | 320×320 | 320×320 |
| 通道数 | 2 (实/虚) | 2 | 2 |
| 去除 PC 框架 | ✅ 无 | ❌ 有 | ✅ **已去除** |
| 添加 INR-DC | ✅ | ❌ | ✅ |
| 添加 Tweedie | ✅ | ❌ | ✅ |
| 添加加噪回映射 | ✅ | ❌ | ✅ |

---

## 7. Hash Encoding 参数

| 参数 | P0 阶段 | 最终 (参考 Shi et al. 2022) |
|------|---------|---------------------------|
| num_levels (L) | **4** | 16 |
| feat_per_level (F) | 2 | 2 |
| log2_hashmap_size | 10 | 15 |
| base_resolution | 16 | 16 |
| finest_resolution | 128 | 512 |

> P0 阶段先用 L=4 验证框架正确性，避免 L=16 导致参数量过大。
> 论文引用的 Shi et al. 2022 使用 L=16，待 P0 通过后再升级。

---

## 8. 验证计划 (P0)

### Step 1: HFS-SDE 原始推理
- 用 HFS-SDE 原生的 PC sampler + phantom 数据，确认环境可运行
- **只做一次，不做修改**，仅确认权重加载正常

### Step 2: 自实现 DDPM 采样 (无 INR)
- 实现 `diffinr_sampler.py` 的基本循环 (Eq.11 + Tweedie)
- 关闭 INR-DC 条件 (`t* > T`，永不触发)
- 在 phantom 上验证输出图像质量

### Step 3: 验证 INR 模块独立功能
- 输入一张已知图像 → Stage 1 prior embedding → 重建 → 对比
- 输入欠采样 k-space → Stage 2 DC refinement → 对比

### Step 4: 完整 DiffINR
- 打开 INR-DC 条件 (t*=1200, k=50)
- 完整跑通 phantom 数据

### Step 5: 对比
- 有 INR-DC vs 无 INR-DC (同一条样链)

---

## 9. 注意事项

1. **score network 输入是连续时间 t ∈ [0, 1]**，与离散步数无关
2. **T=2000 步采样不使用 HFS-SDE 的 VPSDE.discretize()**，而是自实现 Eq.(11)
3. **β_t 在 Eq.(11) 中是连续的 β(t) 除以 T**，对应一步的离散量
4. **两个单独的 MLP** 分别处理实部和虚部
5. 预训练权重下载后存在 `checkpoints/` 下

---

> 文档版本: v0.4
> 修正日期: 2026-06-03
> v0.4 修正: β_t 复用 VPSDE.marginal_prob; Hash Encoding P0 用 L=4; forward_operator 去 adjoint 加 to_complex; INR-DC 条件改为 t <= t*（后半程触发）
