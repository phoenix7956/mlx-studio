"""DMD 采样器 —— Wan2.2 的 **蒸馏快版** (FastWan2.2) 必需.

背景:
   Wan2.2 TI2V-5B 原版 要 40 步. 我们 实测: 720P/121 帧/4 步 = 179 秒
   (44.7 秒/步) —— 40 步 就 30 分钟, 加上 VAE 解码 4.5 分钟, **比 H3 还慢**.
   原因 是 mlx-video 用 的 全注意力 (seq 27,280), 而 我们 的 H3 port 有 自研
   稀疏注意力 Metal kernel.

   FastWan2.2 是 同一个 DiT 的 **DMD 蒸馏版**: 3 步 出片, 不用 CFG.
   3 步 + guide=1 (B=1 批, 2 倍快) ≈ **1/13** 的 去噪 时间.

为什么 要 自己 写:
   上游 mlx-video main **没有** DMD 支持 (fastwan 是 lBroth fork 里 的,
   那份 代码 我们 拿不到/是 截断 的). 所以 这里 按 上游 的 flow-matching
   约定 自己 实现 一个 —— 接口 跟 上游 的 scheduler 完全 一致, 所以 只 要
   把 上游 的 FlowUniPCScheduler 换掉 就 能 用.

DMD 采样 (flow matching 约定):
   x_t = (1-σ)·x0 + σ·ε ,  模型 出 v = ε - x0
   → x0 = x_t - σ·v                        (预测 干净 图)
   → x_{t+1} = (1-σ_next)·x0 + σ_next·ε'  (renoise: ε' 是 新 的 噪声)
"""
import mlx.core as mx
import numpy as np


class FlowDMDScheduler:
    """DMD 采样器 —— 固定 sigma 序列 + renoise.

    sigmas : [1.0, 0.757, 0.522, 0.0]  (FastWan2.2 官方 的 3 步 配方)
    renoise: 每步 重新 注入 噪声 (DMD 的 标准 做法)
    """

    def __init__(self, num_train_timesteps: int = 1000, sigmas=None,
                 renoise: bool = True, seed: int = -1):
        self.num_train_timesteps = num_train_timesteps
        self._dmd_sigmas = list(sigmas) if sigmas else [1.0, 0.757, 0.522, 0.0]
        self.renoise = bool(renoise)
        self.seed = seed
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_steps: int = None, shift: float = 1.0):
        """上游 会 调 这个 —— 我们 用 固定 的 DMD sigma 序列, 忽略 shift."""
        sigmas = list(self._dmd_sigmas)
        if num_steps is not None and num_steps != len(sigmas) - 1:
            # 上游 传 了 别的 步数 —— 按 比例 重新 采样 一条 等距 序列,
            # 但 保留 首尾 (1.0 -> 0.0), 免得 步数 对 不上 时 出 坏 图.
            n = max(1, int(num_steps))
            sigmas = list(np.linspace(1.0, 0.0, n + 1))
        self.sigmas = mx.array(np.array(sigmas, dtype=np.float32))
        self.timesteps = mx.array(
            (np.array(sigmas[:-1]) * self.num_train_timesteps)
            .astype(np.int64).astype(np.float32))
        self._sigmas_float = [float(s) for s in sigmas]
        self._step_index = 0
        self._noise = None

    def step(self, model_output, timestep, sample):
        """一步 DMD: 预测 x0 → renoise 到 下一个 sigma."""
        i = self._step_index
        s = self._sigmas_float[i]
        s_next = self._sigmas_float[i + 1] if i + 1 < len(self._sigmas_float) else 0.0

        # flow matching: v = ε - x0  →  x0 = x_t - σ·v
        x0 = sample - s * model_output

        if s_next <= 0.0:
            out = x0
        elif not self.renoise:
            # 不 renoise: 复用 最初 那份 噪声 (确定性)
            if self._noise is None:
                self._noise = mx.random.normal(sample.shape)
            out = (1.0 - s_next) * x0 + s_next * self._noise
        else:
            # renoise: 每步 注入 新 噪声
            out = (1.0 - s_next) * x0 + s_next * mx.random.normal(sample.shape)

        self._step_index += 1
        return out

    def reset(self):
        self._step_index = 0
        self._noise = None


def install(monkey_target):
    """把 上游 的 FlowUniPCScheduler 换成 DMD 版 —— 只 在 蒸馏 权重 上 用."""
    import mlx_video.models.wan_2.scheduler as SCH

    def _make(num_train_timesteps: int = 1000):
        return FlowDMDScheduler(num_train_timesteps=num_train_timesteps)

    SCH.FlowUniPCScheduler = _make
    return SCH
