"""image_advanced.py — 真 的 图像 条件 生成.

我们 本机 的 Krea-2-Turbo 是 纯 文生图 蒸馏 模型, 之前 的 "编辑 / 变体" 是 PIL 拼 的,
guidance_scale / cond_gain / cond_weights / transparent 是 只 收 不 跑 的 空 参数.
这一 版 把 它们 全 接 到 真正 的 采样 器 上:

  * SDEdit (img2img): VAE 编码 原图 -> 潜变量, 按 strength 加噪, 从 对应 时间步 继续 去噪.
                      这 是 真 的 扩散 编辑 (结构 保留 + 提示词 驱动 变化), 不是 像素 拼接.
  * 真 CFG: 正/负 提示词 各自 编码 成 一个 batch, 速度场
            v = v_uncond + guidance * (v_cond - v_uncond).
  * 参考图 加权: 多 张 参考图 的 潜变量 按 cond_weights 加权平均; cond_gain 缩放 参考 部分
                  相对 源图 的 权重 (0 = 忽略 参考图, >1 = 参考 主导).
  * transparent: 边框 背景 估计 -> Otsu 分割 -> 只 留 连通 到 边框 的 背景 -> 距离变换 羽化,
                 输出 真 的 alpha 通道 (RGBA PNG).
  * 变体 的 色彩 迁移: Reinhard Lab 色彩 迁移 (真 算法, 不是 叠色).
"""

import io
import base64

import numpy as np
import mlx.core as mx
from PIL import Image


def _roundup(v, m):
    return ((v + m - 1) // m) * m


def _resample_schedule(ref, s, n):
    """在 [s, 0] 这段 上 重采样 时间表, 保持 官方 曲线 的 形状.

    官方 的 时间表 是 "高噪声 端 密" 的 (shift 过), 而 img2img 只 在 低噪声 段 走 ——
    直接 从 原表 截 尾巴 会 只 剩 1-2 步, 结构 保不住. 这里 先 造 一 条 密 的 参考 曲线,
    再 把 [s,0] 这段 按 n 步 均匀 重采样.
    """
    ref = np.asarray(ref, dtype=np.float64)          # 降序, 1 -> 0
    idx = int(np.searchsorted(-ref, -float(s), side="left"))
    idx = max(0, min(idx, len(ref) - 1))
    tail = ref[idx:]                                  # 降序, <= s -> 0
    if len(tail) < 2:
        return np.linspace(float(s), 0.0, n + 1)
    xs = np.linspace(0.0, 1.0, len(tail))
    ys = tail[::-1]                                   # 升序
    out = np.interp(np.linspace(0.0, 1.0, n + 1), xs, ys)[::-1]
    out[0] = float(s)
    out[-1] = 0.0
    return [float(v) for v in out]


# ---------------------------------------------------------------- VAE 编码
def encode_image_latent(vae, pil, width, height):
    """PIL RGB -> VAE 潜变量 (1,16,H/8,W/8). 与 vae.decode 同 一个 空间."""
    im = pil.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32) / 127.5 - 1.0          # [-1,1]
    x = mx.array(np.ascontiguousarray(arr.transpose(2, 0, 1))[None])  # (1,3,H,W)
    z = vae.encode(x)
    mx.eval(z)
    # Qwen VAE 是 3D 因果 卷积, 编码 出来 带 一个 时间维 (T=1) -> 压掉, 与 采样器 的
    # (b,16,H/8,W/8) 潜变量 布局 对齐.
    if z.ndim == 5:
        z = z[:, :, 0]
    mx.eval(z)
    return z


def mix_init_latents(src_latent, ref_latents, weights, gain):
    """把 源图 与 多张 参考图 的 潜变量 混成 一个 起始 潜变量.

    - weights : 每张 参考图 的 权重 (cond_weights)
    - gain    : 参考 部分 相对 源图 的 增益 (cond_gain). 没有 源图 时, 整体 按 gain 放大.
    """
    gain = float(gain if gain is not None else 1.0)
    ref_latents = list(ref_latents or [])
    if not ref_latents:
        return src_latent
    ws = list(weights or [])
    if len(ws) != len(ref_latents):
        ws = (ws + [1.0] * len(ref_latents))[:len(ref_latents)]
    ws = [max(0.0, float(w)) for w in ws]
    if sum(ws) <= 0:
        ws = [1.0] * len(ref_latents)
    acc = None
    for w, z in zip(ws, ref_latents):
        zi = z * w
        acc = zi if acc is None else acc + zi
    z_ref = acc / float(sum(ws))
    if src_latent is None:
        # 没有 源图: gain 直接 缩放 参考 的 特征 强度 (归一化 潜变量 的 均值 是 0)
        return z_ref * gain
    return (src_latent + z_ref * gain) / (1.0 + gain)


# ---------------------------------------------------------------- 真 CFG + SDEdit 采样
def sample_advanced(transformer, vae, encode, prompt, *, width=1024, height=1024,
                    steps=8, seed=0, negative_prompt=None, guidance=0.0,
                    init_latent=None, strength=1.0, minres=256, maxres=1280,
                    y1=0.5, y2=1.15, mu=None, dtype=mx.bfloat16, step_callback=None):
    """在 krea2_engine 的 采样器 基础上 加 三 件 东西: 真 CFG / 起始 潜变量 / 负向 提示词.

    返回 像素 数组 (1,H,W,3) 的 float32 [0,1].
    """
    from krea2_engine.sampling import (patchify, unpatchify, build_positions,
                                       timesteps)
    try:
        from krea2_engine.sampling import _trim_context
    except Exception:                                    # pragma: no cover
        def _trim_context(ctx, mask):
            return ctx, mask

    patch = transformer.cfg.patch
    comp = vae.spatial_scale
    align = comp * patch
    width = _roundup(width, align)
    height = _roundup(height, align)
    lat_h, lat_w = height // comp, width // comp

    neg = (negative_prompt or "").strip()
    # 真 CFG: guidance > 1 就 走 双 提示词 batch —— 没 给 负向词 时 用 空 提示词
    # 当 无条件 分支 (数学 上 正确 的 CFG), 不 再 "没 负向词 就 当 没 这 个 参数".
    use_cfg = guidance is not None and float(guidance) > 1.0
    prompts = [prompt, neg] if use_cfg else [prompt]

    ctx, mask = encode(prompts)
    ctx, mask = _trim_context(ctx, mask)
    ctx = ctx.astype(dtype)
    txtlen = ctx.shape[1]
    h_, w_ = lat_h // patch, lat_w // patch

    mx.random.seed(seed)
    noise = mx.random.normal((1, vae.latent_channels, lat_h, lat_w)).astype(dtype)

    x1 = (minres // align) ** 2
    x2 = (maxres // align) ** 2
    ts = timesteps(h_ * w_, steps, x1, x2, y1=y1, y2=y2, mu=mu)

    # 起始 潜变量: SDEdit —— x_s = (1-s)*x0 + s*noise
    start = 0
    if init_latent is not None and float(strength) < 1.0:
        s = max(0.0, min(1.0, float(strength)))
        x_s = (1.0 - s) * init_latent.astype(dtype) + s * noise
        img = patchify(x_s, patch)
        # 只在 [s,0] 段 走 —— 用 官方 曲线 重采样 出 完整 的 步数, 保证 结构 改得 动 也 留得 住
        dense = timesteps(h_ * w_, 64, x1, x2, y1=y1, y2=y2, mu=mu)
        ts = _resample_schedule(dense, s, steps)
        start = 0
    else:
        img = patchify(noise, patch)

    pos = build_positions(len(prompts), txtlen, h_, w_)
    full_mask = mx.concatenate([mask, mx.ones((len(prompts), h_ * w_))], axis=1)
    fused_ctx, cos, sin, add_mask = transformer.prepare_conditioning(ctx, pos, full_mask, dtype)
    mx.eval(fused_ctx, cos, sin, *([add_mask] if add_mask is not None else []))

    total = max(1, len(ts) - 1 - start)
    for i in range(start, len(ts) - 1):
        tc, tp = ts[i], ts[i + 1]
        t = mx.full((len(prompts),), tc, dtype=dtype)
        if use_cfg:
            # cond 与 uncond 走 同一个 batch, 噪声 相同 -> 只有 文本 条件 不同
            img2 = mx.concatenate([img, img], axis=0)
            v2 = transformer.denoise_step(img2, fused_ctx, t, cos, sin, add_mask)
            v_cond, v_unc = v2[0:1], v2[1:2]
            v = v_unc + float(guidance) * (v_cond - v_unc)
        else:
            v = transformer.denoise_step(img, fused_ctx, t, cos, sin, add_mask)
        img = img + (tp - tc) * v
        mx.eval(img)
        if step_callback is not None:
            step_callback(i - start + 1, total)

    latent = unpatchify(img, patch, h_, w_, vae.latent_channels)
    decoded = vae.decode(latent.astype(mx.float32))
    decoded = mx.clip(decoded, -1, 1) * 0.5 + 0.5
    decoded = decoded[:, :, 0]
    mx.eval(decoded)
    arr = np.array(decoded.astype(mx.float32))[0]      # (3,H,W)
    return np.transpose(arr, (1, 2, 0))                # (H,W,3)


# ---------------------------------------------------------------- 透明 抠图
def _otsu(x, nbins=256):
    """Otsu 阈值 (自己 实现 —— skimage 不 在 依赖 里)."""
    hist, edges = np.histogram(x, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w1 = np.cumsum(hist).astype(np.float64)
    w2 = np.cumsum(hist[::-1]).astype(np.float64)[::-1]
    m1 = np.cumsum(hist * centers)
    m2 = np.cumsum((hist * centers)[::-1])[::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = np.where(w1 > 0, m1 / np.maximum(w1, 1e-9), 0.0)
        m2 = np.where(w2 > 0, m2 / np.maximum(w2, 1e-9), 0.0)
    between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return float(centers[:-1][int(np.argmax(between))])


def make_alpha_mask(img):
    """真 的 alpha 遮罩: 边框 背景 估计 -> Otsu -> 连通 到 边框 的 背景 -> 距离变换 羽化.

    返回 (H,W) float32 [0,1]; 抠 不 出 前景 时 返回 None.
    """
    try:
        from scipy import ndimage
    except Exception:
        return None
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    h, w, _ = a.shape
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], axis=0)
    bg = np.median(border, axis=0)
    d = np.linalg.norm(a - bg, axis=2)
    d = d / max(1e-6, float(np.linalg.norm(bg)))
    thr = _otsu(d.ravel())
    bgmask = d < thr
    lab, n = ndimage.label(bgmask)
    if n:
        edge = set(lab[0, :].tolist()) | set(lab[-1, :].tolist())
        edge |= set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
        edge.discard(0)
        if edge:
            keep = np.isin(lab, sorted(edge))
            if keep.any():
                bgmask = keep
    fg = ~bgmask
    if not fg.any() or fg.all():
        return None
    dist = ndimage.distance_transform_edt(bgmask)
    feather = max(2.0, 0.02 * float(np.hypot(h, w)))
    alpha = np.clip(dist / feather, 0.0, 1.0)
    alpha = ndimage.gaussian_filter(alpha, sigma=max(1.0, feather / 3.0))
    alpha = np.where(fg, 1.0, alpha)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def apply_transparent(img):
    """给 图 加 真 的 alpha 通道 (RGBA). 抠 不 出 前景 就 原样 返回."""
    alpha = make_alpha_mask(img)
    if alpha is None:
        return img.convert("RGB")
    rgba = img.convert("RGBA")
    rgba.putalpha(Image.fromarray((alpha * 255).astype(np.uint8), "L"))
    return rgba


# ---------------------------------------------------------------- 色彩 迁移
def reinhard_color_transfer(src, dst, weight=1.0):
    """Reinhard Lab 色彩 迁移: 把 src 的 色调 搬到 dst 上 (weight 0-1)."""
    try:
        from scipy import ndimage
    except Exception:
        return dst
    s = np.asarray(src.convert("RGB")).astype(np.float32) / 255.0
    d = np.asarray(dst.convert("RGB")).astype(np.float32) / 255.0
    if s.shape != d.shape:
        s = np.asarray(src.convert("RGB").resize(dst.size, Image.LANCZOS)).astype(np.float32) / 255.0

    def to_lab(rgb):
        # sRGB -> linear -> XYZ -> Lab (D65)
        m = rgb.reshape(-1, 3)
        lin = np.where(m <= 0.04045, m / 12.92, ((m + 0.055) / 1.055) ** 2.4)
        mat = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
        xyz = lin @ mat.T
        white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
        xyz = xyz / white
        eps = 216.0 / 24389.0
        k = 24389.0 / 27.0
        f = np.where(xyz > eps, np.cbrt(np.maximum(xyz, 0)), (k * xyz + 16.0) / 116.0)
        L = 116.0 * f[:, 1] - 16.0
        a = 500.0 * (f[:, 0] - f[:, 1])
        b = 200.0 * (f[:, 1] - f[:, 2])
        return np.stack([L, a, b], axis=1).reshape(rgb.shape)

    def to_rgb(lab):
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        fy = (L + 16.0) / 116.0
        fx = fy + a / 500.0
        fz = fy - b / 200.0
        eps = 216.0 / 24389.0
        k = 24389.0 / 27.0
        finv = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / k)
        finv_y = np.where(fy ** 3 > eps, fy ** 3, (116.0 * fy - 16.0) / k)
        finv_z = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / k)
        white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
        xyz = np.stack([finv, finv_y, finv_z], axis=-1) * white
        mat = np.array([[3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660, 1.8760108, 0.0415560],
                        [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)
        lin = xyz @ mat.T
        srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.maximum(lin, 0), 1 / 2.4) - 0.055)
        return np.clip(srgb, 0.0, 1.0).reshape(lab.shape)

    ls, ld = to_lab(s), to_lab(d)
    ms, sd = ls.mean(axis=0), ls.std(axis=0) + 1e-6
    md, dd = ld.mean(axis=0), ld.std(axis=0) + 1e-6
    out = (ld - md) * (sd / dd) + ms
    out = ld * (1.0 - weight) + out * weight
    rgb = to_rgb(out)
    return Image.fromarray((rgb * 255).round().clip(0, 255).astype(np.uint8))
