"""我们 自己 的 出 图 引擎 — Krea-2-Turbo (开源 krea2_engine) + MLX.
权重全 在 本 地, 不 碰 远端."""
import os, sys, time, gc
# Kernel code lives in two places: the rea vendor dir (this repo) and the
# shared ~/.ours path. Try both so the module works on either host.
for _p in (
    os.environ.get("H3M_KREA_DIR", ""),
    os.path.expanduser("~/.ours/rea-ours/_vendor/krea2_engine_src"),
    os.path.expanduser("~/Desktop/rea/ours/_vendor/krea2_engine_src"),
):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import _free

# KREA_DIR: first env override, then the two known on-disk layouts
# (~/.ours/models/ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8,
#  本机: ~/.ours/models/models/krea--Krea-2-Turbo/snapshots/master)
def _home():
    """真实 用户 的 home —— 不 信 $HOME (有些 运行环境 HOME 是 临时 的)."""
    try:
        import pwd
        h = pwd.getpwuid(os.getuid()).pw_dir
        if h and os.path.isdir(h):
            return h
    except Exception:
        pass
    return os.path.expanduser("~")


_H = _home()
_KREA_CANDIDATES = [
    os.environ.get("H3M_KREA_DIR_W", ""),
    # 统一 布局: 权重 直接 摊 在 ~/.ours/models/ 根下 (两台 机器 一致)
    os.path.join(_H, ".ours/models/Krea-2-Turbo-MLX-Serve-mixed-4-8"),
    os.path.join(_H, ".ours/models/ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8"),
    # 魔塔 拉 下来 的 那 一 份 (mixed-4-8, 只 ~10G —— 比 bf16 的 26G 省 内存)
    os.path.join(_H, ".ours/models/mt/ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8"),
    os.path.join(_H, ".ours/models/models/krea--Krea-2-Turbo/snapshots/master"),
    os.path.join(_H, ".ours/models/models/krea--Krea-2-Turbo"),
]
def _has_weights(d):
    """目录 里 到底 有没有 权重 —— 光 有 目录 不算数.

    踩过 的 坑: H3M_KREA_DIR_W 指 到了 一个 下载 中断 的 LTX-2.5 空壳 目录
    (只有 .incomplete 文件), 目录 存在 所以 被 选中, 结果 引擎 报
    "Failed to open file .../turbo.safetensors". 真 权重 在 krea 目录 里.
    """
    if not d or not os.path.isdir(d):
        return False
    for n in ("transformer_mixed_4_8.safetensors", "turbo.safetensors"):
        if os.path.isfile(os.path.join(d, n)):
            return True
    return False


KREA_DIR = next((d for d in _KREA_CANDIDATES if _has_weights(d)),
                 next((d for d in _KREA_CANDIDATES if d and os.path.isdir(d)),
                      _KREA_CANDIDATES[0]))

class ImageEngine:
    # 我们 内核 只 跑 Krea-2-Turbo 出图
    model_id = "krea-turbo"

    def __init__(self):
        self.pipe=None
    def load(self):
        if self.pipe: return
        # 反复 load 堆冷页防护: 先卸干净旧引用, 再 load
        self._drop()
        from krea2_engine.pipeline import Krea2Pipeline
        import os as _os
        # Prefer mixed-4-8 if present, fall back to bf16 (turbo.safetensors)
        mixed = _os.path.join(KREA_DIR, "transformer_mixed_4_8.safetensors")
        bf16  = _os.path.join(KREA_DIR, "turbo.safetensors")
        if _os.path.isfile(mixed):
            self.pipe = Krea2Pipeline(transformer_path=mixed, precision="mixed-4-8", base_dir=KREA_DIR)
        else:
            self.pipe = Krea2Pipeline(transformer_path=bf16, precision="bf16", base_dir=KREA_DIR)
        gc.collect()
    def _drop(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe=None
        _free()
    def unload(self):
        self._drop()
    def generate(self, prompt, out="/tmp/krea.png", width=1344, height=768, steps=4, seed=0):
        self.load()
        imgs=self.pipe.generate(prompt, width=width, height=height, steps=steps, seed=seed)
        imgs[0].save(out)
        return out

    # -------------------------------------------------- 真 的 条件 生成 (SDEdit / CFG / 参考图)
    def generate_advanced(self, prompt, out="/tmp/krea.png", width=1024, height=1024,
                          steps=4, seed=0, negative_prompt="", guidance=0.0,
                          source=None, references=None, ref_weights=None, cond_gain=1.0,
                          strength=1.0, transparent=False, color_transfer=None,
                          color_weight=0.5, step_callback=None):
        """真 的 条件 生成:

        source        : 源图 (mode=edit / variation 的 输入) —— SDEdit 的 起始 潜变量
        references    : 参考图 列表 (ref_images) —— 按 ref_weights 加权 混进 起始 潜变量
        ref_weights   : 每张 参考图 的 权重 (cond_weights)
        cond_gain     : 参考图 相对 源图 的 增益 (cond_gain)
        strength      : SDEdit 的 加噪 强度 (0.1 = 几乎 保留 原图, 1.0 = 完全 重画)
        negative_prompt / guidance : 真 CFG
        transparent   : 输出 RGBA (真 alpha 通道)
        color_transfer: 一张 图, 把 它的 色调 迁移 过来 (变体 用)
        """
        import numpy as np
        from PIL import Image
        import image_advanced as ia

        self.load()
        pipe = self.pipe
        vae = pipe.vae

        src_latent = None
        ref_latents = []
        if source is not None:
            src_latent = ia.encode_image_latent(vae, source, width, height)
        for extra in (references or []):
            if extra is None:
                continue
            ref_latents.append(ia.encode_image_latent(vae, extra, width, height))
        init_latent = ia.mix_init_latents(src_latent, ref_latents, ref_weights, cond_gain)

        arr = ia.sample_advanced(pipe.transformer, vae, pipe._encode_cached, prompt,
                                  width=width, height=height, steps=steps, seed=seed,
                                  negative_prompt=negative_prompt, guidance=guidance,
                                  init_latent=init_latent, strength=strength,
                                  step_callback=step_callback)
        img = Image.fromarray((arr * 255.0).round().clip(0, 255).astype(np.uint8))
        if color_transfer is not None:
            img = ia.reinhard_color_transfer(color_transfer, img, color_weight)
        if transparent:
            img = ia.apply_transparent(img)
            out = out if out.lower().endswith(".png") else out + ".png"
        img.save(out)
        return out
