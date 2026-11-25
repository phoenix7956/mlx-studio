"""出图 引擎 (第二 套) —— 用 mflux 跑 Z-Image-Turbo / MOSS-Image-SFT.

为什么 单独 一 个 文件:
  Krea-2-Turbo 走 我们 自己 的 krea2_engine (DiT + Qwen3-VL 条件编码器),
  而 Z-Image / MOSS 是 mflux 量化 的 diffusers 布局 —— 两者 权重 格式 完全不同,
  硬 塞 进 一个 文件 会 互相 污染. 所以 分开, 但 接口 保持 一致:
      load() / unload() / generate(prompt, out=..., width=, height=, steps=, seed=)
  这样 server.py 里 的 「按 model id 选 引擎」 才 是 干净 的.
"""
import os
import gc
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import _free  # noqa: E402


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

# ---- 每个 模型 的 落盘 位置 (挨个 试, 谁 有 权重 用 谁) --------------------
_MODELS = {
    "z-image-turbo": {
        "dirs": [
            os.path.join(_H, ".ours/models/z-image-turbo-mflux-q8"),
            os.path.join(_H, ".ours/models/z-image-turbo"),
            os.path.join(_H, "z-image-turbo-mflux-q8"),
        ],
        "cls": "z_image",
        "quant": 8,
        # turbo 模型 —— 步数 少 是 特性, 不是 参数
        "default_steps": 9,
        "max_steps": 9,
        "guidance": None,
        "cls_name": "ZImageTurbo",
    },
    "moss-image-sft": {
        "dirs": [
            os.path.join(_H, ".ours/models/moss-image-sft-mflux-q8"),
            os.path.join(_H, ".ours/models/moss-image-sft"),
            os.path.join(_H, "moss-image-sft-mflux-q8"),
        ],
        "cls": "moss_image",
        "quant": 8,
        "default_steps": 50,
        "max_steps": 50,
        "guidance": 3.0,
        "cls_name": "MOSSImage",
    },
}


def _has_weights(d):
    """目录 里 到底 有没有 权重 —— 光 有 目录 不算数 (下载 中断 会 留 空壳)."""
    if not d or not os.path.isdir(d):
        return False
    for sub in ("transformer", "text_encoder"):
        p = os.path.join(d, sub)
        if os.path.isdir(p):
            for fn in os.listdir(p):
                if fn.endswith(".safetensors"):
                    return True
    return False


def model_dir(model_id):
    """model id -> 权重 目录 (没 下 载 就 None)."""
    spec = _MODELS.get(str(model_id or "").strip())
    if not spec:
        return None
    for d in spec["dirs"]:
        if _has_weights(d):
            return d
    for d in spec["dirs"]:
        if d and os.path.isdir(d):
            return d
    return None


def spec_for(model_id):
    return _MODELS.get(str(model_id or "").strip())


def known_models():
    return list(_MODELS.keys())


class MFluxImageEngine:
    """mflux 出图 引擎 —— 一次 只 挂 一个 模型."""

    def __init__(self, model_id="z-image-turbo"):
        self.model_id = model_id
        self.pipe = None
        self.spec = spec_for(model_id)

    # ------------------------------------------------------------- 装 / 卸
    def load(self):
        if self.pipe is not None:
            return
        self._drop()
        mid = self.model_id
        spec = self.spec or spec_for(mid)
        if not spec:
            raise ValueError("不 认识 的 出图 模型: %s" % mid)
        d = model_dir(mid)
        if not d:
            raise RuntimeError("「%s」的 权重 还 不 在 本机" % mid)
        if spec["cls"] == "z_image":
            from mflux.models.z_image import ZImageTurbo
            self.pipe = ZImageTurbo(quantize=spec["quant"], model_path=d)
        else:
            from mflux.models.moss_image import MOSSImage
            self.pipe = MOSSImage(quantize=spec["quant"], model_path=d)
        gc.collect()

    def _drop(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        _free()

    def unload(self):
        self._drop()

    # ------------------------------------------------------------- 出图
    def generate(self, prompt, out="/tmp/mflux.png", width=1024, height=1024,
                 steps=0, seed=0):
        self.load()
        spec = self.spec or spec_for(self.model_id)
        st = int(steps or 0) or int(spec["default_steps"])
        st = max(1, min(int(spec["max_steps"]), st))
        g = spec.get("guidance")
        img = self.pipe.generate_image(seed=int(seed), prompt=prompt,
                                       num_inference_steps=st,
                                       height=int(height), width=int(width),
                                       guidance=g)
        # mflux 返回 GeneratedImage —— 取 PIL 图 存 出去
        pil = getattr(img, "image", None)
        if pil is None and isinstance(img, (list, tuple)):
            pil = img[0]
        if pil is None:
            pil = img
        pil.save(out)
        return out

    # 条件 生成 (SDEdit / 参考图 / CFG) —— mflux 这 两个 模型 暂 不 支持,
    # 但 接口 要 有, 这样 api_full 那边 不用 分叉 (走 兜底 管线).
    def generate_advanced(self, *a, **kw):
        raise NotImplementedError("%s 暂 不 支持 条件 生成" % self.model_id)

