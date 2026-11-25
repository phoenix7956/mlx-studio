"""Wan2.2 TI2V-5B 的 MLX 推理 包装 层.

上游: Blaizzy/mlx-video (MIT) —— 引擎 源码 vendor 在 vendor_src/mlx_video/.
这 层 只 做 三件事: DiT 常驻 / 截 帧 / 本地 tokenizer. 不 改 上游 逻辑.
"""
import os, sys, gc, time
from pathlib import Path

import numpy as np

# 上游 vendor 源码 位置 —— 我们 自己 仓库 里 的 vendor_src/mlx_video
def _find_vendor():
    """找 vendor 引擎 源码 (mlx_video/ 的 父目录) —— 几 个 位置 都 试."""
    env = os.environ.get("WAN2_VENDOR_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [
        os.path.join(here, "vendor_src"),              # 开发 态: <repo>/vendor_src
        os.path.join(here, "vendor"),                  # 装 完: <runtime>/vendor
        os.path.expanduser("~/.ours/vendor"),          # 共用 布局
        os.path.join(here, "mlx_video"),               # 直接 指 到 包
    ]
    for c in cands:
        if os.path.isdir(os.path.join(c, "mlx_video")):
            return c
    return cands[0]


VENDOR = _find_vendor()
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)


class Wan2Result:
    """跟 H3 引擎 一样 的 返回 契约 —— server 只 看 这 几个 字段."""

    def __init__(self, video, fps=24, sample_rate=48000, sequence_length=0,
                 seconds_per_step=0.0):
        self.video = video                      # np.uint8 [T, H, W, 3]
        self.fps = fps
        self.sample_rate = sample_rate
        self.sequence_length = sequence_length
        self.seconds_per_step = seconds_per_step
        # Wan2.2 出 的 是 静音 画面 —— 音轨 由 我们 本机 音乐 合成 后 mux 进去.
        n = int(round(len(video) / max(1, fps) * sample_rate))
        self.audio = np.zeros((max(1, n), 2), dtype=np.int16)


def _free_metal():
    gc.collect()
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass
    gc.collect()


class Wan2Pipeline:
    """一个 Wan2.2 TI2V-5B 权重 目录 = 一个 pipeline."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self._model = None        # DiT (常驻)
        self._model_key = None
        self._last = 0.0
        self._config = None
        self._dmd = None

    # ---------------------------------------------------------------- 装 / 卸
    def _load_config(self):
        if self._config is not None:
            return self._config
        from mlx_video.models.wan_2.config import WanModelConfig
        cfg_path = self.model_dir / "config.json"
        if cfg_path.exists():
            import json
            d = json.load(open(cfg_path))
            d.pop("quantization", None)
            for k in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
                if k in d and isinstance(d[k], list):
                    d[k] = tuple(d[k])
            self._config = WanModelConfig(
                **{k: v for k, v in d.items()
                   if k in WanModelConfig.__dataclass_fields__})
        else:
            self._config = WanModelConfig.wan22_ti2v_5b()
        return self._config

    def _dmd_recipe(self):
        """config.json 里 有 没有 DMD 配方 (FastWan2.2 蒸馏版).

        有 -> 3 步 + guide=1 (不用 CFG) + 我们 自己 的 renoise 采样器.
        返回 (sigmas, renoise) 或 None.
        """
        if self._dmd is not None:
            return self._dmd
        self._dmd = None
        cfg_path = self.model_dir / "config.json"
        if cfg_path.exists():
            try:
                import json
                d = json.load(open(cfg_path))
                r = d.get("fastwan_dmd")
                if isinstance(r, dict):
                    self._dmd = (
                        list(r.get("sigmas") or [1.0, 0.757, 0.522, 0.0]),
                        bool(r.get("renoise", True)))
            except Exception:
                self._dmd = None
        return self._dmd

    def default_steps(self):
        """这个 权重 要 几 步 —— DMD 蒸馏版 3 步, 原版 40 步."""
        r = self._dmd_recipe()
        if r:
            return len(r[0]) - 1
        return 40

    def is_dmd(self):
        return self._dmd_recipe() is not None

    def load(self):
        """把 DiT 装 上来 常驻 (T5 / VAE 出片 时 临时 装)."""
        if self._model is not None:
            return
        cfg = self._load_config()
        from mlx_video.models.wan_2.utils import load_wan_model
        import json
        q = None
        cfg_path = self.model_dir / "config.json"
        if cfg_path.exists():
            q = json.load(open(cfg_path)).get("quantization")
        mpath = self.model_dir / "model.safetensors"
        if not mpath.exists():
            raise FileNotFoundError("Wan2.2 权重 不 全: 缺 %s" % mpath)
        self._model = load_wan_model(mpath, cfg, q)
        self._model_key = (str(mpath), q.get("bits") if q else None)
        self._last = time.time()

    def unload(self):
        self._model = None
        self._model_key = None
        self._config = None
        _free_metal()

    @property
    def loaded(self):
        return self._model is not None

    # -------------------------------------------------------------- 出片
    def generate(self, prompt, image=None, width=1280, height=704,
                 num_frames=121, steps=40, seed=0, guide_scale=None,
                 scheduler="unipc", negative_prompt=None, tiling="conservative"):
        """出 一 条 片.

        prompt    : 文字
        image     : 首帧 图 的 路径 (I2V) —— None = 文生视频
        num_frames: 帧数, 必须 4n+1 (Wan 的 VAE 时间 压缩 是 4)
        tiling    : VAE 解码 分块 —— 实测 720P/121帧: conservative 172 秒,
                    default 243 秒, auto(默认) 272 秒, none 615 秒.
                    块 越大 越快 (少 很多 次 重叠 混合), 但 吃 内存.
        """
        import mlx_video.models.wan_2.generate as G
        import mlx_video.models.wan_2.utils as U

        self.load()

        # ① DiT 常驻 —— 截 住 load_wan_model, 第一次 装 完 后面 直接 拿
        cached = self._model
        orig_model = G.load_wan_model

        def _cached_model(path, config, quantization=None, loras=None):
            return cached

        G.load_wan_model = _cached_model

        # ①.5 DMD 蒸馏版 (FastWan2.2): 3 步 + 不用 CFG.
        #      上游 mlx-video 没有 DMD 采样器, 我们 自己 实现 一个,
        #      把 上游 的 FlowUniPCScheduler 换 掉 (接口 一模一样).
        recipe = self._dmd_recipe()
        orig_sched = None
        if recipe:
            from . import dmd as _dmd
            import mlx_video.models.wan_2.scheduler as SCH
            orig_sched = SCH.FlowUniPCScheduler
            _sigmas, _renoise = recipe

            def _dmd_factory(num_train_timesteps=1000):
                return _dmd.FlowDMDScheduler(
                    num_train_timesteps=num_train_timesteps,
                    sigmas=_sigmas, renoise=_renoise)

            SCH.FlowUniPCScheduler = _dmd_factory
            steps = len(_sigmas) - 1
            guide_scale = 1.0          # DMD 蒸馏 过 的 模型 不用 CFG (B=1, 2 倍快)
            negative_prompt = ""       # 不用 CFG 就 没有 负向 提示词

        # ② 截 帧 —— 上游 是 写 mp4, 我们 要 帧 数组
        cap = {}
        orig_save = G.save_video

        def _cap_save(frames, output_path, fps=24):
            cap["frames"] = frames
            cap["fps"] = fps

        G.save_video = _cap_save

        # ③ 本地 tokenizer —— 上游 硬 写 "google/umt5-xxl" (要联网),
        #    我们 装 的 时候 会 把 tokenizer.json 放 在 权重 目录 里.
        import transformers
        orig_tok = transformers.AutoTokenizer.from_pretrained
        local_tok_dir = self.model_dir
        has_local_tok = (local_tok_dir / "tokenizer.json").exists()

        def _tok(name, *a, **kw):
            if has_local_tok and name and "umt5" in str(name):
                return orig_tok(str(local_tok_dir), *a, **kw)
            return orig_tok(name, *a, **kw)

        transformers.AutoTokenizer.from_pretrained = _tok

        t0 = time.time()
        try:
            G.generate_video(
                model_dir=str(self.model_dir),
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                width=int(width),
                height=int(height),
                num_frames=int(num_frames),
                steps=int(steps),
                guide_scale=guide_scale,
                seed=int(seed),
                output_path="",
                scheduler=scheduler,
                tiling=tiling,
            )
        finally:
            G.load_wan_model = orig_model
            G.save_video = orig_save
            transformers.AutoTokenizer.from_pretrained = orig_tok
            if orig_sched is not None:
                import mlx_video.models.wan_2.scheduler as SCH
                SCH.FlowUniPCScheduler = orig_sched
            _free_metal()

        frames = cap.get("frames")
        if frames is None:
            raise RuntimeError("Wan2.2 出片 没 拿到 帧 (引擎 内部 出错 了)")
        fps = int(cap.get("fps") or 24)
        seq = int(frames.shape[0])
        return Wan2Result(
            video=frames, fps=fps,
            sequence_length=seq,
            seconds_per_step=round((time.time() - t0) / max(1, int(steps)), 2),
        )
