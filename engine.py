"""我们 自己 的 mlx_server 推理 引擎 — Apple MLX 本体 + 开源 port, 不 碰 闭源.
会 话 内 DiT(4bit 25G) 常 驻; 出 片 时 调 port 的 pipeline (临 时 载 TE/VAE), 出 完 卸.
峰 值 只 在 出 片 那 几 分 钟, 不 是 长 时 驻 留 87G."""
import os, sys, time, gc
# 内 核 一 份, 机 器 无 关 — 默 认 走 我 们 共 用 的 ~\.ours 布 局 (远端机 与 本 机 都 有 这 条 路).
# 本 机 开 发 覆 盖: H3M_PORT_DIR / H3M_MLX_ROOT / H3M_QUANT_DIR (指 到 本 机 权 重 路 径).
PORT = os.environ.get("H3M_PORT_DIR", os.path.expanduser("~/.ours/rea-ours/_vendor/minimax-h3_mlx_src"))
if PORT not in sys.path: sys.path.insert(0, PORT)
MLX_ROOT = os.environ.get("H3M_MLX_ROOT", os.path.expanduser("~/.ours/models/MiniMax/MiniMax-H3/FL2VA"))
QUANT_DIR = os.environ.get("H3M_QUANT_DIR", os.path.expanduser("~/.ours/quant/h3-mlx-4bit"))


def _model_id_from_quant(p):
    """量化 目录 名 -> 模型 卡 的 id (h3-mlx-4bit -> h3-4bit)."""
    base = os.path.basename(os.path.normpath(p)).lower()
    for pre in ("h3-mlx-", "h3_"):
        if base.startswith(pre):
            return "h3-" + base[len(pre):]
    return base


def _encode_scatter(te, prompt, images):
    """照 port 的 encode 流程 走 一遍, 但 视觉特征 用 masked_scatter 回填.

    port 原样 用 mx.where(image_mask[..., None], hidden[None], inputs_embeds) ——
    两个 形状 广播 不 了 (seq_len 含 文本 token, hidden 只有 图 token).
    mlx_vlm 自己 是 按 位置 填 (masked_scatter), 这里 照 抄.
    """
    import mlx.core as mx
    import numpy as np
    from mlx_vlm.models.qwen3_vl.language import LanguageModel

    input_ids, token_tags, vision_inputs = te.build_request(prompt, images)
    pixel_values, grid_np = vision_inputs
    grid_thw = mx.array(grid_np.astype(np.int32))
    hidden, deepstack_embeds = te.vision(
        mx.array(pixel_values).astype(te.dtype), grid_thw, output_hidden_states=True
    )
    inputs_embeds = te.language.embed_tokens(input_ids)
    image_mask = input_ids == te.image_token_id
    # 按 位置 填: 把 图 特征 塞 到 image token 所在 的 那些 位置
    # inputs_embeds: (1, seq_len, hidden) —— 按 行 (seq 位置) 填, 不 是 按 元素
    seq, hid = inputs_embeds.shape[1], inputs_embeds.shape[2]
    emb2d = inputs_embeds.reshape(seq, hid)
    rows = mx.array(np.where(np.asarray(image_mask).reshape(-1))[0], mx.uint32)
    emb2d[rows] = hidden.astype(inputs_embeds.dtype)
    inputs_embeds = emb2d.reshape(inputs_embeds.shape)

    position_ids, _ = LanguageModel.get_rope_index(
        te, input_ids, image_grid_thw=grid_thw, video_grid_thw=None, attention_mask=None
    )
    hidden_states = te._hidden_states(
        input_ids, position_ids,
        inputs_embeds=inputs_embeds,
        visual_pos_masks=image_mask,
        deepstack_visual_embeds=deepstack_embeds,
    )
    mx.eval(hidden_states)
    return hidden_states, token_tags


MODEL_ID = _model_id_from_quant(QUANT_DIR)

def _free():
    """释放 Metal 池 + 冷页.
    实测: del 模型引用 + gc.collect() 后 MLX Metal 池 active 立刻 回 到 0
    (605MB -> 8B), 所以 关 键 是 先 del 引 用 再 gc.
    mx.metal_flush 不 存 在 (AttributeError), 真 实 API 是 mx.clear_cache()
    (ml 4.x 起 从 mx.metal 迁 到 mx 顶 层, mx.metal.clear_cache 已 deprecated).
    顺序: del 引用 (调 用 方 做) -> gc.collect() -> mx.clear_cache() -> 再 gc.
    purge_inactive_memory 需 要 pyobjc Foundation, h3m3 venv 没 装, 改 不 着 就 跳 过
    (Metal 池 已 经 回 0, 系 统 级 purge 是 后 面 兜 底, 不 影 响 内 核 释 放)."""
    gc.collect()
    try:
        import mlx.core as mx
        # 新 版 mlx: mx.clear_cache(); 老 版: mx.metal.clear_cache()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    gc.collect()

class Engine:
    # 当前 装 的 是 哪 一 份 DiT 量化 权重
    model_id = MODEL_ID

    def __init__(self):
        self.pipe=None; self._last=0
        self._dit=None; self._te=None; self._vv=None; self._av=None
        self._te_has_vision=False
    def load(self):
        if self.pipe: return
        # 防反复 load 堆冷页: 先清干净旧引用再 load (上一轮 unload 没 flush 到 就 在 这 兜 底)
        for a in ("pipe","_dit","_te","_vv","_av"):
            setattr(self,a,None)
        gc.collect(); _free()
        from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
        from minimax_h3_mlx.load import load_dit, load_video_vae, load_audio_vae
        from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder
        from minimax_h3_mlx.config import PipelineConfig
        from pathlib import Path
        root=Path(MLX_ROOT); q=QUANT_DIR
        # DiT 走本地 quant (含 quant_config.json), TE/VAE 走 FL2VA 全精模型
        dit=load_dit(q, verbose=False)
        # 视觉塔 (Qwen3-VL 的 vision) —— 首尾帧 必需: 文字编码器 要 吃 图 才 能 编码 关键帧.
        # 纯文本 出片 不 需要, 所以 先 不 装 (省 ~1.5G 内存 + 加载 时间); 带 图 时 再 补.
        te=MiniMaxH3TextEncoder(root/"text_encoder", load_vision=False)
        vv=load_video_vae(root/"video_vae"); av=load_audio_vae(root/"audio_vae")
        cfg=PipelineConfig.from_model_index(root/"model_index.json")
        p=MiniMaxH3Pipeline(dit,te,vv,av,cfg)
        # 全量 load 完, 让 Metal buffer 立即可被 flush 掉 (Metal shared memory)
        gc.collect()
        # 显式 持 有 子 组 件 引用, 卸 时 逐 个 del, 防 pipeline 内 还 持 有 隐 式 参 照
        self._dit=dit; self._te=te; self._vv=vv; self._av=av
        self._te_has_vision=False
        self.pipe=p; self._last=time.time()

    def _ensure_vision(self):
        """首尾帧 路径 —— 补装 视觉塔 (Qwen3-VL vision).

        编码器 是 用 load_vision=False 建 的 (纯文本 出片 够 用, 省 内存).
        首/尾帧 要 让 文字编码器 看 图, 那 就 得 把 vision 部分 装 上.
        同一个 checkpoint 里 就有 (model.visual.*), 所以 重 载 一次 权重 即可.

        ⚠ 还要 修 一个 权重 布局 的 坑:
          checkpoint (torch 布局) 的 Conv3d 权重 是 (O, C_in, kD, kH, kW)
          MLX 的 Conv3d 要         (O, kD, kH, kW, C_in)   —— 通道 在 最后
          port 的 _load_weights 直接 搬 权重, 没 转 —— 于是 首尾帧 一跑 就
          "input and weight array to match" 报错. 语言塔 的 Linear 是 (out, in),
          两边 一致, 所以 纯文本 出片 看 不 出 问题.
        """
        if self._te_has_vision:
            return
        from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder
        from pathlib import Path
        root=Path(MLX_ROOT)
        te=MiniMaxH3TextEncoder(root/"text_encoder", load_vision=True)
        self._fix_conv_weights(te)
        self._fix_embed_scatter(te)
        self._te=te
        self.pipe.text_encoder=te
        self._te_has_vision=True
        gc.collect(); _free()

    @staticmethod
    def _fix_embed_scatter(te):
        """修 视觉特征 回填 的 广播 bug (port 里 用 mx.where, 但 形状 对 不 上).

        port 原样 是:
            inputs_embeds = mx.where(image_mask[..., None], hidden[None], inputs_embeds)
        image_mask 是 (1, seq_len) -> [..., None] 是 (1, seq_len, 1)
        hidden 是 (n_img_token, hidden) -> [None] 是 (1, n_img_token, hidden)
        两个 形状 广播 不 了 —— seq_len (含 文本 token) 跟 n_img_token (只 图 token) 不等.

        mlx_vlm 自己 是 用 masked_scatter: 把 图 特征 按 位置 填 进 文本 embedding,
        不 参与 广播. 这里 照 同样 的 做法 自己 走 一遍.
        """
        import mlx.core as mx
        import numpy as np
        if getattr(te, "_ours_scatter_fixed", False):
            return
        orig_encode = te.encode

        def encode(prompt, images=None, *a, **kw):
            if not images:
                return orig_encode(prompt, images, *a, **kw)
            try:
                return orig_encode(prompt, images, *a, **kw)
            except Exception as e:
                if "broadcast_shapes" not in str(e):
                    raise
                return _encode_scatter(te, prompt, images)

        te.encode = encode
        te._ours_scatter_fixed = True

    @staticmethod
    def _fix_conv_weights(te):
        """把 视觉塔 里 Conv3d 的 权重 从 torch 布局 转 成 MLX 布局.

        checkpoint: (O, C_in, kD, kH, kW)  →  MLX: (O, kD, kH, kW, C_in)
        """
        try:
            vision = getattr(te, "vision", None)
            if vision is None:
                return
            pe = getattr(vision, "patch_embed", None)
            proj = getattr(pe, "proj", None) if pe else None
            if proj is None:
                return
            w = proj.weight
            if w is not None and w.ndim == 5 and w.shape[1] == 3:
                # (O, C, kD, kH, kW) -> (O, kD, kH, kW, C)
                import mlx.core as mx
                new = mx.transpose(w, (0, 2, 3, 4, 1))
                proj.weight = new
                mx.eval(new)
                print("[engine] 视觉塔 Conv3d 权重 已 转成 MLX 布局 %s -> %s"
                      % (w.shape, new.shape), file=sys.stderr, flush=True)
        except Exception as e:
            print("[engine] 视觉塔 权重 布局 修正 失败: %s" % e, file=sys.stderr, flush=True)
    def unload(self):
        if self.pipe or self._has():
            for a in ("pipe","_dit","_te","_vv","_av"):
                setattr(self,a,None)
            gc.collect(); _free()
    def _has(self):
        return any(getattr(self,a,None) is not None for a in ("pipe","_dit","_te","_vv","_av"))
    def generate(self, prompt, seconds=5.0, steps=8, seed=0, height=None, width=None,
                 images=None, keyframe_anchors=(), ref_images=None):
        """出片.

        height/width     : 画布 覆盖 (H3 只 在 768 短边 发布, 别 的 尺寸 属于 离群)
        images           : 关键帧 图 (first / last) —— 闭源 契约 的
                           first_frame_image / last_frame_image
        keyframe_anchors : ("first",) / ("first", "last")
        ref_images       : 参考图 (只 进 文本塔) —— H3 官方 的 reference 模式,
                           提示词 里 用 <Picture N> 指代 (官方 文案: @图片N)
        """
        self.load()
        if images or ref_images:
            # 带 首尾帧 —— 必须 有 视觉塔, 不然 编码器 直接 抛 "cannot take images"
            self._ensure_vision()
        kw = {}
        if height:
            kw["height"] = int(height)
        if width:
            kw["width"] = int(width)
        if images:
            kw["images"] = images
            kw["keyframe_anchors"] = tuple(keyframe_anchors or ())
        if ref_images:
            kw["ref_images"] = list(ref_images)
        # drop_adaln=False —— AdaLN 投影 (13B 参数) **不要 删**.
        #
        # 上游 默认 True: 调制表 建 好 后 就 把 每 个 block 的 adaln_proj 删掉, 省 ~6.5G (4bit).
        # 但 那 是 **一次性** 操作 —— 权重 删了 就 永远 回不来.
        # 而 我们 的 出片 引擎 是 **常驻 服务**, 一条 请求 可能 连着 跑 好几次:
        #   · chain_windows 长片: 第 2 个 窗口 有 首帧 条件 → 调度表 变了 → 缓存 要 重建
        #     → 重建 要 调 adaln_proj → 权重 已经 被 删 → "QuantizedLinear has no attribute 'weight'"
        #   · 批量 出片 / 不同 长度 / 不同 步数: 同理
        # 一次 调度表 变了 就 崩, 所以 长片 这条路 从来 没 跑 通 过.
        #
        # 不 drop 的 代价: 4bit 下 多 占 ~6.5G 显存 (本机 128G / 远端机 96G 都 宽裕),
        # 换来 的 是 任何 次数 的 连续 出片 都 不 会 崩 —— 值.
        res = self.pipe(prompt, duration_seconds=seconds, num_inference_steps=steps,
                        seed=seed, drop_adaln=False, **kw)
        self._last = time.time()
        return res


# =====================================================================
# Wan2.2 TI2V-5B —— 第二个 出片 引擎 (开源, Apache-2.0).
# 和 H3 是 兄弟 关系, 不 是 替代:
#   H3    —— 支持 完整 首尾帧, 出片 慢 (33B DiT)
#   Wan2  —— 只有 首帧 (I2V), 出片 快 (5B DiT), 720P/24fps, 自带 静音
# 引擎 代码 = vendor 进来 的 mlx-video (MIT) + 我们 自己 的 port_wan2_mlx.
# =====================================================================
WAN2_DEFAULT_STEPS = 40          # Wan2.2 官方 40 步 (guide 5.0 / shift 5.0)
WAN2_FPS = 24
WAN2_MAX_AREA = 901120           # 1280x704 = 720P, 模型 的 上限
WAN2_FAST_DIR = os.path.expanduser("~/.ours/models/wan2-fastwan-q8")
WAN2_BASE_DIR = os.path.expanduser("~/.ours/models/wan2-ti2v-5b")


class Wan2Engine:
    """Wan2.2 TI2V-5B 引擎 —— 跟 Engine 一样 的 接口, server 不用 改 太多."""

    def __init__(self, model_dir=None, model_id="wan2-ti2v-5b"):
        self.model_id = model_id
        self.model_dir = model_dir or self._pick_dir()
        self.pipe = None
        self._last = 0

    @staticmethod
    def _pick_dir():
        """权重 目录 —— 优先 FastWan (DMD 3 步, 快 10 倍), 退而 求 次 原版.

        FastWan 是 同一个 DiT 的 DMD 蒸馏版 —— 实测 720P/5秒 = 3.4 分钟,
        原版 40 步 要 33 分钟 (比 H3 还 慢), 所以 优先 用 蒸馏版.
        """
        env = os.environ.get("WAN2_MLX_DIR")
        if env:
            return env
        for d in (WAN2_FAST_DIR, WAN2_BASE_DIR):
            if os.path.isfile(os.path.join(d, "model.safetensors")):
                return d
        return WAN2_FAST_DIR

    # ------------------------------------------------------------ 装 / 卸
    def load(self):
        if self.pipe is not None:
            return
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                "Wan2.2 权重 目录 不 在: %s —— 先 在 「模型」页 下载 这个 模型"
                % self.model_dir)
        from port_wan2_mlx.pipeline import Wan2Pipeline
        self.pipe = Wan2Pipeline(self.model_dir)
        self.pipe.load()
        self._last = time.time()

    def unload(self):
        if self.pipe is not None:
            self.pipe.unload()
        self.pipe = None
        gc.collect()
        _free()

    def _has(self):
        return self.pipe is not None

    # -------------------------------------------------------------- 出片
    def generate(self, prompt, seconds=5.0, steps=WAN2_DEFAULT_STEPS, seed=0,
                 height=None, width=None, images=None, keyframe_anchors=(),
                 ref_images=None):
        """出片 —— Wan2.2 是 文生视频 / 图生视频 (首帧) 二合一.

        只有 首帧: Wan 的 架构 没有 尾帧 条件 (那是 H3 的 活).
        尾帧 传 进来 -> 直接 报 清楚, 别 默默 丢掉.
        参考图 (ref_images): Wan2.2 没有 reference 模式, 只 能 把 图 当 首帧.
        """
        self.load()
        anchors = tuple(keyframe_anchors or ())
        if "last" in anchors:
            raise RuntimeError(
                "Wan2.2 只 支持 首帧 (图生视频), 没有 尾帧 条件 —— "
                "要 完整 首尾帧 请 选 H3")
        if ref_images:
            raise RuntimeError(
                "Wan2.2 没有 参考图 模式 (reference) —— 要 参考图 请 选 H3")

        # 帧数 必须是 4n+1 (Wan 的 VAE 时间 压缩 是 4)
        want = int(round(float(seconds) * WAN2_FPS))
        num_frames = max(1, (want // 4) * 4 + 1)
        if num_frames != want:
            # 5s -> 120 帧 不 合法, 抬 到 121 (5.04s)
            num_frames = ((want - 1) // 4) * 4 + 1

        # 画布: 默认 720P; 模型 上限 901120 像素 (1280x704)
        w = int(width or 1280)
        h = int(height or 704)
        if w * h > WAN2_MAX_AREA:
            scale = (WAN2_MAX_AREA / float(w * h)) ** 0.5
            w = int(w * scale) // 32 * 32
            h = int(h * scale) // 32 * 32

        img_path = None
        if images:
            from PIL import Image
            import tempfile, os as _os
            im = images[0]
            if not isinstance(im, Image.Image):
                im = Image.fromarray(im)
            fd, img_path = tempfile.mkstemp(suffix=".png")
            _os.close(fd)
            im.convert("RGB").save(img_path)

        # 步数: 界面 传 的 可能 是 H3 的 值 (8 步). 步数 由 权重 自己 决定 ——
        #   DMD 蒸馏版 = 3 步 (pipeline 内部 会 强制), 原版 Wan2.2 = 40 步.
        #   低于 12 步 一律 当 成 「没指定」, 按 权重 的 配方 走.
        steps = int(steps or 0)
        if steps < 12:
            steps = self.pipe.default_steps() if self.pipe else WAN2_DEFAULT_STEPS

        try:
            res = self.pipe.generate(
                prompt=prompt, image=img_path, width=w, height=h,
                num_frames=num_frames, steps=steps,
                seed=int(seed))
        finally:
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass
        self._last = time.time()
        return res
