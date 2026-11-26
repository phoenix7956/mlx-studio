"""我们自研内核 —— 闭源 MLX-Serve 的 完整功能面 (按 逆向 契约 全量落地).

对应 逆向 规格: ~/Desktop/rea/soak-results/mlxserve/mlx-serve_逆向规格.md
对应 原版 机制 表: ~/Documents/Craft Anything/mlx-serve/app/MECHANICS.md

本模块 只管 HTTP 面; 算力 走 srv.engines (llm / image / video / music).
没 接 的 能力 (TTS / 3D / sound / embeddings / decisions) 也 把 契约 做 全 ——
没 对应 模型 时 按 原版 语义 回 错 (点名 能力), 不 装 能.
"""
import base64
import json
import math
import os
import re
import sys
import time
import uuid
import outputs

# ---- 能力词表 (照 抄 原版 capability) ----
# reasoning_effort -> 思考 阶段 的 token 预算.
# 与 原版 语义 对齐: 思考 token 独立 于 max_tokens 之外, 正文 照样 拿 满 max_tokens.
REASONING_BUDGET = {"none": 0, "minimal": 256, "low": 512,
                    "medium": 1024, "high": 2048, "xhigh": 4096}
DEFAULT_REASONING_EFFORT = "medium"

CAPS = {
    "llm": ["chat", "streaming"],
    "image": ["image"],
    "music": ["music"],
    "video": ["video"],
    "embeddings": ["embeddings"],
    "tts": ["audio"],
    "voice": ["voice"],
    "sound": ["sound"],
    "3d": ["3d"],
    "decisions": ["decisions"],
    "stt": ["transcription"],
}

# 原版 默认: 请求 不给 max_tokens 时 的 serve 默认 (MECHANICS §13)
DEFAULT_MAX_TOKENS = 100


# ============================================================ 能力 引擎 取用
def cap_engine(srv, kind):
    """取 (或 建) 一个 能力 引擎. 轻量 / 无状态 的 直接 new, 重 的 走 srv.load()."""
    e = srv.engines.get(kind)
    if e is None:
        import cap_engines as C
        cls = {"embeddings": C.EmbedEngine, "tts": C.TTSEngine,
               "sound": C.SoundEngine, "music_synth": C.MusicSynthEngine,
               "3d": C.ThreeDEngine, "stt": C.ASREngine,
               "edits": C.ImageEditEngine,
               "decisions": C.DecisionEngine}.get(kind)
        if cls is None:
            raise KeyError("no engine for capability '%s'" % kind)
        e = cls()
        srv.engines[kind] = e
    # 取 用 就 记 时间 —— 不然 这 些 引擎 永远 没 进 过 置闲 计时, 挂 着 不 卸
    try:
        srv.touch(kind)
    except Exception:
        pass
    return e


# ============================================================ 神经 TTS (Qwen3-TTS)
NEURAL_TTS_IDS = {"qwen3-tts-base", "ours/qwen3-tts", "qwen3-tts",
                  "qwen3-tts-0.6b-base"}


def is_neural_tts_model(mid):
    """这个 model id 是不是 指向 神经 TTS (Qwen3-TTS)."""
    m = str(mid or "").strip().lower()
    if not m:
        return False
    if m in NEURAL_TTS_IDS:
        return True
    return "qwen3-tts" in m or "qwen3_tts" in m


def neural_tts_engine(srv):
    """取 (或 建) 神经 TTS 引擎. 权重 不 在 也 返回 对象 (available() 为 False)."""
    e = (getattr(srv, "engines", None) or {}).get("tts_neural")
    if e is None:
        try:
            import qwen3_tts_engine as Q
        except Exception:
            return None
        e = Q.Qwen3TTSEngine()
        if not hasattr(srv, "engines") or srv.engines is None:
            return None
        srv.engines["tts_neural"] = e
    # 注意: 这里 **不能** touch —— 本 函数 也 被 「列 音色」 接口 调 (前端 每次 刷新 音色 列表 都 调),
    # 一 旦 touch, 神经 TTS 的 计时 就 永远 被 刷新 -> 永远 不 会 被 自动 卸载.
    # 真正 用 到 它 的 地方 (加载 / 合成) 才 touch.
    return e


def neural_tts_probe():
    """只 探测 神经 TTS 的 静态 信息 (型号 / 音色 表 / 是否 可用) —— 给 「列 音色」 这 类
    **查询** 接口 用.
    关键: 这里 **不** 注册 引擎、**不** 刷新 计时 —— 否则 前端 每次 刷新 音色 列表,
    神经 TTS 就 被 「当成 在用」, 永远 不 会 被 自动 卸载.
    真正 合成 时 (audio_speech) 才 用 neural_tts_engine() 注册 并 touch."""
    try:
        import qwen3_tts_engine as Q
        return Q.Qwen3TTSEngine()
    except Exception:
        return None


def default_model_with_cap(srv, cap):
    """找 一个 带 该 能力 的 模型 (给 不 传 model 的 调用方 用).

    闭源 契约: 服务 端 没有 可用 模型 时, 生成 类 端点 一律 503 no_model
    ("No default model configured") —— 所以 这里 只 认 在位 的 模型,
    不 在位 的 往后 排, 全 都 不 在位 才 算 "没 模型".

    默认 优先 (catalog 里 标 了 default 的): 默认 出图 模型 是 MOSS-Image-SFT
    (Apache-2.0, 商用 没 门槛); 之前 只 按 目录 顺序 取 第一个, 拿到 的 是
    Krea-2-Turbo (社区许可, 有 $1M 营收 门槛) —— 不 对.
    这 只是 「请求 不 指定 model 时 用 谁」, 界面 上 不 给 谁 贴 「主力」 标签
    —— 用 哪个 模型 是 用户 自己 定 的.
    """
    ents = model_entries(srv)
    for e in ents:
        if cap in e["capabilities"] and e.get("default") and e.get("state") != "absent":
            return e["id"]
    for e in ents:
        if cap in e["capabilities"] and e.get("state") != "absent":
            return e["id"]
    for e in ents:
        if cap in e["capabilities"]:
            return e["id"]
    for e in ents:
        if cap in e["capabilities"] and e.get("default"):
            return e["id"]
    return ""


def _music_max_duration():
    """出歌 引擎 的 时长 真 上限 (秒) —— 界面 滑块 按 这个 画, 别 画 到 600 再 被 截断."""
    try:
        import music_mlx
        return float(music_mlx.max_duration())
    except Exception:
        return 480.0


def load_image_any(src):
    """file:// 路径 / data URL / base64 / bytes -> PIL.Image."""
    from PIL import Image
    import base64 as _b64
    import io as _io
    if isinstance(src, (bytes, bytearray)):
        return Image.open(_io.BytesIO(bytes(src)))
    s = str(src)
    if s.startswith("file://"):
        return Image.open(s[7:])
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    if os.path.isfile(s):
        return Image.open(s)
    raw = _b64.b64decode(s + "=" * (-len(s) % 4))
    return Image.open(_io.BytesIO(raw))


# ============================================================ 错误 / SSE 契约
def err(h, code, message, typ="invalid_request", param=None, extra=None):
    """原版 错误 契约: {"error":{"message":...,"type":...,"param":...,"code":...}} + 4xx/5xx."""
    # 闭源 契约: error 对象 里 param 一定 在 (没 指名 时 是 null), 不 是 省略.
    e = {"message": message, "type": typ, "param": param or None, "code": code}
    if extra:
        e.update(extra)
    h._j({"error": e}, code)


# ---- 参数 校验 (照 抄 闭源 客户端 的 约束) ----
class ParamError(ValueError):
    """带 param 名 的 参数 错误 (OpenAI 错误 契约: error.param 指名字段)."""
    def __init__(self, msg, param=None):
        super().__init__(msg)
        self.param = param


def num_param(b, key, default=None, lo=None, hi=None, integer=False):
    """从 body 取 数值 参数; 非法 / 越界 抛 ValueError (调用方 转 400)."""
    if key not in b or b.get(key) is None or b.get(key) == "":
        return default
    v = b.get(key)
    try:
        f = float(v)
    except Exception:
        raise ParamError("'%s' must be a number" % key, key)
    if not math.isfinite(f):
        raise ParamError("'%s' must be a finite number" % key, key)
    if integer and float(f) != int(f):
        raise ParamError("'%s' must be a whole number" % key, key)
    if lo is not None and f < lo:
        raise ParamError("'%s' must be between %s and %s" % (key, lo, hi), key)
    if hi is not None and f > hi:
        raise ParamError("'%s' must be between %s and %s" % (key, lo, hi), key)
    return int(f) if integer else f


def str_param(b, key, default=None, choices=None, lower=False):
    v = b.get(key)
    if v is None or v == "":
        return default
    if not isinstance(v, str):
        raise ParamError("'%s' must be a string" % key, key)
    if lower:
        v = v.lower()
    if choices and v not in choices:
        raise ParamError("'%s' must be one of: %s" % (key, ", ".join(choices)), key)
    return v


def b64_param(b, key):
    """取 base64 (或 data URL) 字段, 归一化成 纯 base64."""
    v = b.get(key)
    if not v:
        return None
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
        if not v:
            return None
    if not isinstance(v, str):
        return None
    m = re.match(r"^data:[^;]+;base64,(.*)$", v, re.S)
    if m:
        v = m.group(1)
    return v.strip() or None


def sse_start(h):
    # 注意: Connection 必须 是 close. 之前 写 keep-alive 会 覆盖 BaseHTTPRequestHandler
    # 的 默认 关连接 行为 -> 客户端 (urllib / fetch) 等 不到 EOF, 流式 请求 一直 挂着.
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.send_header("Cache-Control", "no-cache")
    h.send_header("Connection", "close")
    h.send_header("Access-Control-Allow-Origin", "*")
    h.end_headers()


def sse(h, obj):
    try:
        h.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
        h.wfile.flush()
    except Exception:
        pass


def sse_event(h, name, obj):
    """Responses API 的 SSE 形态: event: <type> + data: {...}."""
    try:
        h.wfile.write(("event: %s\ndata: %s\n\n"
                       % (name, json.dumps(obj, ensure_ascii=False))).encode())
        h.wfile.flush()
    except Exception:
        pass


def sse_done(h):
    try:
        h.wfile.write(b"data: [DONE]\n\n")
        h.wfile.flush()
    except Exception:
        pass


def send_wav(h, path):
    """裸 WAV bytes (原版 媒体 端点 的 裸 二进制 形态)."""
    try:
        size = os.path.getsize(path)
    except Exception:
        size = 0
    h.send_response(200)
    h.send_header("Content-Type", "audio/wav")
    h.send_header("Content-Length", str(size))
    h.end_headers()
    try:
        with open(path, "rb") as f:
            h.wfile.write(f.read())
    except Exception:
        pass


def out_url(h, path):
    """生成 产物 的 HTTP URL (我们 内核 会 把 /out/ 挂 出来).

    闭源 明确 说 "this server does not host generated files" —— 它 只 给 b64_json.
    我们 多 托管 一下, 所以 额外 给 一个 能 直接 加载 的 URL.
    原来 这里 返 "file://" + path, 但 网页 控制台 (http 页面) 根本 加载 不 了
    file:// —— 界面 上 会 是 一 张 空白. 这里 换成 同源 的 /out/<名字>.
    """
    # 统一 输出 目录 里 的 文件 带 子目录 (音频/xxx.mp3) —— /out/ 认 得
    name = outputs.rel(path) if path else ""
    try:
        host = h.headers.get("Host")
    except Exception:
        host = None
    if not host:
        try:
            host = "%s:%s" % h.server.server_address[:2]
        except Exception:
            host = "127.0.0.1"
    return "http://%s/out/%s" % (host, name)


def b64_file(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


# ============================================================ 模型清单 (原版 形状)
_LLM_META_CACHE = {}


def llm_model_meta():
    """LLM 权重 目录 的 架构 信息 (vocab_size / num_layers / quantization).

    闭源 /v1/models 的 meta 里 有 这 几 个 字段 (壳 的 模型 卡 会 读), 我们 从
    权重 目录 的 config.json 读 真 值 —— 读 一次 就 缓存.
    """
    if _LLM_META_CACHE:
        return _LLM_META_CACHE
    out = {"vocab_size": 0, "num_layers": 0, "architecture": "", "quantization": ""}
    try:
        import llm_engine as _le
        import os as _os
        import json as _json
        mp = _le.MODEL
        cfg = _json.load(open(_os.path.join(mp, "config.json")))
        out["vocab_size"] = int(cfg.get("vocab_size", 0) or 0)
        out["num_layers"] = int(cfg.get("num_hidden_layers", 0) or 0)
        arch = cfg.get("model_type") or ""
        if not arch:
            a = cfg.get("architectures") or [""]
            arch = a[0]
        out["architecture"] = arch
        for qf in ("quant_config.json", "config.json"):
            p = _os.path.join(mp, qf)
            if not _os.path.exists(p):
                continue
            qc = _json.load(open(p))
            g = qc.get("quantization") or qc.get("quantization_config") or {}
            if isinstance(g, dict):
                out["quantization"] = g.get("quant_method") or g.get("bits") or ""
            elif g:
                out["quantization"] = str(g)
            if out["quantization"]:
                break
    except Exception:
        pass
    _LLM_META_CACHE.update(out)
    return out


def model_entries(srv):
    """原版 /v1/models 的 data 行: id / object / capabilities / state / meta."""
    out = []
    # 引擎 -> 它 装 的 是 哪 一 份 权重. 必须 按 权重 算 loaded:
    # 按 "能力 引擎 在 不 在" 算 会 让 目录 里 所有 模型 都 显示 loaded
    # (视频 只 装 了 h3-4bit, 但 h3-8bit / h3-6bit 也 会 显示 loaded).
    loaded_ids = set()
    for _kind, _eng in (getattr(srv, "engines", None) or {}).items():
        if isinstance(_eng, dict):
            _mid = _eng.get("model_id")
        else:
            _mid = getattr(_eng, "model_id", None)
        # 有些 引擎 有 state() —— 它 说 的 是 权重 到底 在 不 在 显存 里.
        # 神经 TTS 的 引擎 对象 懒 创建 (权重 还没 装), 光 看 引擎 在 不 在 会 误报 loaded.
        _st = None
        if not isinstance(_eng, dict):
            try:
                _fn = getattr(_eng, "state", None)
                _s = _fn() if callable(_fn) else None
                if isinstance(_s, dict) and "loaded" in _s:
                    _st = bool(_s["loaded"])
            except Exception:
                _st = None
        if _mid and _st is not False:
            loaded_ids.add(str(_mid))
    for it in srv.models_resp().get("models", []):
        cap = it.get("cap", "llm")
        caps = it.get("caps") or CAPS.get(cap, ["chat"])
        mid = it.get("id")
        loaded = mid in loaded_ids
        present = bool(it.get("present"))
        # 出歌 (music) 是 独立 服务 (ACE-Step :7861) —— 它 一直 挂着 音乐 模型 权重 (~23G 显存).
        # 服务 在 跑 就 算 「已加载」, 模型页 那 一行 才 给 「卸载」 (卸载 = 把 服务 停掉, 释放 显存).
        if cap == "music":
            # 出歌 权重 真 挂 在 内存 里 没 —— 如实 报 (卸载 后 = False),
            # 免得 模型页 一直 显示 「已加载」, 用户 以为 还在 占 显存.
            # 只 有 「当前 真正 在 跑 的 那 一 个」 报 loaded —— 免得 三 行 全 显示 已加载.
            _meng=(getattr(srv,"engines",None) or {}).get("music")
            active_mid=(_meng.get("model_id") if isinstance(_meng,dict)
                        else getattr(_meng,"model_id",None))
            # 我们 自研 的 引擎 会 自动 起/停 —— 直接 问 它 现在 跑 的 是 哪个,
            # 别 拿 srv.engines 里 可能 过 时 的 登记.
            try:
                import music_workbench as _MW
                _st=_MW.our_mlx_status()
                if _st.get("running") and _st.get("model"):
                    active_mid=_st.get("model")
            except Exception:
                pass
            loaded = bool(it.get("loaded_hint"))
            if active_mid:
                loaded = loaded and str(active_mid)==str(mid)
            elif loaded:
                # 没 登记 活跃 引擎 (老 路径) —— ACE-Step :7861 在 跑 就 算 挂着
                loaded=False
                try:
                    import urllib.request as _ur
                    import os as _os
                    _ace = _os.environ.get("ACE_MUSIC_URL", "http://127.0.0.1:7861")
                    _ur.urlopen(_ace + "/health", timeout=4).read()
                    loaded = (str(mid)=="ace-step")
                except Exception:
                    pass
            present = True
        # 能不能 删 (磁盘 上 有 权重 文件夹) —— 前端 据此 决定 给 不给 「删除」 按钮
        _loc = os.path.expanduser(str(it.get("local", "") or ""))
        # 能 删 权重 (磁盘 上 有 文件夹) —— 出歌 模型 也 一样 (ACE-Step 7G /
        # Music3 8.6G / 闭源 弃用 那 份 4G 都 应该 能 删)
        deletable = bool(_loc) and os.path.isdir(_loc) and not it.get("builtin")
        lm = llm_model_meta() if cap == "llm" else {}
        out.append({
            "id": mid,
            "object": "model",
            # 「别名」/「已弃用」 的 条目 —— 主列表 里 折 起来 (还 在 折叠 区 里,
            # 能 加载 / 能 删 权重, 但 不 占 主视线)
            "hidden": bool(it.get("hidden")),
            "deprecated": bool(it.get("deprecated")),
            "alias_of": it.get("alias_of") or "",
            "object": "model",
            "created": 0,
            "owned_by": "mlx-serve",
            # 闭源 行 里 还有 alias (另 一 个 名字) 与 error (装载 失败 原因)
            "alias": it.get("alias", it.get("id")),
            "name": it.get("name", it.get("id")),
            "loaded": loaded,
            # 我们 多 一个 absent 态 (未 下 载); 闭源 只 有 ready/unloaded/remote
            "state": "loaded" if loaded else ("ready" if present else "absent"),
            "capabilities": caps,
            "input_modalities": ["text", "image"] if "vision" in caps else ["text"],
            "size": it.get("size", ""),
            "bytes_on_disk": it.get("bytes", 0),
            "bytes_resident": 0,
            "deletable": deletable,
            # 「下载」 —— 模型页 的 行 操作 只有 下载 + 删除 (加载/卸载 归 项目 页).
            # downloadable=False 的 模型 (h3-8bit 等) 没 有 公开 下载 源, 前端 就 不 给 下载 按钮.
            "downloadable": bool(it.get("downloadable")),
            "pull": it.get("pull") or "",
            "engine": it.get("engine", "our-server"),
            "architecture": it.get("arch") or it.get("cap"),
            "error": it.get("error", ""),
            # 出片 的 默认 参数 (步数/帧数/画布) —— catalog 里 写 的 是 真 值,
            # 前端 照 这 个 填 表, 界面 跟 引擎 永远 一致 (别 再 前端 写死 一 份).
            "steps": it.get("steps") or 0,
            "frames": it.get("frames") or 0,
            "width": it.get("width") or 0,
            "height": it.get("height") or 0,
            # 出图 模型 的 「上限配置 / 实测 / 许可」 —— 界面 上 直接 摆 一 张
            # 对比表 (Krea-32 步 vs Z-Image-9 步 vs MOSS-50 步), 不 让 用户 自个儿 猜.
            "max_steps": it.get("max_steps") or 0,
            "default_steps": it.get("default_steps") or 0,
            "bench_4mp_sec": it.get("bench_4mp_sec") or 0,
            "bench_note": it.get("bench_note") or "",
            "license": it.get("license") or "",
            # 许可 分级 —— 前端 模型页 照 这个 显示 徽章 + 「只看 自由许可」 筛选.
            #   license_free = True  -> Apache-2.0 / MIT / 本项目内置 (随便 用)
            #   license_free = False -> 社区许可 (H3 / Krea-2 / Gemma / Llama / IndexTTS2)
            "license_spdx": it.get("license_spdx") or "",
            "license_free": bool(it.get("license_free")),
            "license_url": it.get("license_url") or "",
            # 许可 条款 的 key —— 前端 点 徽章 就 用 它 去 /v1/licenses/<key> 取
            # 商用门槛 / 地区限制 / 署名要求 + 协议 原文.
            "license_key": _license_key(it),
            # 默认 模型 —— 请求 不 指定 model 时 用 它. 不 在 界面 上 标 「主力」:
            # 用 哪个 模型 是 用户 定 的.
            "default": bool(it.get("default")),
            "default_note": it.get("default_note") or "",
            # 该 模型 支持 的 最大 边长 (px) —— MOSS/Z-Image 能 到 3072, Krea-2 只 到 2048.
            # 界面 的 尺寸 输入框 上限 跟 API 的 校验 都 按 它 走.
            "max_size": int(it.get("max_size") or 2048),
            "max_size_note": it.get("max_size_note") or "",
            # 四 档 质量 各 对应 多少 步 —— 每个 模型 自己 的 步数 规矩 不一样
            # (Krea 4~32 步, Z-Image-Turbo 只 到 9 步, MOSS 要 50 步 + CFG).
            # 界面 的 质量 四 档 永远 可点, 各 档 落到 各 模型 自己 的 步数 上.
            "quality_steps": [int(x) for x in (it.get("quality_steps") or [])][:4],
            # 一 句 适合 什么 场景 —— 界面 上 给 用户 参考, 不 替 用户 做 决定.
            "fit_note": it.get("fit_note") or "",
            # 魔搭 repo —— 下载 优先 走 魔搭 (国内 最快); pull 是 HF, 兜底.
            "ms": it.get("ms") or "",
            "guidance": it.get("guidance") if it.get("guidance") is not None else 0,
            "meta": {
                "architecture": it.get("arch") or it.get("cap"),
                "engine": it.get("engine", "our-server"),
                "context_length": it.get("ctx") or getattr(srv, "ctx_size", 32768),
                "quantization": it.get("quant", ""),
                "model_type": it.get("arch") or it.get("cap"),
                # 闭源 meta 里 的 那 几 个 字段 (壳 的 模型 卡 会 读)
                "bytes_on_disk": it.get("bytes", 0),
                "vocab_size": it.get("vocab") or lm.get("vocab_size", 0),
                "num_layers": it.get("layers") or lm.get("num_layers", 0),
                "model_max_tokens": it.get("ctx") or getattr(srv, "ctx_size", 32768),
                "max_model_len": it.get("ctx") or getattr(srv, "ctx_size", 32768),
                "is_moe": bool(it.get("is_moe", False)),
                "mtp_available": bool(it.get("mtp", False)),
            },
        })
    return out


def find_model(srv, mid):
    for e in model_entries(srv):
        if e["id"] == mid:
            return e
    return None


def _license_key(it):
    """模型行 -> 许可 条款 的 key. 认 不出来 返回 ''."""
    try:
        import licenses as _lic
        return _lic.key_for(it) or ""
    except Exception:
        return ""


def think_budget_for(srv, mid, body, thinking=True):
    """决定 本次 请求 的 思考 预算 (0 = 不 给 思考 预算, 预算 全 给 正文).

    - reasoning_effort=none / enable_thinking=false -> 0 (关 思考)
    - 显式 给 了 reasoning_effort -> 按 档位 给
    - 没 指定 -> 只 给 声明 了 reasoning 能力 的 模型 开 (别的 模型 不 白 花 token)
    """
    if not thinking:
        return 0
    eff = body.get("reasoning_effort")
    if eff is None and isinstance(body.get("reasoning"), dict):
        eff = (body.get("reasoning") or {}).get("effort")
    if eff:
        return REASONING_BUDGET.get(str(eff).lower(),
                                   REASONING_BUDGET[DEFAULT_REASONING_EFFORT])
    e = find_model(srv, mid)
    if e and "reasoning" in (e.get("capabilities") or []):
        return REASONING_BUDGET[DEFAULT_REASONING_EFFORT]
    return 0


def resolve_model(srv, h, mid, cap, param="model"):
    """解析 生成 请求 的 模型 —— 闭源 契约.

    闭源 行为 (对拍 得到 的):
      - 没 指定 model -> 用 服务 的 默认 模型; 没有 默认 模型 -> 503 no_model
        ("No default model configured"), 且 这个 检查 在 body 校验 之前.
      - 指定 了 但 不 存在 -> 404
      - 存在 但 没 该 能力 -> 400 (点名 能力)
    返回 (model_id, entry); 任何 一个 是 None 都 表示 已经 回过 错.
    """
    mid = (mid or "").strip()
    if not mid:
        mid = default_model_with_cap(srv, cap)
        if not mid:
            err(h, 503, "No default model configured", typ="no_model")
            return None, None
    e = find_model(srv, mid)
    if e is None:
        err(h, 404, "model '%s' not found — call GET /v1/models for the list" % mid,
            param=param)
        return None, None
    caps = e.get("capabilities") or []
    if cap and cap not in caps:
        err(h, 400, "model '%s' does not have capability '%s' (has: %s)"
            % (mid, cap, ",".join(caps)), param=param)
        return None, None
    return mid, e


def need_cap(srv, mid, cap, h):
    """按 原版 语义: 没有 该 能力 的 模型 -> 点名 说清 (不 装 能)."""
    e = find_model(srv, mid)
    if e is None:
        err(h, 404, "model '%s' not found — call GET /v1/models for the list" % mid,
            typ="invalid_request", param="model")
        return None
    if cap not in e["capabilities"]:
        err(h, 400, "model '%s' does not have capability '%s' (has: %s)"
            % (mid, cap, ",".join(e["capabilities"])), param="model")
        return None
    return e


# ============================================================ 提示词 / 对话 辅助
def _content_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for x in c:
            if not isinstance(x, dict):
                continue
            if x.get("type") == "text":
                parts.append(x.get("text", ""))
            elif x.get("type") == "image_url":
                parts.append("[image]")
            elif x.get("type") == "thinking":
                parts.append("")
        return " ".join(p for p in parts if p)
    return ""


def messages_to_prompt(msgs):
    """没 模板 时 的 兜底 组装 (有 tokenizer 时 走 原生 chat template)."""
    sys_txt, turns = "", []
    for m in msgs or []:
        role = m.get("role", "user")
        txt = _content_text(m.get("content"))
        if role == "system":
            sys_txt += txt + "\n"
        else:
            turns.append("%s: %s" % (role, txt))
    return (sys_txt.strip() + "\n" if sys_txt else "") + "\n".join(turns)


def split_thinking(text):
    """Qwen3 类 模型 的 <think>...</think> 拆 成 reasoning_content + content.

    没 闭合 的 <think> (被 max_tokens 截断) 也 要 当 思考 处理, 别 把 思考 混 进 正文.
    """
    text = text or ""
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    if m:
        return m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()
    m2 = re.search(r"<think>(.*)$", text, re.S)
    if m2:
        return m2.group(1).strip(), ""
    if "</think>" in text:
        # 孤立 的 </think> (没 配对 的 开 标签): 只 当 标签 处理, 别 混 进 正文
        return "", text.replace("</think>", "").strip()
    return "", text


def split_thinking_raw(text):
    """不 strip 的 思考/正文 拆分 (流式 增量 用).

    strip 会 让 偏移 算 不 准 (前导 换行 一 出现 就 被 吃掉, 后面的 增量 就 错 位),
    所以 流式 必须 用 原文, 非流式 才 用 strip 过 的 split_thinking.
    """
    text = text or ""
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    if m:
        return m.group(1), (text[:m.start()] + text[m.end():])
    m2 = re.search(r"<think>(.*)$", text, re.S)
    if m2:
        return m2.group(1), ""
    if "</think>" in text:
        return "", text.replace("</think>", "")
    return "", text


def extract_tool_call(text, tools):
    """尽力 的 工具调用 识别: 模型 回 的 JSON 里 带 tool/arguments 就 认."""
    if not tools:
        return None
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("tool") or obj.get("name") or obj.get("function")
    if not name:
        return None
    args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    return {"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


# ============================================================ 对话 (OpenAI 形状)
def _record_gen_timing(srv, t0, first_at, ntok, plen):
    """把 生成 计时 喂 给 监控 卡 (decode tok/s / 首字延迟 / prefill tok/s)."""
    try:
        now = time.time()
        total = max(1e-6, now - t0)
        if first_at:
            ttft = (first_at - t0) * 1000.0
            dec = (ntok / max(1e-6, now - first_at)) if ntok else 0.0
            pre = (plen / 2.0) / max(1e-6, first_at - t0)
        else:
            ttft = 0.0
            dec = 0.0
            pre = 0.0
        srv.record_timing(ttft_ms=ttft, decode_tok_s=dec, prefill_tok_s=pre)
        with srv.metrics_lock:
            srv.generation_tokens += max(1, ntok)
    except Exception:
        pass


def build_chat_prompt(eng, msgs, tools=None, rf=None, thinking=True):
    """组 chat prompt: 优先 用 模型 自带 的 chat template, 再 叠 工具/JSON/思考 指令."""
    prompt = ""
    if eng is not None and hasattr(eng, "chat_prompt"):
        try:
            prompt = eng.chat_prompt(msgs, enable_thinking=thinking)
        except Exception:
            prompt = ""
    if not prompt:
        prompt = messages_to_prompt(msgs)
    if tools:
        norm = []
        for t in tools:
            if isinstance(t, dict) and t.get("function"):
                fn = t["function"]
                norm.append({"name": fn.get("name"), "parameters": fn.get("parameters")})
            else:
                norm.append({"name": t.get("name"), "input_schema": t.get("input_schema")})
        prompt += "\n\n可用工具(JSON): " + json.dumps(norm, ensure_ascii=False)
        prompt += ("\n\n如果 需要 调用 工具, 只 输出 一个 JSON 对象 (不要 任何 解释): "
                   "{\"tool\": \"<工具名>\", \"arguments\": {<参数>}}. "
                   "不用 工具 时 就 直接 回答 问题.")
    if rf:
        if rf.get("type") == "json_schema":
            prompt += "\n\n只输出符合这个 schema 的 JSON, 不要解释: " + json.dumps(
                rf.get("json_schema", {}), ensure_ascii=False)
        elif rf.get("type") == "json_object":
            prompt += "\n\n只输出 JSON, 不要解释."
    if not thinking:
        prompt += "\n\n(直接回答, 不要输出思考过程)"
    return prompt


def chat_completions(srv, h, b):
    msgs = b.get("messages")
    if not isinstance(msgs, list) or not msgs:
        err(h, 400, "missing 'messages' (array of {role, content})", param="messages")
        return
    model, _me = resolve_model(srv, h, b.get("model") or "", "chat")
    if model is None:
        return
    # max_tokens=0 是 非法 的 (不 是 "不 限制" —— 之前 会 静默 回落 默认, 害 人)
    for _k in ("max_tokens", "max_completion_tokens"):
        if _k in b and b.get(_k) is not None:
            try:
                _mv = int(b.get(_k))
            except Exception:
                err(h, 400, "'%s' must be a whole number" % _k, param=_k)
                return
            if _mv < 1:
                err(h, 400, "'%s' must be at least 1" % _k, param=_k)
                return
    max_tokens = int(b.get("max_tokens") or b.get("max_completion_tokens")
                     or getattr(srv, "max_tokens", DEFAULT_MAX_TOKENS))
    tools = b.get("tools") or []
    try:
        temp = num_param(b, "temperature", 0.6, 0.0, 2.0)
        top_p = num_param(b, "top_p", 0.9, 0.0, 1.0)
        top_k = num_param(b, "top_k", 0, 0, 1000, integer=True)
    except ValueError as e:
        err(h, 400, str(e), param=getattr(e, "param", None))
        return
    tool_choice = b.get("tool_choice")
    if tool_choice is not None and not isinstance(tool_choice, (str, dict)):
        err(h, 400, "'tool_choice' must be a string or an object", param="tool_choice")
        return
    if isinstance(tool_choice, str) and tool_choice not in ("none", "auto", "required"):
        names = [t.get("function", {}).get("name") for t in tools if isinstance(t, dict)]
        if tool_choice not in names:
            err(h, 400, "'tool_choice' names a tool that was not provided", param="tool_choice")
            return
    rf = b.get("response_format") or {}
    if tool_choice == "none":
        tools = []
    if tool_choice == "required" and tools:
        rf = dict(rf or {})
    thinking = not (b.get("reasoning_effort") == "none" or b.get("enable_thinking") is False)
    tb = think_budget_for(srv, model, b, thinking)
    try:
        srv.load("llm")
    except Exception as e:
        err(h, 503, "llm engine load failed: %s" % e, typ="server_error")
        return
    eng = srv.engines.get("llm")
    prompt = build_chat_prompt(eng, msgs, tools=tools, rf=rf, thinking=thinking)
    created = int(time.time())
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    if b.get("stream"):
        sse_start(h)
        full = ""
        t0 = time.time()
        first_at = 0
        ntok = 0
        sent_c = 0
        sent_r = 0
        try:
            for piece in eng.stream(prompt, max_tokens=max_tokens, think_budget=tb,
                                    temp=temp, top_p=top_p, top_k=top_k):
                if not first_at:
                    first_at = time.time()
                ntok += 1
                full += piece
                # 原版 契约: 思考 走 delta.reasoning_content, 正文 走 delta.content
                # 用 不 strip 的 拆分, 偏移 才 算得 准 (前导 换行 不能 被 吃掉)
                rsn, ctt = split_thinking_raw(full)
                if len(rsn) > sent_r:
                    sse(h, {"id": cid, "object": "chat.completion.chunk", "created": created,
                            "model": model,
                            "choices": [{"index": 0,
                                         "delta": {"reasoning_content": rsn[sent_r:]},
                                         "finish_reason": None}]})
                    sent_r = len(rsn)
                if len(ctt) > sent_c:
                    sse(h, {"id": cid, "object": "chat.completion.chunk", "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": ctt[sent_c:]},
                                         "finish_reason": None}]})
                    sent_c = len(ctt)
        except Exception as e:
            sse(h, {"error": {"message": str(e), "type": "server_error"}})
            sse_done(h)
            return
        fin = "length" if (getattr(eng, "last_info", {}) or {}).get("truncated") else "stop"
        sse(h, {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": fin}]})
        if (b.get("stream_options") or {}).get("include_usage"):
            sse(h, {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": model, "choices": [],
                    "usage": {"prompt_tokens": len(prompt) // 4,
                              "completion_tokens": len(full) // 4,
                              "total_tokens": (len(prompt) + len(full)) // 4}})
        sse_done(h)
        srv.touch("llm")
        _record_gen_timing(srv, t0, first_at, ntok, len(prompt))
        return
    t0 = time.time()
    try:
        raw = eng.generate(prompt, max_tokens=max_tokens, think_budget=tb,
                           temp=temp, top_p=top_p, top_k=top_k)
    except Exception as e:
        err(h, 503, "llm inference failed: %s" % e, typ="server_error")
        return
    srv.touch("llm")
    _record_gen_timing(srv, t0, 0, 0, len(prompt))
    reasoning, content = split_thinking(raw)
    truncated = bool((getattr(eng, "last_info", {}) or {}).get("truncated"))
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    tc = extract_tool_call(content, tools)
    if tc:
        msg["content"] = content.replace(tc["function"]["arguments"], "").strip() or None
        msg["tool_calls"] = [tc]
    h._j({"id": cid, "object": "chat.completion", "created": created, "model": model,
          "choices": [{"index": 0, "message": msg,
                       "finish_reason": ("tool_calls" if tc else
                                         ("length" if truncated else "stop"))}],
          "usage": {"prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(content) // 4,
                    "total_tokens": (len(prompt) + len(content)) // 4}})


def completions(srv, h, b):
    """原版 /v1/completions: prompt 字符串 或 数组."""
    p = b.get("prompt")
    if p is None:
        err(h, 400, "missing 'prompt' (string or array of strings)", param="prompt")
        return
    prompts = p if isinstance(p, list) else [p]
    n = max(1, int(b.get("n") or 1))
    max_tokens = int(b.get("max_tokens") or DEFAULT_MAX_TOKENS)
    model, _me = resolve_model(srv, h, b.get("model") or "", "chat")
    if model is None:
        return
    srv.load("llm")
    eng = srv.engines.get("llm")
    created = int(time.time())
    choices = []
    for i in range(n):
        txt = eng.generate("\n".join(str(x) for x in prompts), max_tokens=max_tokens)
        choices.append({"index": i, "text": txt, "finish_reason": "stop", "logprobs": None})
    srv.touch("llm")
    h._j({"id": "cmpl-" + uuid.uuid4().hex[:24], "object": "text_completion",
          "created": created, "model": model, "choices": choices,
          "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})


def embeddings(srv, h, b):
    """原版 /v1/embeddings: 真 语义 向量 (Qwen3-Embedding-0.6B)."""
    inp = b.get("input")
    if inp is None:
        err(h, 400, "missing 'input' (string or array of strings)", param="input")
        return
    model = b.get("model") or ""
    if not model:
        model = default_model_with_cap(srv, "embeddings")
    e = find_model(srv, model)
    if e is None:
        if not (b.get("model") or ""):
            err(h, 503, "No default model configured", typ="no_model")
            return
        err(h, 404, "model '%s' not found — call GET /v1/models for the list" % model,
            param="model")
        return
    if "embeddings" not in e["capabilities"]:
        err(h, 400, "model '%s' does not have capability 'embeddings' (has: %s)"
            % (model, ",".join(e["capabilities"])), param="model")
        return
    texts = inp if isinstance(inp, list) else [inp]
    texts = [t if isinstance(t, str) else json.dumps(t, ensure_ascii=False) for t in texts]
    if not texts:
        err(h, 400, "'input' is empty", param="input")
        return
    try:
        dim = int(b.get("dimensions")) if b.get("dimensions") else None
    except Exception:
        dim = None
    try:
        eng = cap_engine(srv, "embeddings")
        vecs = eng.embed(texts, dim=dim)
    except Exception as ex:
        err(h, 503, "embedding engine failed: %s" % ex, typ="server_error")
        return
    data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]
    ntok = sum(len(t) // 3 + 1 for t in texts)
    h._j({"object": "list", "data": data, "model": model,
          "usage": {"prompt_tokens": ntok, "total_tokens": ntok}})


def tokenize(srv, h, b):
    txt = b.get("text") or b.get("prompt") or ""
    if not txt:
        err(h, 400, "missing 'text'", param="text")
        return
    srv.load("llm")
    eng = srv.engines.get("llm")
    ids = eng.tokenize(txt) if hasattr(eng, "tokenize") else []
    h._j({"tokens": ids, "count": len(ids)})


def detokenize(srv, h, b):
    ids = b.get("tokens") or []
    if not isinstance(ids, list):
        err(h, 400, "'tokens' must be an array of token ids", param="tokens")
        return
    srv.load("llm")
    eng = srv.engines.get("llm")
    txt = eng.detokenize(ids) if hasattr(eng, "detokenize") else ""
    h._j({"text": txt})


# ============================================================ Anthropic 形状
def anthropic_messages(srv, h, b):
    msgs = b.get("messages")
    if not isinstance(msgs, list) or not msgs:
        err(h, 400, "missing 'messages' (array of {role, content})", param="messages")
        return
    model, _me = resolve_model(srv, h, b.get("model") or "", "chat")
    if model is None:
        return
    max_tokens = int(b.get("max_tokens") or DEFAULT_MAX_TOKENS)
    system = b.get("system")
    norm = []
    if system:
        norm.append({"role": "system",
                     "content": system if isinstance(system, str) else _content_text(system)})
    norm.extend(msgs)
    tools = b.get("tools") or []
    try:
        srv.load("llm")
    except Exception as e:
        err(h, 503, "llm engine load failed: %s" % e, typ="server_error")
        return
    eng = srv.engines.get("llm")
    tb = think_budget_for(srv, model, b)
    # 思考 关闭 时 (reasoning_effort=none / enable_thinking=false), 模板 也 要 关
    prompt = build_chat_prompt(eng, norm, tools=tools, thinking=(tb > 0))
    mid = "msg_" + uuid.uuid4().hex[:24]
    if b.get("stream"):
        sse_start(h)
        sse(h, {"type": "message_start",
                "message": {"id": mid, "type": "message", "role": "assistant",
                            "model": model, "content": [], "stop_reason": None,
                            "usage": {"input_tokens": len(prompt) // 4, "output_tokens": 0}}})
        out = ""
        sent_c = 0
        sent_r = 0
        idx_t = -1        # thinking 块 的 index
        idx_x = -1        # text 块 的 index
        try:
            for piece in eng.stream(prompt, max_tokens=max_tokens, think_budget=tb):
                out += piece
                # Anthropic 形状: 思考 走 thinking_delta, 正文 走 text_delta
                rsn, ctt = split_thinking_raw(out)
                if rsn and idx_t < 0:
                    idx_t = 0
                    sse(h, {"type": "content_block_start", "index": idx_t,
                            "content_block": {"type": "thinking", "thinking": ""}})
                if ctt and idx_x < 0:
                    idx_x = idx_t + 1 if idx_t >= 0 else 0
                    sse(h, {"type": "content_block_start", "index": idx_x,
                            "content_block": {"type": "text", "text": ""}})
                if rsn and len(rsn) > sent_r:
                    sse(h, {"type": "content_block_delta", "index": idx_t,
                            "delta": {"type": "thinking_delta", "thinking": rsn[sent_r:]}})
                    sent_r = len(rsn)
                if ctt and len(ctt) > sent_c:
                    sse(h, {"type": "content_block_delta", "index": idx_x,
                            "delta": {"type": "text_delta", "text": ctt[sent_c:]}})
                    sent_c = len(ctt)
        except Exception as e:
            sse(h, {"type": "error", "error": {"type": "server_error", "message": str(e)}})
            return
        if idx_t >= 0:
            sse(h, {"type": "content_block_stop", "index": idx_t})
        if idx_x >= 0:
            sse(h, {"type": "content_block_stop", "index": idx_x})
        trunc = bool((getattr(eng, "last_info", {}) or {}).get("truncated"))
        sse(h, {"type": "message_delta",
                "delta": {"stop_reason": "max_tokens" if trunc else "end_turn",
                          "stop_sequence": None},
                "usage": {"output_tokens": len(out) // 4}})
        sse(h, {"type": "message_stop"})
        srv.touch("llm")
        return
    try:
        raw = eng.generate(prompt, max_tokens=max_tokens, think_budget=tb)
    except Exception as e:
        err(h, 503, "llm inference failed: %s" % e, typ="server_error")
        return
    srv.touch("llm")
    reasoning, content = split_thinking(raw)
    trunc = bool((getattr(eng, "last_info", {}) or {}).get("truncated"))
    blocks = [{"type": "text", "text": content}]
    if reasoning:
        # 原版 把 思考 也 带 出来; text 块 仍 放 第一个 (跟 闭源 壳 的 读法 一致)
        blocks.append({"type": "thinking", "thinking": reasoning})
    h._j({"id": mid, "type": "message", "role": "assistant", "model": model,
          "content": blocks,
          "stop_reason": "max_tokens" if trunc else "end_turn", "stop_sequence": None,
          "usage": {"input_tokens": len(prompt) // 4, "output_tokens": len(content) // 4}})


# ============================================================ OpenAI Responses 形状
def responses(srv, h, b):
    prep = responses_prepare(srv, b)
    if not prep.get("ok"):
        err(h, prep.get("code", 400), prep.get("error", "bad request"),
            param=prep.get("param"))
        return
    msgs = prep["msgs"]
    model = prep["model"]
    max_tokens = prep["max_tokens"]
    tb = prep["tb"]
    eng = prep["eng"]
    if b.get("stream"):
        responses_stream(srv, h, eng, model, msgs, b, max_tokens, tb)
        return
    rid = "resp_" + uuid.uuid4().hex[:24]
    prompt = build_chat_prompt(eng, msgs, thinking=(tb > 0))
    try:
        raw = eng.generate(prompt, max_tokens=max_tokens, think_budget=tb)
    except Exception as e:
        err(h, 503, "llm inference failed: %s" % e, typ="server_error")
        return
    srv.touch("llm")
    reasoning, content = split_thinking(raw)
    trunc = bool((getattr(eng, "last_info", {}) or {}).get("truncated"))
    body = {
        "id": rid, "object": "response", "created_at": int(time.time()), "model": model,
        "status": "incomplete" if trunc else "completed",
        "output": [{"id": "msg_" + uuid.uuid4().hex[:12], "type": "message",
                    "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": content, "annotations": []}]}],
        "usage": {"input_tokens": len(prompt) // 4, "output_tokens": len(content) // 4,
                  "total_tokens": (len(prompt) + len(content)) // 4},
    }
    srv.store_response(rid, body)
    h._j(body)


def responses_prepare(srv, b):
    """解析 /v1/responses 请求: -> {ok,msgs,model,max_tokens,tb,eng} 或 {ok:False,...}.

    HTTP (SSE) 与 WebSocket 两条 通路 共用.
    """
    inp = b.get("input")
    if inp is None:
        return {"ok": False, "code": 400, "error": "missing 'input'", "param": "input"}
    if isinstance(inp, str):
        msgs = [{"role": "user", "content": inp}]
    elif isinstance(inp, list):
        msgs = []
        for it in inp:
            if isinstance(it, dict) and it.get("type") in (None, "message"):
                msgs.append({"role": it.get("role", "user"), "content": it.get("content", "")})
            elif isinstance(it, str):
                msgs.append({"role": "user", "content": it})
    else:
        return {"ok": False, "code": 400, "error": "'input' must be a string or an array",
                "param": "input"}
    if b.get("instructions"):
        msgs.insert(0, {"role": "system", "content": b["instructions"]})
    model = b.get("model") or ""
    e = find_model(srv, model)
    if e is None:
        if not (b.get("model") or ""):
            return {"ok": False, "code": 503,
                    "error": "No default model configured", "type": "no_model"}
        return {"ok": False, "code": 404,
                "error": "model '%s' not found — call GET /v1/models for the list" % model,
                "param": "model"}
    if "chat" not in (e.get("capabilities") or []):
        return {"ok": False, "code": 400,
                "error": "model '%s' does not have capability 'chat' (has: %s)"
                         % (model, ",".join(e.get("capabilities") or [])),
                "param": "model"}
    max_tokens = int(b.get("max_output_tokens") or DEFAULT_MAX_TOKENS)
    tb = think_budget_for(srv, model, b)
    try:
        srv.load("llm")
    except Exception as ex:
        return {"ok": False, "code": 503, "error": "llm engine load failed: %s" % ex}
    eng = srv.engines.get("llm")
    if eng is None:
        return {"ok": False, "code": 503, "error": "llm engine not available"}
    return {"ok": True, "msgs": msgs, "model": model, "max_tokens": max_tokens,
            "tb": tb, "eng": eng}


def responses_get(srv, h, rid):
    body = srv.get_response(rid)
    if body is None:
        err(h, 404, "response '%s' not found" % rid)
        return
    h._j(body)


def responses_delete(srv, h, rid):
    ok = srv.delete_response(rid)
    h._j({"id": rid, "object": "response.deleted", "deleted": bool(ok)})


def _response_text(body):
    """从 一条 response body 里 抽出 正文 (兼容 多种 形状)."""
    try:
        return body["output"][0]["content"][0]["text"] or ""
    except Exception:
        pass
    try:
        return body["output_text"] or ""
    except Exception:
        return ""


def responses_compact(srv, h, b):
    """原版 /v1/responses/compact: 把 一条 长 response 压 短 (摘要 + 关键结论).

    闭源 契约 (对拍 得到 的):
      - 没 给 model            -> 422 "model is required" (type=invalid_request_error)
      - 给了 model 没 给 input -> 400 "'input' is a required field"
      - 成功 -> {"id":"comp_<ts>_<n>","object":"response.compaction",
                 "output":[{"type":"compaction","id":"cmp_<ts>_<n>","encrypted_content":"<b64>"}],
                 "usage":{...}}
    我们 比 闭源 多 做 一 件 事: 真 用 模型 压 摘要, 结果 放 在 "summary" 字段
    (闭源 只 把 消息 打包 进 encrypted_content, usage 全 0 —— 摘要 在 客户端 做).
    """
    model = (b.get("model") or "").strip()
    if not model:
        err(h, 422, "model is required", typ="invalid_request_error")
        return
    rid = b.get("response_id") or b.get("id")
    body = srv.get_response(rid) if rid else None
    if body is None:
        body = b.get("response") or b
    inp = b.get("input")
    if inp is None and isinstance(body, dict):
        inp = body.get("input")
    if inp is None and not rid:
        err(h, 400, "'input' is a required field", typ="invalid_request_error")
        return
    text = _response_text(body) if rid else _input_text(inp)
    if not text:
        err(h, 400, "'input' is a required field", typ="invalid_request_error")
        return
    summary = ""
    engine_used = "truncate"
    try:
        srv.load("llm")
        eng = srv.engines.get("llm")
        if eng is not None:
            # 走 chat 模板 + 关 思考 + 贪心 解码:
            #   裸 prompt + 默认 采样 (temp 0.6) 会 让 小模型 自由发挥, 摘要 里 冒出
            #   原文 根本 没有 的 "Nature 论文" 之类 幻觉. 摘要 这 类 任务 必须 忠实.
            msgs = [
                {"role": "system",
                 "content": ("你 是一个 忠实 的 摘要器. 把 用户 给 的 文本 压缩 成 一段 摘要.\n"
                             "规则:\n"
                             "1. 只 概括 原文 已有 的 内容, 绝不 补充 原文 没有 的 事实、"
                             "数字、日期、文献 或 人名.\n"
                             "2. 保留 关键 结论 与 数字, 去掉 寒暄 与 重复.\n"
                             "3. 只 输出 摘要 本身, 不 输出 任何 前缀、解释 或 标题."),
                 },
                {"role": "user", "content": text[:8000]},
            ]
            prompt = build_chat_prompt(eng, msgs, thinking=False)
            out = eng.generate(prompt, max_tokens=512, think_budget=0, temp=0.0)
            _th, out = split_thinking(out or "")
            summary = (out or "").strip()
            # 模型 有 时 会 给 摘要 加 上 "摘要:" / "好的," 之类 的 壳 —— 去掉
            for pre in ("摘要：", "摘要:", "摘要 :", "好的，", "好的,", "以下是摘要：", "以下是摘要:"):
                if summary.startswith(pre):
                    summary = summary[len(pre):].lstrip()
            if summary:
                engine_used = "llm-summarize"
            else:
                summary = ""
    except Exception:
        summary = ""
    if not summary:
        summary = text[:400] + ("…" if len(text) > 400 else "")
        engine_used = "truncate"
    # 闭源 形状: response.compaction + output[].encrypted_content
    msgs = _input_msgs(inp) if inp is not None else []
    if not msgs:
        msgs = [{"role": "user", "content": text}]
    try:
        blob = base64.b64encode(json.dumps(
            {"v": 1, "msgs": msgs}, ensure_ascii=False).encode("utf-8")).decode()
    except Exception:
        blob = ""
    ts = int(time.time())
    n = uuid.uuid4().hex[:8]
    h._j({"id": "comp_%d_%s" % (ts, n), "object": "response.compaction",
          "created_at": ts,
          "output": [{"type": "compaction", "id": "cmp_%d_%s" % (ts, n),
                      "encrypted_content": blob,
                      # —— 闭源 只 打包 消息 (摘要 在 客户端 做); 我们 多 给 一 份 真 摘要
                      "summary": summary,
                      "engine": engine_used}],
          "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0}}})


# ============================================================ Ollama 形状
def ollama_chat(srv, h, b):
    msgs = b.get("messages") or []
    model, _me = resolve_model(srv, h, b.get("model") or "", "chat")
    if model is None:
        return
    srv.load("llm")
    eng = srv.engines.get("llm")
    tb = think_budget_for(srv, model, b)
    prompt = build_chat_prompt(eng, msgs, thinking=(tb > 0))
    created = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())
    if b.get("stream", True):
        h.send_response(200)
        h.send_header("Content-Type", "application/x-ndjson")
        h.end_headers()
        out = ""
        sent_c = 0
        for piece in eng.stream(prompt, max_tokens=DEFAULT_MAX_TOKENS, think_budget=tb):
            out += piece
            # Ollama 形状 没有 思考 概念 -> 只 吐 正文
            _r, ctt = split_thinking_raw(out)
            if len(ctt) > sent_c:
                h.wfile.write((json.dumps({"model": model, "created_at": created,
                                           "message": {"role": "assistant", "content": ctt[sent_c:]},
                                           "done": False}) + "\n").encode())
                h.wfile.flush()
                sent_c = len(ctt)
        h.wfile.write((json.dumps({"model": model, "created_at": created,
                                   "message": {"role": "assistant", "content": ""},
                                   "done": True, "total_duration": 0,
                                   "prompt_eval_count": 0,
                                   "eval_count": len(out) // 4}) + "\n").encode())
        srv.touch("llm")
        return
    txt = eng.generate(prompt, max_tokens=DEFAULT_MAX_TOKENS, think_budget=tb)
    srv.touch("llm")
    _r, txt = split_thinking(txt)
    h._j({"model": model, "created_at": created,
          "message": {"role": "assistant", "content": txt}, "done": True})


def ollama_generate(srv, h, b):
    prompt = b.get("prompt") or ""
    model, _me = resolve_model(srv, h, b.get("model") or "", "chat")
    if model is None:
        return
    srv.load("llm")
    eng = srv.engines.get("llm")
    tb = think_budget_for(srv, model, b)
    created = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())
    if b.get("stream", True):
        h.send_response(200)
        h.send_header("Content-Type", "application/x-ndjson")
        h.end_headers()
        out = ""
        sent_c = 0
        for piece in eng.stream(prompt, max_tokens=DEFAULT_MAX_TOKENS, think_budget=tb):
            out += piece
            _r, ctt = split_thinking_raw(out)
            if len(ctt) > sent_c:
                h.wfile.write((json.dumps({"model": model, "created_at": created,
                                           "response": ctt[sent_c:], "done": False}) + "\n").encode())
                h.wfile.flush()
                sent_c = len(ctt)
        h.wfile.write((json.dumps({"model": model, "created_at": created, "response": "",
                                   "done": True, "eval_count": len(out) // 4}) + "\n").encode())
        srv.touch("llm")
        return
    txt = eng.generate(prompt, max_tokens=DEFAULT_MAX_TOKENS, think_budget=tb)
    srv.touch("llm")
    _r, txt = split_thinking(txt)
    h._j({"model": model, "created_at": created, "response": txt, "done": True})


def ollama_embed(srv, h, b):
    """Ollama /api/embed (老 的 /api/embeddings 也 兼容)."""
    inp = b.get("input")
    if inp is None:
        inp = b.get("prompt")
    if inp is None:
        err(h, 400, "missing 'input' (or 'prompt')", param="input")
        return
    texts = inp if isinstance(inp, list) else [inp]
    texts = [t if isinstance(t, str) else json.dumps(t, ensure_ascii=False) for t in texts]
    model = b.get("model") or default_model_with_cap(srv, "embeddings")
    try:
        eng = cap_engine(srv, "embeddings")
        vecs = eng.embed(texts)
    except Exception as ex:
        err(h, 503, "embedding engine failed: %s" % ex, typ="server_error")
        return
    legacy = h.path.rstrip("/").endswith("/api/embeddings")
    if legacy:
        h._j({"embedding": vecs[0] if vecs else []})
    else:
        h._j({"model": model, "embeddings": vecs,
              "prompt_eval_count": sum(len(t) // 3 + 1 for t in texts),
              "total_duration": 0, "load_duration": 0})


def audio_transcriptions(srv, h, b):
    """POST /v1/audio/transcriptions: multipart 文件 -> 文本 (faster-whisper)."""
    files = b.get("__files__") or {}
    data = None
    name = None
    for k, v in files.items():
        if k in ("file", "audio", "audio_file") or k.startswith("file"):
            data = v.get("data")
            name = v.get("filename")
            break
    if data is None:
        err(h, 400, "missing 'file' (multipart upload of an audio file)", param="file")
        return
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    try:
        eng = cap_engine(srv, "stt")
        r = eng.transcribe(path, language=b.get("language") or None)
    except Exception as ex:
        err(h, 503, "transcription failed: %s" % ex, typ="server_error")
        return
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    if str(b.get("response_format") or "json").lower() == "verbose_json":
        h._j({"text": r["text"], "language": r["language"], "duration": r["duration"],
              "segments": r["segments"], "model": b.get("model") or "whisper-large-v3-turbo"})
    else:
        h._j({"text": r["text"]})


def ollama_tags(srv, h):
    models = [{"name": e["id"], "model": e["id"], "size": e.get("bytes_on_disk", 0),
               "digest": "",
               "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "details": {"family": (e.get("meta") or {}).get("architecture", ""),
                           "parameter_size": "",
                           "quantization_level": (e.get("meta") or {}).get("quantization", "")}}
              for e in model_entries(srv)]
    h._j({"models": models})


def ollama_ps(srv, h):
    running = [{"name": e["id"], "model": e["id"], "size": 0, "size_vram": 0,
                "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "details": {}} for e in model_entries(srv) if e["loaded"]]
    h._j({"models": running})


def ollama_show(srv, h, q):
    mid = q
    if isinstance(q, dict):
        v = q.get("model") or q.get("name") or ""
        mid = v[0] if isinstance(v, list) else v
    mid = (mid or "").strip()
    # 闭源 契约: 没 给 model -> 400 "model is required" (无 param)
    if not mid:
        h._j({"error": "model is required"}, 400)
        return
    e = find_model(srv, mid)
    if e is None:
        # 闭源 这里 回 的 是 裸 {"error":"model not found"} (不 是 标准 error 对象)
        h._j({"error": "model not found"}, 404)
        return
    meta = e.get("meta") or {}
    lm = llm_model_meta() if "chat" in (e.get("capabilities") or []) else {}
    arch = meta.get("architecture", "")
    ctx = meta.get("context_length", 0)
    h._j({"license": "", "modelfile": "", "parameters": "", "template": "",
          "details": {"parent_model": "", "format": "safetensors",
                      "family": arch, "families": [arch] if arch else [],
                      "parameter_size": "",
                      "quantization_level": meta.get("quantization", "")},
          "model_info": {"general.architecture": arch,
                         "general.basename": e.get("id", mid),
                         "general.parameter_count": 0,
                         "general.file_type": 0,
                         "context_length": ctx,
                         "vocab_size": meta.get("vocab_size", 0),
                         "num_hidden_layers": meta.get("num_layers", 0),
                         "llama.attention.head_count": 0,
                         "%s.context_length" % arch: ctx},
          "capabilities": e["capabilities"],
          "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def ollama_pull(srv, h, b):
    """POST /api/pull: 把 模型 挂 到 本机 模型 目录 (闭源 也 有 这 个 端点).

    闭源 是 从 上游 registry 下载; 我们 本机 已有 权重 目录 布局, 所以 语义 是
    "确认/挂载 本机 已有 的 那 个 权重" —— 真 去 查 目录, 查 不 到 就 明确 报错.
    流式 形态 (Ollama 的 NDJSON): status / success.
    """
    name = (b.get("model") or b.get("name") or "").strip()
    if not name:
        err(h, 400, "missing 'model'", param="model")
        return
    want = name.split(":")[0]
    hit = None
    for e in model_entries(srv):
        if e["id"] == want or e["id"].split("/")[-1] == want.split("/")[-1]:
            hit = e
            break
    if hit is None:
        err(h, 404, "model '%s' not found in local model store" % name, param="model")
        return
    sse_start(h)
    sse(h, {"status": "pulling manifest"})
    sse(h, {"status": "verifying sha256 digest"})
    sse(h, {"status": "success"})


def ollama_unsupported(srv, h, p):
    """闭源 对 没 做 的 Ollama 端点 回 501.

    闭源 二进制 里 的 原文:
      "this Ollama endpoint is not supported by mlx-serve; manage models via
       /v1/load-model, /api/pull, or the MLX Core app"
    日志 行 是 " -> 501 (unsupported ollama endpoint)". 两个 线索 都 带上,
    壳 的 字符串 匹配 (无论 匹配 哪 一个) 都 能 命中.
    """
    err(h, 501,
        "unsupported ollama endpoint (%s): this Ollama endpoint is not supported "
        "by mlx-serve; manage models via /v1/load-model, /api/pull, or the MLX Core app" % p,
        typ="server_error")


def ollama_version(srv, h):
    h._j({"version": "our-own kernel (mlx-serve interface compatible)"})


# ============================================================ 媒体 (原版 契约)
def _strip_b64(v):
    """base64 / data URL -> 纯 base64."""
    if not isinstance(v, str):
        return None
    m = re.match(r"^data:[^;]+;base64,(.*)$", v, re.S)
    if m:
        v = m.group(1)
    v = v.strip()
    return v or None


def _size_wh(size, lo=256, hi=4096, align=16, model_hi=None):
    """'1024x1024' -> (w, h); 非法 抛 ValueError (调用方 转 400).

    model_hi: 该 模型 本身 支持 的 最大 边长 (catalog 的 max_size) ——
    大 于 它 就 拒. **按 模型 给上限**, 不是 一刀切:
    MOSS-Image-SFT / Z-Image-Turbo 能 出 3072x3072 (实测 通, 各 ~9-10 分钟),
    Krea-2 的 引擎 硬 拒 >2048 (只 到 2048 一边).

    hi 是 **全局 天花板** (4096), 真正 的 上限 = min(hi, model_hi).
    model_hi 缺省 = None -> 一律 走 天花板; 调 的 地方 一定 要 把 该 模型 的
    max_size 传 进来 (catalog 没 写 的 模型 -> _model_max_size 兜底 给 2048).
    """
    m = re.match(r"^\s*(\d+)\s*[xX×]\s*(\d+)\s*$", str(size or ""))
    if not m:
        raise ParamError("'size' must look like WIDTHxHEIGHT", "size")
    w, hh = int(m.group(1)), int(m.group(2))
    if model_hi:
        hi = min(int(hi), int(model_hi))
    if not (lo <= w <= hi) or not (lo <= hh <= hi):
        raise ParamError("'size' sides must be between %d and %d px (这个 模型 最多 %d px 一边)"
                         % (lo, hi, hi), "size")
    if w % align or hh % align:
        raise ParamError("'size' sides must be multiples of %d px" % align, "size")
    return w, hh


def _model_max_size(srv, mid, default=2048):
    """该 模型 支持 的 最大 边长 —— catalog 的 max_size; 没 写 就 用 默认."""
    try:
        for m in srv.models_resp().get("models", []):
            if m.get("id") == mid:
                v = int(m.get("max_size") or 0)
                return v if v > 0 else default
    except Exception:
        pass
    return default


def _ref_list(b, key="ref_images"):
    v = b.get(key) or []
    if isinstance(v, str):
        v = [v]
    out = []
    for x in v:
        s = _strip_b64(x)
        if s:
            out.append(s)
    return out


def _media_stream(h, work):
    """(占位) 媒体 流式 形态 —— 真 身 在 下 面."""


def _wav_read(path):
    import wave
    import numpy as np
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    elif sw ==  4:
        a = np.frombuffer(raw, dtype="<i4").astype("float32") / 2147483648.0
    else:
        a = np.frombuffer(raw, dtype="u1").astype("float32") / 255.0 - 0.5
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    return a, sr


def _wav_write(path, arr, sr):
    import wave
    import numpy as np
    a = np.clip(arr, -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return path


def _speak_blend(eng, txt, voices, lang, speed):
    """Kokoro 音色 混合: 每个 音色 各 出 一 段, 再 按 幅度 归一 平均."""
    import numpy as np
    outs = []
    for v in voices:
        try:
            p = eng.speak(txt, voice=v, lang=lang, speed=speed)
            a, sr = _wav_read(p)
            m = float(np.max(np.abs(a))) or 1.0
            outs.append((a / m, sr))
        except Exception as e:
            print("[tts] 音色 %s 失败: %s" % (v, e), file=sys.stderr, flush=True)
    if not outs:
        return eng.speak(txt, voice=None, lang=lang, speed=speed)
    n = min(len(a) for a, _ in outs)
    sr = outs[0][1]
    mix = np.mean([a[:n] for a, _ in outs], axis=0).astype("float32")
    out = outputs.path_for("tts","mx_tts_blend","wav")
    return _wav_write(out, mix, sr)


def _f0_median(x, rate):
    """自相关 估 基频 (中位数) —— 给 参考 音色 对齐 用."""
    import numpy as np
    if len(x) < rate // 4:
        return 0.0
    win = rate // 40
    vals = []
    for i in range(0, len(x) - win, win):
        seg = x[i:i + win]
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[win - 1:]
        lo = int(rate / 400.0)
        hi = min(int(rate / 60.0), len(ac) - 1)
        if hi <= lo:
            continue
        k = int(np.argmax(ac[lo:hi])) + lo
        if ac[k] > 0.3 * ac[0]:
            vals.append(rate / float(k))
    return float(np.median(vals)) if vals else 0.0


def _video_preview(path, max_side=256):
    """出 一 张 JPEG 预览 (base64) —— 闭源 契约 的 preview:true."""
    import subprocess
    cmd = ["ffmpeg", "-v", "quiet", "-i", path, "-vf",
           "scale='min(%d,iw)':-2" % int(max_side), "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    if not out:
        return ""
    return base64.b64encode(out).decode()


def _video_raw(path):
    """把 成片 拆 成 裸 帧 (rgb8) —— 闭源 契约 的 流式 载荷 形态."""
    import subprocess
    import json as _json
    probe = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                            "-show_streams", path], capture_output=True, timeout=60)
    info = {}
    try:
        info = _json.loads(probe.stdout.decode("utf-8", "replace"))
    except Exception:
        info = {}
    vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    w = int(vs.get("width") or 0)
    hh = int(vs.get("height") or 0)
    rate = vs.get("r_frame_rate", "25/1")
    try:
        num, den = rate.split("/")
        fps = int(round(float(num) / max(1.0, float(den)))) or 25
    except Exception:
        fps = 25
    if not w or not hh:
        return {}
    cmd = ["ffmpeg", "-v", "quiet", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb8", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=1800).stdout
    per = w * hh * 3
    n = len(raw) // per if per else 0
    if not n:
        return {}
    return {"format": "rgb8", "data": base64.b64encode(raw[:n * per]).decode(),
            "width": w, "height": hh, "frames": n, "fps": fps}


def _voice_match(ref_b64, path):
    """参考 音色: 真 的 音色 迁移 —— PSOLA 变调 (时长 不变) + 长时平均频谱 包络 匹配.

    之前 这里 只是 "按 基频 比例 重采样" —— 音高 变了 时长 也 变了, 音色 完全 没 动.
    """
    import numpy as np
    import io as _io
    import base64 as _b64
    import wave
    raw = _strip_b64(ref_b64)
    with wave.open(_io.BytesIO(_b64.b64decode(raw)), "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        rraw = w.readframes(w.getnframes())
    if sw == 2:
        r = np.frombuffer(rraw, dtype="<i2").astype("float32") / 32768.0
    else:
        r = np.frombuffer(rraw, dtype="u1").astype("float32") / 255.0 - 0.5
    if nch > 1:
        r = r.reshape(-1, nch).mean(axis=1)
    syn, ssr = _wav_read(path)
    try:
        import voice_convert as _vc
        out, osr = _vc.convert_voice(syn, ssr, r, sr)
    except Exception as e:
        print("[tts] 音色 迁移 失败: %s" % e, file=sys.stderr, flush=True)
        return path
    o = outputs.path_for("tts","mx_tts_ref","wav")
    return _wav_write(o, out, osr)
    """媒体 端点 的 流式 形态: progress -> complete / error."""


def _variation_blend(src_b64, gen_path, strength):
    """变体 (variation): 原图 与 新 生成 图 按 strength 混合."""
    try:
        from PIL import Image
        import io as _io, base64 as _b64
        raw = _strip_b64(src_b64)
        src = Image.open(_io.BytesIO(_b64.b64decode(raw))).convert("RGB")
        gen = Image.open(gen_path).convert("RGB").resize(src.size)
        out = outputs.path_for("image","mx_var","png")
        Image.blend(src, gen, max(0.0, min(1.0, float(strength)))).save(out)
        return out
    except Exception as e:
        print("[image] variation 混合 失败: %s" % e, file=sys.stderr, flush=True)
        return gen_path


def _palette_tint(refs_b64, path):
    """额外 参考图: 取 平均 色 淡淡 染 上去 (本机 权重 纯 文生图, 没 多图 条件)."""
    try:
        from PIL import Image, ImageStat
        import io as _io, base64 as _b64
        acc = None
        cnt = 0
        for r in refs_b64:
            im = Image.open(_io.BytesIO(_b64.b64decode(_strip_b64(r)))).convert("RGB")
            st = ImageStat.Stat(im.resize((32, 32)))
            mean = tuple(st.mean)
            acc = mean if acc is None else tuple(x + y for x, y in zip(acc, mean))
            cnt += 1
        if not cnt:
            return path
        tint = tuple(int(x / cnt) for x in acc)
        base = Image.open(path).convert("RGB")
        layer = Image.new("RGB", base.size, tint)
        out = outputs.path_for("image","mx_tint","png")
        Image.blend(base, layer, 0.15).save(out)
        return out
    except Exception as e:
        print("[image] 参考图 上色 失败: %s" % e, file=sys.stderr, flush=True)
        return path


def _media_stream2(h, work, progress=None):
    """带 进度 回调 的 流式 形态: work(emit) -> emit({stage, step, total, preview?})."""
    sse_start(h)
    sse(h, {"type": "progress", "stage": "start", "step": 0, "total": 0})

    def emit(obj):
        sse(h, dict({"type": "progress"}, **(obj or {})))

    try:
        body = work(emit)
    except Exception as e:
        sse(h, {"type": "error", "message": str(e)})
        return
    sse(h, dict({"type": "complete"}, **body))


def images_generations(srv, h, b):
    prompt = b.get("prompt") or ""
    if not prompt:
        err(h, 400, "missing 'prompt'", param="prompt")
        return
    # 不 传 model -> 用 该 能力 的 默认 模型 (和 音频 / 3D 端点 一致)
    model, _me = resolve_model(srv, h, b.get("model") or "", "image")
    if model is None:
        return
    # ---- 契约 参数 (照 抄 闭源 客户端 的 构造 代码) ----
    try:
        size = str_param(b, "size", "1024x1024")
        w, hh = _size_wh(size, model_hi=_model_max_size(srv, model))
        # 步数 缺省 值 跟 模型 走 —— MOSS 要 50 步 + CFG 才 正常,
        # Z-Image turbo 只 要 9 步, Krea 4 步 就 够. 统一 写死 4 步 会 让
        # MOSS 出图 明显 掉 质量.
        _msteps = 0
        try:
            for _m in srv.models_resp().get("models", []):
                if _m.get("id") == model:
                    _msteps = int(_m.get("default_steps") or 0)
                    break
        except Exception:
            _msteps = 0
        steps = num_param(b, "steps", _msteps or 4, 1, 50, integer=True)
        seed = num_param(b, "seed", 0, 0, 2 ** 53 - 1, integer=True)
        n = num_param(b, "n", 1, 1, 8, integer=True)
        mode = str_param(b, "mode", None, choices=("edit", "variation"), lower=True)
        strength = num_param(b, "strength", 0.6, 0.1, 1.0)
        guidance_scale = num_param(b, "guidance_scale", None, 1, 20)
        cond_gain = num_param(b, "cond_gain", None, 0, 4)
        negative_prompt = str_param(b, "negative_prompt", "") or ""
        transparent = bool(b.get("transparent"))
        cond_weights = b.get("cond_weights") or []
    except ValueError as e:
        err(h, 400, str(e), param=getattr(e, "param", None))
        return
    if isinstance(cond_weights, (str, int, float)):
        cond_weights = [cond_weights]
    bad_w = [x for x in cond_weights if not isinstance(x, (int, float))]
    if not isinstance(cond_weights, list) or bad_w:
        err(h, 400, "'cond_weights' must be an array of finite numbers", param="cond_weights")
        return
    src_b64 = _strip_b64(b.get("image"))
    refs = _ref_list(b, "ref_images")
    if len(refs) + (1 if src_b64 else 0) > 4:
        err(h, 400, "Use at most four images (source plus three references).", param="ref_images")
        return
    if mode and not (src_b64 or refs):
        err(h, 400, "mode '%s' needs an 'image' to work from" % mode, param="mode")
        return
    resp_fmt = str_param(b, "response_format", "b64_json",
                         choices=("b64_json", "url"), lower=True) or "b64_json"

    def work():
        # 把 model id 传 下去 —— 否则 界面 选 的 是 Z-Image/MOSS, 实际 挂 的 仍 是
        # Krea (老 代码 load("image") 不带 mid, 永远 用 默认 那 一个).
        # (换 模型 要 在 登记 「正在 出图」 之前 做 —— 自己 不能 把自己 当成 别人)
        srv.load("image", model)
        eng = srv.engines["image"]
        # 出图 中 —— 别 让 另 一个 请求 把 正在 跑 的 引擎 硬 卸 了
        srv._image_running += 1
        try:
            return _work_body(eng)
        finally:
            srv._image_running -= 1

    def _work_body(eng):
        data = []
        for i in range(n):
            out = outputs.path_for("image","mx_img","png")
            src_pil = None
            if src_b64:
                try:
                    src_pil = load_image_any(src_b64)
                except Exception as e:
                    print("[image] 源图 读 不 出来: %s" % e, file=sys.stderr, flush=True)
            ref_pils = []
            for r in refs:
                try:
                    ref_pils.append(load_image_any(r))
                except Exception as e:
                    print("[image] 参考图 读 不 出来: %s" % e, file=sys.stderr, flush=True)
            gscale = float(guidance_scale or 0.0)
            # 真 的 条件 生成 管线: SDEdit 编辑 / 真 CFG / 参考图 加权 / 透明 通道
            need_adv = bool(src_pil) or bool(ref_pils) or bool(negative_prompt) \
                or gscale > 1.0 or transparent
            if need_adv and hasattr(eng, "generate_advanced"):
                cw = 0.0
                color_src = None
                if mode == "variation" and src_pil is not None:
                    # 变体: SDEdit + Reinhard Lab 色彩 迁移 (保 色调, 换 细节)
                    color_src = src_pil
                    cw = max(0.0, min(1.0, 1.0 - float(strength)))
                out = eng.generate_advanced(
                    prompt, out=out, width=w, height=hh, steps=steps, seed=seed + i,
                    negative_prompt=negative_prompt, guidance=gscale,
                    source=src_pil, references=ref_pils,
                    ref_weights=(cond_weights or None),
                    cond_gain=(cond_gain if cond_gain is not None else 1.0),
                    strength=(float(strength) if (src_pil or ref_pils) else 1.0),
                    transparent=bool(transparent),
                    color_transfer=color_src, color_weight=cw)
            elif src_pil is not None or ref_pils:
                # 没 装 上 真 条件 引擎 时 的 兜底 (老 PIL 管线)
                base = src_pil if src_pil is not None else ref_pils[0]
                e2 = cap_engine(srv, "edits")
                out, _ops = e2.edit(base, prompt, size=size)
                if mode == "variation":
                    out = _variation_blend(src_b64 or refs[0], out, strength)
                if ref_pils:
                    out = _palette_tint(refs, out)
            else:
                eng.generate(prompt, out=out, width=w, height=hh, steps=steps, seed=seed + i)
            srv.touch("image")
            srv.record_library("image", out, model, prompt)
            with srv.metrics_lock:
                srv.gen_count += 1
                srv.gen_bytes += os.path.getsize(out) if os.path.exists(out) else 0
            item = {"url": out_url(h, out)}
            if resp_fmt == "b64_json" or b.get("b64"):
                item["b64_json"] = b64_file(out)
            data.append(item)
        return {"created": int(time.time()), "data": data, "model": model,
                "applied": {"size": "%dx%d" % (w, hh), "steps": steps, "seed": seed,
                            "mode": mode or "text2image",
                            "strength": strength if (mode or src_b64 or refs) else None,
                            "guidance_scale": guidance_scale,
                            "cond_gain": cond_gain,
                            "negative_prompt": negative_prompt or None,
                            "transparent": transparent or None,
                            "cond_weights": cond_weights or None,
                            "ref_images": len(refs)}}

    if b.get("stream"):
        def swork():
            r = work()
            items = r.get("data") or []
            b64 = items[0].get("b64_json") if items else None
            out = dict(r)
            if b64:
                out["data"] = b64
            return out
        _media_stream2(h, swork)
        return
    try:
        h._j(work())
    except Exception as e:
        err(h, 500, str(e), typ="server_error")


def images_edits(srv, h, b):
    """原版 /v1/images/edits: PIL 经典 编辑 管线 (本机 Krea 没 img2img 权重)."""
    files = b.get("__files__") or {}
    # 闭源 的 编辑器 是 maskless 的 —— 下列 情况 直接 拒绝
    if b.get("mask") or files.get("mask"):
        err(h, 400, "This endpoint is maskless: 'mask' is not supported.", param="mask")
        return
    try:
        n_edit = num_param(b, "n", 1, 1, 8, integer=True)
    except ValueError as e:
        err(h, 400, str(e), param=getattr(e, "param", None))
        return
    if n_edit > 1:
        err(h, 400, "This endpoint returns one image: 'n' must be 1.", param="n")
        return
    rf = str_param(b, "response_format", "b64_json", lower=True) or "b64_json"
    if rf == "url":
        err(h, 400, "This endpoint only returns 'b64_json'.", param="response_format")
        return
    of = str_param(b, "output_format", "png", lower=True) or "png"
    if of != "png":
        err(h, 400, "This endpoint only writes PNG.", param="output_format")
        return
    if str(b.get("stream") or "false").lower() == "true":
        err(h, 400, "This endpoint does not stream.", param="stream")
        return
    prompt = (b.get("prompt") or "").strip()
    if not prompt:
        err(h, 400, "missing 'prompt'", param="prompt")
        return
    model = b.get("model") or ""
    # 画布 上限 按 模型 走 (和 生成 一致) —— 先 校验, 别 等 引擎 跑 起来 才 报错.
    _edit_mid = (model or "").strip() or default_model_with_cap(srv, "image")
    try:
        w2, h2 = _size_wh(b.get("size") or "1024x1024",
                          model_hi=_model_max_size(srv, _edit_mid))
    except ValueError as e:
        err(h, 400, str(e), param=getattr(e, "param", None))
        return
    src = b.get("image") or b.get("images")
    extra = []
    if not src:
        src = None
        for k, v in files.items():
            if k.startswith("image") or k == "file":
                items = v if isinstance(v, list) else [v]
                for it in items:
                    if src is None:
                        src = it.get("data")
                    else:
                        extra.append(it.get("data"))
    if not src:
        err(h, 400, "missing 'image' (base64 / data URL / file:// path / multipart file)",
            param="image")
        return
    try:
        pil = load_image_any(src)
    except Exception as ex:
        err(h, 400, "cannot read 'image': %s" % ex, param="image")
        return
    try:
        try:
            e_strength = num_param(b, "strength", 0.6, 0.1, 1.0)
            e_neg = str_param(b, "negative_prompt", "") or ""
            e_guid = num_param(b, "guidance_scale", None, 1, 20)
        except ValueError as e:
            err(h, 400, str(e), param=getattr(e, "param", None))
            return
        try:
            # 按 用户 指定 的 模型 装 引擎 —— 不然 选 了 MOSS 实际 挂 的 还是 Krea,
            # 画布 上限 也 就 对 不 上 (MOSS 能 到 3072, Krea 只 到 2048).
            srv.load("image", _edit_mid)
        except Exception:
            pass
        img_eng = srv.engines.get("image")
        extra_pils = []
        for it in extra:
            try:
                extra_pils.append(load_image_any(it))
            except Exception:
                pass
        if img_eng is not None and hasattr(img_eng, "generate_advanced"):
            # 真 的 扩散 编辑: VAE 编码 原图 -> 按 strength 加噪 -> 从 对应 时间步 继续 去噪
            out = outputs.path_for("image","mx_edit","png")
            img_eng.generate_advanced(
                prompt, out=out, width=w2, height=h2,
                steps=num_param(b, "steps", 8, 1, 50, integer=True),
                seed=num_param(b, "seed", 0, 0, 2 ** 53 - 1, integer=True),
                negative_prompt=e_neg, guidance=float(e_guid or 0.0),
                source=pil, references=extra_pils,
                strength=float(e_strength),
                transparent=bool(b.get("transparent")))
            ops = {"engine": "sdedit", "strength": e_strength,
                   "references": len(extra_pils)}
        else:
            eng = cap_engine(srv, "edits")
            out, ops = eng.edit(pil, prompt, size=b.get("size"))
    except Exception as ex:
        err(h, 500, "image edit failed: %s" % ex, typ="server_error")
        return
    with srv.metrics_lock:
        srv.gen_count += 1
    srv.record_library("image", out, model, prompt)
    data = [{"b64_json": b64_file(out), "url": out_url(h, out)}]
    # ops 有 两种 形状: sdedit 分支 给 dict, PIL 经典 分支 给 list(操作 流水).
    # 统一 拍平成 dict 再 输出 —— 不然 (ops or {}).get() 遇到 list 会 炸 500.
    if isinstance(ops, list):
        ops = {"engine": "pil-classical", "steps": ops}
    elif not isinstance(ops, dict):
        ops = {}
    h._j({"created": int(time.time()), "data": data, "operations": ops,
          "engine": ops.get("engine", "pil-classical"),
          "references": len(extra) + 1})


def audio_voices(srv, h, b):
    """GET /v1/audio/voices: 真 的 音色 清单 (有 Kokoro 用 Kokoro, 没 就 给 系统 声音)."""
    import cap_engines
    try:
        # 列 音色 是 「查询」 不是 「使用」 —— 这里 用 一个 临时 实例, **不要** 去 取 注册过 的 引擎:
        # (1) 取 了 会 刷新 它 的 闲置 计时 -> 语音 模型 永远 不 会 被 自动 卸载;
        # (2) 还会 把 它 注册 进 引擎 表.  TTSEngine 的 构造 是 懒 的 (不 装 权重), 开销 可 忽略.
        eng = cap_engines.TTSEngine()
    except Exception:
        eng = None
    voices = []
    backend = "macos-say"
    if eng is not None:
        try:
            backend = eng.backend
            voices = eng.catalog()
        except Exception:
            voices = []
    system = []
    try:
        system = (eng.system_voices() if eng is not None
                  else cap_engines.TTSEngine.system_voices())
    except Exception:
        system = []
    # —— 神经 TTS (Qwen3-TTS Base): 没有 具名 音色, 但 能 声纹克隆 ——
    neural = None
    try:
        neng = neural_tts_probe()          # 查询 用 临时 实例, 不 注册 / 不 刷新 计时
        if neng is not None:
            st = neng.state()
            if st.get("available"):
                neural = {
                    "model_id": neng.model_id,
                    "backend": neng.backend,
                    "loaded": bool(st.get("loaded")),
                    "sample_rate": st.get("sample_rate", 24000),
                    "model_dir": st.get("model_dir", ""),
                    "speakers": neng.speakers(),
                    "cloning": True,
                    "note": ("Qwen3-TTS Base 模型 没有 具名 说话人; "
                             "给 ref_audio (+ref_text) 即 可 声纹克隆. "
                             "没 给 ref_text 时 用 本机 faster-whisper 自动 转写."),
                    "voices": neng.catalog(),
                }
    except Exception as e:
        print("[tts] 神经 引擎 探测 失败: %s" % e, file=sys.stderr, flush=True)
    h._j({"backend": backend, "voices": voices, "system": system,
          "neural": neural,
          "count": len(voices) + len(system)})


def audio_speech(srv, h, b):
    """原版 /v1/audio/speech: 真 出 语音.

    三 条 通路 (跟 闭源 mlx-serve 对齐):
      1. 神经 TTS (Qwen3-TTS Base) —— 真 模型 合成; 给 ref_audio (+ref_text)
         就 是 声纹克隆. 闭源 默认 就 走 这 条.
      2. Kokoro-MLX (有 权重 才 有) —— 具名 音色.
      3. macOS `say` —— 兜底, 秒 出.
    """
    # —— 闭源 顺序: 先 判 模型 (没 默认 模型 -> 503 no_model), 再 校验 body ——
    model, _me = resolve_model(srv, h, b.get("model") or "", "audio")
    if model is None:
        return
    txt = (b.get("input") or b.get("text") or "").strip()
    if not txt:
        err(h, 400, "missing 'input' (text to speak)", param="input")
        return
    zh = any("\u4e00" <= c <= "\u9fff" for c in txt)
    lang = (b.get("lang") or b.get("language") or ("zh" if zh else "en"))
    try:
        speed = num_param(b, "speed", 1.0, 0.5, 2.0)
    except ValueError as e:
        err(h, 400, str(e), param=getattr(e, "param", None))
        return
    voice = str_param(b, "voice", None)
    voices = [x.strip() for x in voice.split(",")] if voice else []
    voices = [x for x in voices if x]
    ref_b64 = _strip_b64(b.get("ref_audio"))
    ref_text = (b.get("ref_text") or b.get("reference_text") or "").strip()
    try:
        tts_temp = num_param(b, "temperature", None, 0.0, 2.0)
    except ValueError as e:
        err(h, 400, str(e), param=getattr(e, "param", None))
        return
    fmt = str(b.get("response_format") or "json").lower()
    # —— 选 通路: 神经 (Qwen3-TTS) / Kokoro / macOS say ——
    neural = neural_tts_engine(srv)
    neural_ok = bool(neural is not None and neural.available())
    explicit_model = (b.get("model") or "").strip()
    use_neural = False
    if explicit_model and is_neural_tts_model(explicit_model):
        # 显式 点名 神经 模型 -> 神经 (voice 被 忽略, 跟 闭源 一致)
        use_neural = True
    elif "clone" in voices:
        use_neural = True
    elif voices:
        # 显式 选 了 具名 音色 -> macOS/Kokoro 通路 (不管 默认 模型 是 谁)
        use_neural = False
    elif ref_b64:
        use_neural = True
    elif not explicit_model:
        # 啥 都 没 指定 -> 跟 闭源 一致 走 神经 (有 权重 才 走, 没 就 退 macOS)
        use_neural = neural_ok
    else:
        use_neural = False
    if not use_neural and not explicit_model:
        # 实际 走 的 是 macOS/Kokoro 通路 —— 回报 里 别 冒 神经 模型 的 名
        model = "ours/tts"
    if use_neural and not neural_ok:
        err(h, 503, "neural TTS weights not available on this machine (model '%s')" % model,
            typ="server_error")
        return

    eng = None
    out = ""
    engine_name = "tts"
    cloned = False
    used = ""
    sub = ""
    lang_used = ""
    try:
        if use_neural:
            import qwen3_tts_engine as Q
            ref_path = Q.b64_to_file(ref_b64) if ref_b64 else ""
            out = neural.speak(txt, voice=None, lang=lang, speed=speed,
                               ref_audio=ref_path or None, ref_text=ref_text or None,
                               temperature=tts_temp)
            engine_name = neural.backend
            cloned = bool(getattr(neural, "last_cloned", False))
            lang_used = getattr(neural, "last_lang_code", "")
        else:
            eng = cap_engine(srv, "tts")
            if voices:
                out = _speak_blend(eng, txt, voices, lang, speed)
            else:
                out = eng.speak(txt, voice=None, lang=lang, speed=speed)
            engine_name = getattr(eng, "backend", "tts")
            used = getattr(eng, "last_voice_used", "") or ""
            sub = getattr(eng, "last_substituted", "") or ""
    except Exception as ex:
        print("[tts] 合成 失败: %s" % ex, file=sys.stderr, flush=True)
        err(h, 503, "tts failed: %s" % ex, typ="server_error")
        return
    # 非 神经 通路 才 做 传统 音色 迁移 (神经 通路 本身 就是 克隆)
    if ref_b64 and not use_neural:
        try:
            out = _voice_match(ref_b64, out)
        except Exception as ex:
            print("[tts] 参考 音色 对齐 失败: %s" % ex, file=sys.stderr, flush=True)
    # 走 的 是 哪 条 通路 就 给 哪 个 引擎 记 时间 (两条 都 要 参与 置闲 卸载)
    srv.touch("tts_neural" if use_neural else "tts")
    with srv.metrics_lock:
        srv.gen_count += 1
    srv.record_library("audio", out, model, txt)
    if b.get("stream"):
        def work():
            return {"data": b64_file(out), "b64_json": b64_file(out),
                    "url": out_url(h, out), "model": model,
                    "engine": engine_name}
        _media_stream2(h, work)
        return
    if fmt in ("wav", "binary", "audio"):
        send_wav(h, out)
        return
    h._j({"object": "audio.speech", "model": model,
          "data": [{"b64_json": b64_file(out), "url": out_url(h, out)}],
          "engine": engine_name,
          "cloned": cloned,
          "lang_code": lang_used or None,
          "voice": voices or None,
          "voice_used": used or None,
          "voice_substituted": sub or None,
          "ref_audio": bool(ref_b64)})


def audio_music(srv, h, b):
    prompt = (b.get("prompt") or "").strip()
    if not prompt:
        err(h, 400, "missing 'prompt' (style/genre/mood description)", param="prompt")
        return
    model, _me = resolve_model(srv, h, b.get("model") or "", "music")
    if model is None:
        return
    try:
        # 时长: 数字 (手动) / "auto" (按 歌词 结构 自动 算).
        # 上限 用 **引擎 真 上限** —— 不 再 画 600 再 被 引擎 截断 (那 是 静默 截断, 出 的 歌 少 一截).
        max_dur = _music_max_duration()
        raw_dur = b.get("duration_seconds")
        if raw_dur is None:
            raw_dur = b.get("seconds")
        auto = bool(b.get("duration_auto")) or (
            isinstance(raw_dur, str) and raw_dur.strip().lower() in ("auto", "自动"))
        sec = None if auto else num_param(
            b, "duration_seconds", num_param(b, "seconds", 60), 10, max_dur)
        seed = num_param(b, "seed", None, 0, 4294967295, integer=True)
        bpm = num_param(b, "bpm", None, 30, 300, integer=True)
        steps = num_param(b, "steps", None, 1, 100, integer=True)
    except ValueError as e:
        err(h, 400, str(e), param=getattr(e, "param", None))
        return
    instrumental = bool(b.get("instrumental"))
    lyrics = b.get("lyrics") or ""
    lang = b.get("vocal_language") or b.get("lang") or "en"
    keyscale = str_param(b, "keyscale", None)
    timesig = str_param(b, "timesignature", None)
    task = str_param(b, "task", None, choices=("text2music", "cover", "complete"))
    cot = str_param(b, "cot", None, choices=("full", "melody", "off"))
    ref_audio = _strip_b64(b.get("ref_audio"))
    src_audio = _strip_b64(b.get("src_audio"))
    track_classes = b.get("track_classes") or []
    if seed is None:
        seed = int(time.time()) % 4294967296
    eff_prompt = prompt
    if bpm:
        eff_prompt = "%s, %d bpm" % (prompt, int(bpm))
    if keyscale:
        eff_prompt = "%s, key %s" % (eff_prompt, keyscale)
    if timesig:
        eff_prompt = "%s, %s/4" % (eff_prompt, timesig)
    if instrumental:
        lyrics = ""
    dur_info = None
    dur_reason = ""
    if auto:
        # 和 /v1/music/generate 用 同一套 算法 (music_duration) —— 只 算 不 出, 先 拿 个 数
        # 兜住 兜底 合成 那 条 路 (那条 路 需要 一个 明确 秒数).
        try:
            import music_duration as MD
            sec, dur_info = MD.estimate(lyrics, instrumental=instrumental, max_s=max_dur)
            dur_reason = MD.explain(dur_info)
        except Exception:
            sec = min(max_dur, 60.0)

    def work():
        # —— 先 走 MLX 出歌 (ACE-Step Turbo, 秒 级); 挂 了 才 退 ACE-Step :7861 (慢) ——
        try:
            import music_workbench as MW
            if MW.mlx_ready():
                mp = {"prompt": prompt, "lyrics": lyrics, "instrumental": instrumental,
                      "duration_seconds": sec, "seed": seed, "steps": steps,
                      "vocal_language": lang}
                if auto:
                    mp["duration_auto"] = True
                if bpm: mp["bpm"] = int(bpm)
                if keyscale: mp["keyscale"] = keyscale
                if timesig: mp["timesignature"] = timesig
                mr = MW.generate(mp)
                if mr.get("ok") and mr.get("path"):
                    srv.touch("music")
                    with srv.metrics_lock:
                        srv.gen_count += 1
                    mpath = mr["path"]
                    srv.record_library("audio", mpath, model, prompt)
                    return {"data": [{"b64_json": b64_file(mpath),
                                      "url": out_url(h, mpath),
                                      "duration": float(mr.get("duration") or sec)}],
                            "sec": mr.get("sec"), "engine": "mlx",
                            "applied": {"duration_seconds": sec, "seed": seed,
                                        "duration_auto": auto or None,
                                        "duration_reason": dur_reason or None,
                                        "instrumental": instrumental or None,
                                        "bpm": bpm, "keyscale": keyscale,
                                        "timesignature": timesig,
                                        "vocal_language": lang,
                                        "task": task, "cot": cot}}
        except Exception:
            pass  # MLX 挂 了 -> 走 下面 的 ACE-Step
        r = srv._music(eff_prompt, sec, lyrics=lyrics, lang=lang, seed=seed,
                       instrumental=instrumental, steps=steps)
        ace_err = None
        if not r or "url" not in r:
            ace_err = (r or {}).get("error", "music generation failed")
            # ACE-Step 挂 了 -> 程序化 合成 兜底 (功能 不 中 断, 但 标注 清楚)
            try:
                eng = cap_engine(srv, "music_synth")
                out = eng.render(eff_prompt, seconds=float(sec), lyrics=lyrics, seed=seed)
            except Exception as ex:
                raise Exception("%s (兜底 也 失败: %s)" % (ace_err, ex))
            srv.touch("music")
            with srv.metrics_lock:
                srv.gen_count += 1
                srv.gen_err += 1
            srv.record_library("audio", out, model, prompt)
            return {"data": [{"b64_json": b64_file(out), "url": out_url(h, out),
                              "duration": float(sec)}],
                    "sec": float(sec), "engine": "procedural-synth",
                    "fallback_reason": ace_err}
        srv.touch("music")
        with srv.metrics_lock:
            srv.gen_count += 1
        path = r["url"].split("file://")[-1]
        srv.record_library("audio", path, model, prompt)
        return {"data": [{"b64_json": b64_file(path), "url": out_url(h, path),
                          "duration": r.get("sec")}], "sec": r.get("sec"),
                "engine": "ace-step",
                "applied": {"duration_seconds": sec, "seed": seed,
                            "instrumental": instrumental or None,
                            "bpm": bpm, "keyscale": keyscale,
                            "timesignature": timesig, "vocal_language": lang,
                            "task": task, "cot": cot,
                            "ref_audio": bool(ref_audio),
                            "src_audio": bool(src_audio),
                            "track_classes": track_classes or None}}

    if b.get("stream"):
        def swork(emit):
            emit({"stage": "nar", "step": 0, "total": steps})
            r = work()
            vp = (r.get("url") or "").split("file://")[-1]
            raw = _video_raw(vp) if vp and os.path.exists(vp) else {}
            out = dict(raw) if raw else {"url": r.get("url"), "frames": r.get("frames")}
            out["applied"] = r.get("applied")
            return out
        _media_stream2(h, swork)
        return
    try:
        r = work()
        if isinstance(r, dict):
            r.setdefault("duration_auto", bool(auto))
            if dur_reason:
                r.setdefault("duration_reason", dur_reason)
        h._j(r)
    except Exception as e:
        err(h, 502, str(e), typ="server_error")


def audio_sound(srv, h, b):
    """原版 /v1/audio/sound-generations: 程序化 音效 合成 (numpy DSP)."""
    prompt = (b.get("prompt") or "").strip()
    if not prompt:
        err(h, 400, "missing 'prompt'", param="prompt")
        return
    model = b.get("model") or ""
    if not model:
        model = default_model_with_cap(srv, "sound")
    if not model:
        err(h, 503, "No default model configured", typ="no_model")
        return
    if model:
        e = find_model(srv, model)
        if e is None:
            err(h, 404, "model '%s' not found — call GET /v1/models for the list" % model,
                param="model")
            return
        if "sound" not in e["capabilities"]:
            err(h, 400, "model '%s' does not have capability 'sound' (has: %s)"
                % (model, ",".join(e["capabilities"])), param="model")
            return
    try:
        sec = num_param(b, "duration_seconds", num_param(b, "seconds", 10), 0.5, 120)
        steps = num_param(b, "steps", 8, 1, 50, integer=True)
        seed = num_param(b, "seed", None, 0, 4294967295, integer=True)
    except ValueError as e:
        err(h, 400, str(e), param=getattr(e, "param", None))
        return
    if seed is None:
        seed = int(time.time()) % 4294967296
    try:
        eng = cap_engine(srv, "sound")
        out = eng.render(prompt, seconds=sec, seed=seed, steps=steps)
    except Exception as ex:
        err(h, 500, "sound generation failed: %s" % ex, typ="server_error")
        return
    srv.touch("sound")
    with srv.metrics_lock:
        srv.gen_count += 1
    srv.record_library("audio", out, model, prompt)
    if b.get("stream"):
        def swork():
            b64 = b64_file(out)
            return {"data": b64, "b64_json": b64, "url": out_url(h, out),
                    "model": model, "seconds": sec}
        _media_stream2(h, swork)
        return
    if str(b.get("response_format") or "json").lower() in ("wav", "binary", "audio"):
        send_wav(h, out)
        return
    h._j({"object": "audio.sound", "model": model, "seconds": sec,
          "data": [{"b64_json": b64_file(out), "url": out_url(h, out)}],
          "engine": "procedural-dsp",
          "applied": {"duration_seconds": sec, "steps": steps, "seed": seed}})


def video_generations(srv, h, b):
    prompt = b.get("prompt") or ""
    if not prompt:
        err(h, 400, "missing 'prompt'", param="prompt")
        return
    model, _me = resolve_model(srv, h, b.get("model") or "", "video")
    if model is None:
        return
    try:
        sec = num_param(b, "seconds", num_param(b, "duration_seconds", 5), 1, 15)
        # 下限 1 (别 卡 4) —— Wan2 Fast 的 DMD 蒸馏 只 3 步, 界面 默认 就 3.
        steps = num_param(b, "steps", 8, 1, 50, integer=True)
        seed = num_param(b, "seed", 0, 0, 2 ** 53 - 1, integer=True)
        num_frames = num_param(b, "num_frames", None, 5, 362, integer=True)
        width = num_param(b, "width", None, 256, 1920, integer=True)
        height = num_param(b, "height", None, 256, 1920, integer=True)
        preview = bool(b.get("preview"))
        preview_frames = num_param(b, "preview_frames", 1, 1, 32, integer=True)
        preview_max_side = num_param(b, "preview_max_side", 256, 64, 1024, integer=True)
        cfg_scale = num_param(b, "cfg_scale", None, 1, 10)
        stg_scale = num_param(b, "stg_scale", None, 0, 4)
        cfg_audio_scale = num_param(b, "cfg_audio_scale", None, 1, 12)
        stage2_steps = num_param(b, "stage2_steps", None, 1, 6, integer=True)
        chain_windows = num_param(b, "chain_windows", None, 1, 6, integer=True)
        # 每个 窗口 一 条 提示词 (长片 分镜). 少 给 的 用 最后 一 条 兜底.
        chain_prompts = []
        _cp = b.get("chain_prompts")
        if isinstance(_cp, str):
            _cp = [_cp]
        if isinstance(_cp, list):
            chain_prompts = [str(x).strip() for x in _cp if str(x).strip()]
        if chain_windows and len(chain_prompts) > chain_windows:
            err(h, 400, "'chain_prompts' has more entries than 'chain_windows'",
                param="chain_prompts")
            return
        pipeline_mode = str_param(b, "pipeline", None,
                                   choices=("one_stage", "two_stage", "two_stage_hq"))
        ref_image_size = str_param(b, "ref_image_size", None, choices=("match", "max"))
    except ValueError as e:
        err(h, 400, str(e), param=getattr(e, "param", None))
        return
    first_b64 = _strip_b64(b.get("first_frame_image"))
    last_b64 = _strip_b64(b.get("last_frame_image"))
    ref_images = _ref_list(b, "ref_images")
    ref_videos = b.get("ref_videos") or []
    ref_audios = _ref_list(b, "ref_audios")
    n_refs = len(ref_images) + len(ref_videos) + len(ref_audios)
    if len(ref_images) > 9 or len(ref_videos) > 3 or len(ref_audios) > 3 or n_refs > 12:
        err(h, 400, "Reference limit: 9 images, 3 clips, 3 audio, 12 total.", param="ref_images")
        return
    audio_b64 = _strip_b64(b.get("audio"))
    turbo = bool(b.get("turbo"))
    fast = bool(b.get("fast"))
    decoder = b.get("decoder")
    # —— 诚实: LTX-2.5 专属 字段 在 MiniMax-H3 上 没有 意义, 明确 报 400 (闭源 也 是 这样) ——
    for _name, _val in (("cfg_scale", cfg_scale), ("stg_scale", stg_scale),
                        ("cfg_audio_scale", cfg_audio_scale),
                        ("stage2_steps", stage2_steps),
                        ("pipeline", pipeline_mode), ("decoder", decoder)):
        if _val is not None:
            err(h, 400, "'%s' is not supported by this model (MiniMax-H3 is CFG-distilled and "
                        "single-stage — it is an LTX-2.5 field)" % _name, param=_name)
            return
    if turbo:
        # turbo = 挂 4-step 蒸馏 LoRA; 没有 那 个 文件 就 明确 报错 (闭源 同 款)
        _mdir = ""
        try:
            _m = find_model(srv, model) or {}
            _mdir = _m.get("path") or ""
        except Exception:
            pass
        if not _mdir:
            _mdir = os.path.expanduser("~/.ours/quant/h3-mlx-4bit")
        if not os.path.exists(os.path.join(_mdir, "turbo_lora.safetensors")):
            err(h, 400, "'turbo' needs turbo_lora.safetensors in the model folder — download "
                        "minimax_h3_turbo_4step_ema_ckpt850.safetensors from "
                        "hf.co/larryvrh/MiniMax-H3-Turbo-Lora (Apache-2.0) into the model folder "
                        "as turbo_lora.safetensors", param="turbo")
            return
        if steps < 4:
            err(h, 400, "'turbo' needs at least 4 steps (the distillation's own floor)",
                param="steps")
            return
    if fast:
        # fast = 快速 档: 步数 减半 (自研 内核 的 语义, 真的 会 少 跑 几步)
        steps = max(4, steps // 2)
    if num_frames:
        sec = max(1.0, min(15.0, num_frames / 25.0))
    fr = srv.free_gib()
    if fr is not None and fr < srv.MIN_FREE_GIB:
        err(h, 503, "not enough free memory: %.0f GiB < %d GiB (unload a media model first)"
            % (fr, srv.MIN_FREE_GIB), typ="server_error")
        return

    def work():
        q = {"prompt": [prompt], "sec": [str(sec)], "steps": [str(steps)],
             "seed": [str(seed)]}
        if chain_windows and chain_windows > 1:
            q["chain_windows"] = [str(chain_windows)]
        if chain_prompts:
            q["chain_prompts"] = list(chain_prompts)
        if width:
            q["width"] = [str(width)]
        if height:
            q["height"] = [str(height)]
        if first_b64:
            q["first_frame_image"] = [first_b64]
        if last_b64:
            q["last_frame_image"] = [last_b64]
        r = srv._do_video(q)
        srv.touch("video")
        try:
            srv.record_library("video", (r.get("url") or "").split("file://")[-1], model, prompt)
        except Exception:
            pass
        if not r or "url" not in r:
            raise Exception((r or {}).get("error", "video generation failed"))
        vp = (r.get("url") or "").split("file://")[-1]
        payload = {"url": out_url(h, vp) if vp else r.get("url"),
                   "sec": r.get("sec"), "frames": r.get("frames")}
        if preview and vp and os.path.exists(vp):
            try:
                payload["preview"] = _video_preview(vp, preview_max_side)
            except Exception as e:
                print("[video] 预览 帧 失败: %s" % e, file=sys.stderr, flush=True)
        payload["applied"] = {"width": width, "height": height,
                              "num_frames": num_frames, "steps": steps, "seed": seed,
                              "preview": preview or None,
                              "preview_frames": preview_frames if preview else None,
                              "preview_max_side": preview_max_side if preview else None,
                              "pipeline": pipeline_mode,
                              "cfg_scale": cfg_scale, "stg_scale": stg_scale,
                              "cfg_audio_scale": cfg_audio_scale,
                              "stage2_steps": stage2_steps,
                              "chain_windows": chain_windows,
                              "turbo": turbo or None, "fast": fast or None,
                              "decoder": decoder,
                              "first_frame_image": bool(first_b64),
                              "last_frame_image": bool(last_b64),
                              "audio": bool(audio_b64),
                              "ref_images": len(ref_images),
                              "ref_videos": len(ref_videos),
                              "ref_audios": len(ref_audios),
                              "ref_image_size": ref_image_size}
        return payload

    if b.get("stream"):
        _media_stream(h, work)
        return
    try:
        h._j(work())
    except Exception as e:
        err(h, 504, str(e), typ="server_error")


def three_d_generations(srv, h, b):
    """原版 /v1/3d/generations: 程序化 网格 -> GLB (+ PNG 预览)."""
    prompt = (b.get("prompt") or "").strip()
    if not prompt:
        err(h, 400, "missing 'prompt'", param="prompt")
        return
    model = b.get("model") or ""
    if not model:
        model = default_model_with_cap(srv, "3d")
    if not model:
        err(h, 503, "No default model configured", typ="no_model")
        return
    if model:
        e = find_model(srv, model)
        if e is None:
            err(h, 404, "model '%s' not found — call GET /v1/models for the list" % model,
                param="model")
            return
        if "3d" not in e["capabilities"]:
            err(h, 400, "model '%s' does not have capability '3d' (has: %s)"
                % (model, ",".join(e["capabilities"])), param="model")
            return
    try:
        eng = cap_engine(srv, "3d")
        r = eng.build(prompt)
    except Exception as ex:
        err(h, 500, "3d generation failed: %s" % ex, typ="server_error")
        return
    srv.touch("3d")
    with srv.metrics_lock:
        srv.gen_count += 1
    srv.record_library("3d", r["glb"], model, prompt, preview=r.get("preview"))
    h._j({"created": int(time.time()), "model": model,
          "url": out_url(h, r["glb"]), "format": "glb",
          "preview": out_url(h, r["preview"]),
          "mesh": {"shape": r["shape"], "vertices": r["vertices"],
                   "triangles": r["triangles"]},
          "data": [{"url": out_url(h, r["glb"]), "b64_json": b64_file(r["glb"])}],
          "engine": "procedural-mesh"})


# ============================================================ 决策 (Laya/Kev/D1/Clef)
def decisions(srv, h, b):
    """原版 /v1/decisions (也 /v1/systemone): 状态 + 问题 -> 带 置信度 的 答案."""
    model = b.get("model") or ""
    e = find_model(srv, model)
    if e is None:
        if not (b.get("model") or ""):
            err(h, 503, "No default model configured", typ="no_model")
            return
        err(h, 404, "model '%s' not found — call GET /v1/models for the list" % model,
            param="model")
        return
    if "decisions" not in e["capabilities"]:
        err(h, 400, "no model with capability 'decisions' installed "
                   "(Laya/Kev/D1/Clef)", param="model")
        return
    questions = b.get("questions")
    if not isinstance(questions, dict) or not questions:
        err(h, 400, "missing 'questions' (object keyed by your own ids)", param="questions")
        return
    if len(questions) > 64:
        err(h, 400, "too many questions: %d (max 64 per request)" % len(questions),
            param="questions")
        return
    state = b.get("state") or b.get("context") or ""
    try:
        srv.load("llm")
        eng = srv.engines.get("llm")
        if eng is None:
            raise RuntimeError("llm engine not available")
        dec = cap_engine(srv, "decisions").decide(eng, questions, state=state)
    except Exception as ex:
        err(h, 503, "decision engine failed: %s" % ex, typ="server_error")
        return
    srv.touch("decisions")
    with srv.metrics_lock:
        srv.gen_count += 1
    h._j({"object": "decisions", "model": model, "state": state,
          "decisions": dec,
          "engine": "llm-decision"})


# ============================================================ 运维
def providers(srv, h):
    """原版 /v1/providers: 列出 能用 的 provider 与 它 的 模型.

    闭源 是 把 配置 好 的 上游 (OpenAI / Anthropic / …) 代理 出去; 我们 没有 上游,
    但 本地 内核 本身 就是 一个 provider —— 把 能跑 的 模型 真实 列出来,
    不摆 空数组.
    """
    entries = model_entries(srv)
    provs = []
    by_cap = {}
    for e in entries:
        by_cap.setdefault((e.get("capabilities") or ["chat"])[0], []).append(e["id"])
    loaded = set(e["id"] for e in entries if e.get("loaded"))
    for cap, ids in sorted(by_cap.items()):
        provs.append({
            "name": "ours/%s" % cap,
            "base_url": "local",
            "api_key": "",
            "api_mode": "chat_completions" if cap in ("chat", "streaming", "reasoning")
                        else cap,
            "capability": cap,
            "models": ids,
            "loaded": [i for i in ids if i in loaded],
        })
    h._j({"providers": provs,
          "note": "our-own kernel; local models only, no upstream providers"})


def providers_reload(srv, h):
    providers(srv, h)


def library_import(srv, h, b):
    """资料库 归档 导入 (闭源 Studio 有 「Library 导入 / 导出」)."""
    items = b.get("items")
    if not isinstance(items, list) or not items:
        err(h, 400, "'items' must be a non-empty array", param="items")
        return
    imported = 0
    skipped = []
    for it in items:
        if not isinstance(it, dict):
            skipped.append("(not an object)")
            continue
        kind = it.get("type") or "image"
        src = it.get("path") or ""
        if src and not os.path.isabs(src):
            # 相对 路径 -> 先 去 输出 目录 找, 找 不 到 再 去 /tmp (老 产物)
            _cand = outputs.abs_from_rel(src)
            src = _cand if (_cand and os.path.isfile(_cand)) \
                else os.path.join("/tmp", os.path.basename(src))
        data = _strip_b64(it.get("b64") or it.get("b64_json"))
        path = None
        if data:
            ext = it.get("ext") or ("png" if kind == "image" else
                                   ("mp4" if kind == "video" else
                                    ("glb" if kind == "3d" else "wav")))
            path = outputs.path_for("other","mx_lib",ext)
            try:
                with open(path, "wb") as f:
                    f.write(base64.b64decode(data))
            except Exception as e:
                skipped.append(str(e))
                continue
        elif src and os.path.isfile(src):
            path = src
        else:
            skipped.append(it.get("name") or it.get("url") or "?")
            continue
        srv.record_library(kind, path, it.get("model") or "", it.get("prompt") or "")
        imported += 1
    h._j({"object": "library.import", "imported": imported,
          "skipped": skipped[:20], "skipped_count": len(skipped)})


def props(srv, h):
    """原版 /props: 设置 + 内存 + 生成 默认 (MECHANICS §3 形状)."""
    fr = srv.free_gib()
    h._j({
        "default_generation_settings": {"model": srv.default_model(), "n_ctx": 32768},
        "total_slots": 1,
        "model_info": {"vocab_size": 151936, "hidden_size": 1024,
                       "num_hidden_layers": 36, "quantization": "4bit"},
        "memory": {"active_bytes": int((srv.rss_gib() or 0) * 1073741824),
                   "peak_bytes": 0,
                   "available_bytes": int((fr or 0) * 1073741824),
                   "max_safe_context": 32768, "cache_bytes": 0,
                   "weights_bytes": 0, "kv_cache_bytes": 0},
        "ngram_warm": {"bytes": 0, "total": 0},
        # 闭源 契约: /props 带 batching 键 (supported / reason / max_group)
        "batching": {"supported": False, "reason": "no_model", "max_group": 32},
        "settings": {
            "version": "our-own kernel",
            "engine": "mlx",
            # —— 上 一次 出片 的 实测 速度 (界面 用 它 估 预计 耗时) ——
            "last_video_sec": getattr(srv, "last_video_sec", 0),
            "last_video_steps": getattr(srv, "last_video_steps", 0),
            "last_video_w": getattr(srv, "last_video_w", 0),
            "last_video_h": getattr(srv, "last_video_h", 0),
            "last_video_frames": getattr(srv, "last_video_frames", 0),
            "last_video_rows": getattr(srv, "last_video_rows", 0),
            "last_video_sec_per_step": getattr(srv, "last_video_sec_per_step", 0),
            "video_fit_a": getattr(srv, "video_fit_a", 0),
            "video_fit_b": getattr(srv, "video_fit_b", 0),
            "video_fit_n": getattr(srv, "video_fit_n", 0),
            # 出歌 引擎 的 时长 真 上限 (界面 滑块 按 这个 画 —— 别 画 到 600 再 被 截断)
            "music_max_duration": _music_max_duration(),
            "kv_quant": os.environ.get("MLX_SERVE_KV_RESERVE", "off"),
            "kv_attn_mode": "off",
            "decode_attn_quant": 0,
            "prefill_chunk": 16384,
            "mtp": {"default_on": False, "block_size": 0, "native": {},
                    "acceptance": None, "acceptance_param": None, "greedy_tail": None,
                    "depth": None, "adaptive": None, "max_ctx": 0},
            "drafter": "none",
            "pld": {"default_on": False, "draft_len": 5, "key_len": 3},
            "max_concurrent": 1,
            "prefill_decode_share": 0,
            "prefix_cache": {"ram_enabled": False, "mem_bytes": 0, "disk_bytes": 0},
            "os_reserve_gib": srv.os_reserve_gib,
            "wired_margin_gib": 8,
            "idle_evict_secs": srv.idle_evict_secs,
            "max_resident_models": 3,
        },
    })


def metrics_prometheus(srv, h):
    """原版 Prometheus 指标 名 (MECHANICS §6)."""
    m = srv.metrics_snapshot()
    lines = [
        "# HELP prompt_tokens_total Prompt tokens processed",
        "# TYPE prompt_tokens_total counter",
        "prompt_tokens_total %d" % m.get("prompt_tokens", 0),
        "# HELP generation_tokens_total Generated tokens",
        "# TYPE generation_tokens_total counter",
        "generation_tokens_total %d" % m.get("generation_tokens", 0),
        "# HELP requests_success_total Successful requests",
        "# TYPE requests_success_total counter",
        "requests_success_total %d" % m.get("req_ok", 0),
        "# HELP requests_failed_total Failed requests",
        "# TYPE requests_failed_total counter",
        "requests_failed_total %d" % m.get("req_err", 0),
        "# HELP requests_running In-flight requests",
        "# TYPE requests_running gauge",
        "requests_running %d" % m.get("running", 0),
        "# HELP gpu_utilization_pct GPU utilization percent",
        "# TYPE gpu_utilization_pct gauge",
        "gpu_utilization_pct %s" % (m.get("gpu_utilization_pct") or 0),
        "# HELP memory_mb Process memory in MB",
        "# TYPE memory_mb gauge",
        "memory_mb %d" % int((m.get("rss_gib") or 0) * 1024),
        "# HELP process_start_time_seconds Process start time",
        "# TYPE process_start_time_seconds gauge",
        "process_start_time_seconds %d" % m.get("start_time", 0),
        "# HELP mlx_active_bytes Active MLX bytes",
        "# TYPE mlx_active_bytes gauge",
        "mlx_active_bytes %d" % m.get("mlx_active_bytes", 0),
        "# HELP mlx_cache_bytes MLX cache bytes",
        "# TYPE mlx_cache_bytes gauge",
        "mlx_cache_bytes %d" % m.get("mlx_cache_bytes", 0),
        "# HELP prefix_cache_hits_total Prefix cache hits",
        "# TYPE prefix_cache_hits_total counter",
        "prefix_cache_hits_total %d" % m.get("prefix_cache_hits", 0),
        "# HELP prefix_cache_queries_total Prefix cache queries",
        "# TYPE prefix_cache_queries_total counter",
        "prefix_cache_queries_total %d" % m.get("prefix_cache_queries", 0),
    ]
    h.send_response(200)
    h.send_header("Content-Type", "text/plain; version=0.0.4")
    h.end_headers()
    h.wfile.write(("\n".join(lines) + "\n").encode())


# ============================================================ 配音工作台 (声音克隆 + 批量配音)
def _voice(srv):
    """取 配音 工作台 单例, 并 挂 到 引擎 表 —— 置闲 30min 自动 卸载 才 生效."""
    import voice_studio as VS
    st = VS.get_studio()
    try:
        srv.engines.setdefault("voice", st)
    except Exception:
        pass
    return st


def _voice_out(h, path):
    return out_url(h, path) if path else ""


def voice_list(srv, h, q):
    """GET /v1/voices —— 音色库 + 引擎 清单."""
    import voice_studio as VS
    st = _voice(srv)
    srv.touch("voice")
    profs = st.profiles()
    for p in profs:
        p["sample_url"] = _voice_out(h, p.get("sample"))
        p["ref_url"] = _voice_out(h, p.get("ref_audio"))
        p["engine_name"] = (VS.ENGINE_BY_ID.get(p.get("engine")) or {}).get(
            "name", p.get("engine"))
    h._j({"object": "list", "voices": profs, "profiles": profs,
          "engines": VS.engine_list(), "state": st.state(),
          "count": len(profs)})


def _voice_ref_file(b, files):
    """从 multipart / 路径 里 拿 参考 音频 —— 返回 临时 文件 路径."""
    import tempfile
    import base64
    files = files or {}
    f = files.get("ref_audio") or files.get("audio") or files.get("file")
    if isinstance(f, list):
        f = f[0] if f else None
    if f:
        ext = os.path.splitext(f.get("filename") or "")[1] or ".wav"
        d = tempfile.mkdtemp(prefix="mx_ref_")
        p = os.path.join(d, "ref" + ext)
        with open(p, "wb") as w:
            w.write(f.get("data") or b"")
        return p, d
    # JSON 里 直接 给 base64 (界面 照 音频页 的 做法)
    b64 = b.get("ref_audio") or b.get("ref_audio_b64") or b.get("audio")
    if isinstance(b64, str) and b64.strip():
        s = b64.strip()
        if s.startswith("data:"):
            s = s.split(",", 1)[-1]
        try:
            raw = base64.b64decode(s + "=" * (-len(s) % 4))
        except Exception:
            raw = b""
        if raw:
            d = tempfile.mkdtemp(prefix="mx_ref_")
            p = os.path.join(d, "ref.wav")
            with open(p, "wb") as w:
                w.write(raw)
            return p, d
    src = str(b.get("ref_audio_path") or b.get("audio_path") or "").strip()
    if src:
        return src, ""
    return "", ""


def voice_clone(srv, h, b):
    """POST /v1/voices —— 一键 克隆 声音 (参考 音频 -> 音色)."""
    import shutil
    import voice_studio as VS
    st = _voice(srv)
    srv.touch("voice")
    src, tmpdir = _voice_ref_file(b, b.get("__files__"))
    if not src or not os.path.isfile(src):
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        err(h, 400, "要 一个 参考 音频 (字段 ref_audio)")
        return
    try:
        prof = st.clone(
            name=str(b.get("name") or "").strip(),
            ref_audio=src,
            engine_id=str(b.get("engine") or b.get("model") or "qwen3-tts-base"),
            ref_text=str(b.get("ref_text") or ""),
            language=str(b.get("language") or b.get("lang") or "zh"),
            kind=str(b.get("kind") or "clone"),
            instruct=str(b.get("instruct") or ""),
            sample_text=str(b.get("sample_text") or ""),
            make_sample=str(b.get("make_sample", "1")).lower() not in ("0", "false", "no"),
        )
    except Exception as e:
        err(h, 400, "克隆 失败: %s" % e)
        return
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    prof["sample_url"] = _voice_out(h, prof.get("sample"))
    prof["ref_url"] = _voice_out(h, prof.get("ref_audio"))
    h._j({"ok": True, "voice": prof, "profile": prof})


def voice_speak(srv, h, b):
    """POST /v1/voices/speak —— 用 某 个 音色 合成 一 条 语音."""
    st = _voice(srv)
    srv.touch("voice")
    srv._begin_use("voice")
    try:
        pid = str(b.get("voice") or b.get("profile_id") or b.get("voice_id") or "")
        prof = st.profile(pid) if pid else None
        if prof is None:
            err(h, 404, "没 这个 音色")
            return
        p = st.synthesize(prof, str(b.get("text") or b.get("input") or ""),
                          speed=float(b.get("speed") or 1.0),
                          lang=b.get("lang") or b.get("language"),
                          engine_id=b.get("engine"))
    except Exception as e:
        err(h, 400, "合成 失败: %s" % e)
        return
    finally:
        srv._end_use("voice")
    h._j({"ok": True, "url": _voice_out(h, p), "path": p,
          "voice": prof.get("name"), "engine": prof.get("engine")})


def _voice_video_path(v):
    """视频 可以 给 绝对 路径, 也 可以 给 /out/ 的 相对 路径."""
    v = str(v or "").strip()
    if not v:
        return ""
    if v.startswith("file://"):
        v = v[7:]
    if os.path.isabs(v) and os.path.isfile(v):
        return v
    try:
        import outputs as O
        rel = v.split("/out/", 1)[-1] if "/out/" in v else v
        p = O.abs_from_rel(rel)
        if p and os.path.isfile(p):
            return p
    except Exception:
        pass
    return ""


def voice_dub(srv, h, b):
    """POST /v1/voices/dub —— 批量 自动 配音."""
    st = _voice(srv)
    srv.touch("voice")
    lines = b.get("lines") or b.get("items") or b.get("segments") or []
    if not lines:
        err(h, 400, "没有 台词 (lines)")
        return
    srv._begin_use("voice")
    try:
        r = st.batch(
            lines,
            profile_id=str(b.get("voice") or b.get("profile_id") or "") or None,
            engine_id=str(b.get("engine") or "") or None,
            speed=float(b.get("speed") or 1.0),
            lang=b.get("lang") or b.get("language"),
            fmt=str(b.get("format") or "wav"),
            join=bool(b.get("join", True)),
            video=_voice_video_path(b.get("video") or b.get("video_path")),
            keep_original=bool(b.get("keep_original")),
            name=str(b.get("name") or "批量配音"),
        )
    except Exception as e:
        err(h, 400, "批量 配音 失败: %s" % e)
        return
    finally:
        srv._end_use("voice")
    try:
        srv.record_library("tts", r.get("joined") or "", model="voice-studio",
                           prompt="%d 条 台词" % len(lines))
    except Exception:
        pass
    out = dict(r)
    for it in out.get("items", []):
        it["audio_url"] = _voice_out(h, it.get("path"))
    out["joined_audio_url"] = _voice_out(h, r.get("joined"))
    out["mp3_audio_url"] = _voice_out(h, r.get("mp3"))
    out["video_audio_url"] = _voice_out(h, r.get("video_out"))
    h._j(out)


def voice_delete(srv, h, pid):
    """DELETE /v1/voices/<id> —— 删 音色."""
    st = _voice(srv)
    srv.touch("voice")
    ok = st.delete_profile(pid)
    h._j({"ok": bool(ok), "deleted": pid if ok else ""})


# ============================================================ 许可 条款
def licenses_list(srv, h):
    """GET /v1/licenses —— 全部 许可 条款 (含 商用门槛 / 地区限制 / 署名要求)."""
    try:
        import licenses as _lic
        docs = _lic.list_docs()
    except Exception as e:
        err(h, 500, "读 许可 清单 失败: %s" % e)
        return
    h._j({"object": "list", "data": docs})


def license_doc(srv, h, key):
    """GET /v1/licenses/<key> —— 一份 许可 的 条款 + 协议 原文."""
    key = (key or "").strip()
    try:
        import licenses as _lic
        docs = {d["key"]: d for d in _lic.list_docs()}
        d = docs.get(key)
        if not d:
            err(h, 404, "没有 这份 许可: %s" % key)
            return
        d = dict(d)
        d["text"] = _lic.read(key)
        d["object"] = "license"
        h._j(d)
    except Exception as e:
        err(h, 500, "读 许可 原文 失败: %s" % e)


# ============================================================ 路由 表
def handle(srv, h, method, path, query, body):
    """返回 True = 已 处理. 未 处理 的 落回 老 路由."""
    p = path.rstrip("/") or "/"

    if method == "GET":
        if p == "/v1/models":
            h._j({"object": "list", "data": model_entries(srv)})
            return True
        if p == "/v1/licenses":
            licenses_list(srv, h)
            return True
        if p.startswith("/v1/licenses/"):
            license_doc(srv, h, p.rsplit("/", 1)[-1])
            return True
        if p == "/v1/providers":
            providers(srv, h)
            return True
        if p == "/v1/audio/voices":
            audio_voices(srv, h, query)
            return True
        if p == "/v1/voices":
            voice_list(srv, h, query)
            return True
        if p == "/v1/library":
            # **不** 在 请求 线程 里 扫 产物 目录 —— launchd 上下文 里 枚举 ~/Documents
            # 可能 被 macOS 卡 住 (opendir 阻塞), 整个 界面 就 卡死.
            # 扫 缩 放 到 后台 线程 (带 超时), 这里 只 读 已经 有 的 索引.
            try: srv.start_library_scan()
            except Exception: pass
            items=srv.library_items()
            h._j({"object": "list", "items": items, "count": len(items)})
            return True
        if p == "/v1/library/export":
            items = srv.library_items()
            h._j({"object": "library.export", "version": 1,
                  "exported": int(time.time()), "items": items})
            return True
        if p == "/props":
            props(srv, h)
            return True
        if p == "/metrics":
            metrics_prometheus(srv, h)
            return True
        if p == "/api/tags":
            ollama_tags(srv, h)
            return True
        if p == "/api/ps":
            ollama_ps(srv, h)
            return True
        if p == "/api/version":
            ollama_version(srv, h)
            return True
        if p == "/api/show":
            ollama_show(srv, h, query)
            return True
        if p == "/api/embed" or p == "/api/embeddings":
            err(h, 405, "GET not allowed on %s (POST)" % p, typ="invalid_request")
            return True
        if p.startswith("/v1/responses/"):
            responses_get(srv, h, p.rsplit("/", 1)[-1])
            return True
        return False

    if method == "DELETE":
        if p.startswith("/v1/responses/"):
            responses_delete(srv, h, p.rsplit("/", 1)[-1])
            return True
        if p.startswith("/v1/voices/"):
            voice_delete(srv, h, p.rsplit("/", 1)[-1])
            return True
        return False

    if method == "POST":
        if p == "/v1/chat/completions":
            chat_completions(srv, h, body)
            return True
        if p == "/v1/completions":
            completions(srv, h, body)
            return True
        if p == "/v1/embeddings":
            embeddings(srv, h, body)
            return True
        if p == "/v1/tokenize":
            tokenize(srv, h, body)
            return True
        if p == "/v1/detokenize":
            detokenize(srv, h, body)
            return True
        if p == "/v1/messages":
            anthropic_messages(srv, h, body)
            return True
        if p == "/v1/responses":
            responses(srv, h, body)
            return True
        if p == "/v1/responses/compact":
            responses_compact(srv, h, body)
            return True
        if p == "/v1/providers/reload":
            providers_reload(srv, h)
            return True
        if p == "/v1/library/import":
            library_import(srv, h, body)
            return True
        if p == "/v1/models/rescan":
            try:
                srv.scan_model_dirs()
            except Exception as e:
                err(h, 500, "rescan failed: %s" % e, typ="server_error")
                return True
            h._j({"rescanned": True, "models": len(srv.models_resp().get("models", []))})
            return True
        if p == "/v1/images/generations":
            images_generations(srv, h, body)
            return True
        if p == "/v1/images/edits":
            images_edits(srv, h, body)
            return True
        if p == "/v1/audio/speech":
            audio_speech(srv, h, body)
            return True
        if p == "/v1/audio/voices":
            audio_voices(srv, h, body)
            return True
        if p == "/v1/voices":
            voice_clone(srv, h, body)
            return True
        if p == "/v1/voices/speak":
            voice_speak(srv, h, body)
            return True
        if p == "/v1/voices/dub":
            voice_dub(srv, h, body)
            return True
        if p == "/v1/audio/music-generations":
            audio_music(srv, h, body)
            return True
        if p == "/v1/audio/sound-generations":
            audio_sound(srv, h, body)
            return True
        if p == "/v1/video/generations":
            video_generations(srv, h, body)
            return True
        if p == "/v1/3d/generations":
            three_d_generations(srv, h, body)
            return True
        if p in ("/v1/decisions", "/v1/systemone"):
            decisions(srv, h, body)
            return True
        if p == "/api/chat":
            ollama_chat(srv, h, body)
            return True
        if p == "/api/generate":
            ollama_generate(srv, h, body)
            return True
        if p == "/api/embed" or p == "/api/embeddings":
            ollama_embed(srv, h, body)
            return True
        if p == "/api/show":
            # 闭源: POST /api/show -> 200 ({...context_length...}) / 404 (unknown model ...)
            ollama_show(srv, h, body)
            return True
        if p == "/api/pull":
            ollama_pull(srv, h, body)
            return True
        if p.startswith("/api/") and p not in ("/api/chat", "/api/generate",
                                               "/api/embed", "/api/embeddings"):
            # 闭源 对 没 做 的 Ollama 端点: 501 (unsupported ollama endpoint)
            ollama_unsupported(srv, h, p)
            return True
        if p == "/v1/audio/transcriptions":
            audio_transcriptions(srv, h, body)
            return True
        return False

    return False


# ============================================================ Responses 流式 事件 (SSE / WebSocket 共用)
def responses_events(srv, eng, model, msgs, b, max_tokens, tb):
    """Responses API 的 事件 序列 (闭源 事件 名 一致):

    response.created -> response.in_progress -> response.output_item.added ->
    response.content_part.added -> [response.reasoning_summary_*.delta] ->
    response.output_text.delta -> response.output_text.done ->
    response.content_part.done -> response.output_item.done -> response.completed

    产 (event_type, data) 二元组 —— SSE 与 WebSocket 两条 通路 共用.
    """
    rid = "resp_" + uuid.uuid4().hex[:24]
    item_id = "msg_" + uuid.uuid4().hex[:12]
    prompt = build_chat_prompt(eng, msgs, thinking=(tb > 0))
    base = {"id": rid, "object": "response", "created_at": int(time.time()),
            "model": model, "status": "in_progress", "output": []}
    yield ("response.created", {"type": "response.created", "response": dict(base)})
    yield ("response.in_progress", {"type": "response.in_progress", "response": dict(base)})
    yield ("response.output_item.added",
           {"type": "response.output_item.added", "output_index": 0,
            "item": {"id": item_id, "type": "message", "role": "assistant",
                     "status": "in_progress", "content": []}})
    yield ("response.content_part.added",
           {"type": "response.content_part.added", "item_id": item_id,
            "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": ""}})
    full = ""
    sent_c = 0
    sent_r = 0
    rsn_open = False
    try:
        for piece in eng.stream(prompt, max_tokens=max_tokens, think_budget=tb):
            full += piece
            rsn, ctt = split_thinking_raw(full)
            if len(rsn) > sent_r:
                if not rsn_open:
                    rsn_open = True
                    yield ("response.reasoning_summary_part.added",
                           {"type": "response.reasoning_summary_part.added",
                            "item_id": item_id, "output_index": 0, "summary_index": 0,
                            "part": {"type": "summary_text", "text": ""}})
                yield ("response.reasoning_summary_text.delta",
                       {"type": "response.reasoning_summary_text.delta",
                        "item_id": item_id, "output_index": 0, "summary_index": 0,
                        "delta": rsn[sent_r:]})
                sent_r = len(rsn)
            if len(ctt) > sent_c:
                yield ("response.output_text.delta",
                       {"type": "response.output_text.delta", "item_id": item_id,
                        "output_index": 0, "content_index": 0, "delta": ctt[sent_c:]})
                sent_c = len(ctt)
    except Exception as e:
        yield ("response.failed",
               {"type": "response.failed",
                "response": dict(base, status="failed",
                                 error={"code": "server_error", "message": str(e)})})
        return
    trunc = bool((getattr(eng, "last_info", {}) or {}).get("truncated"))
    reasoning, content = split_thinking(full)
    if rsn_open:
        yield ("response.reasoning_summary_text.done",
               {"type": "response.reasoning_summary_text.done", "item_id": item_id,
                "output_index": 0, "summary_index": 0, "text": reasoning})
        yield ("response.reasoning_summary_part.done",
               {"type": "response.reasoning_summary_part.done", "item_id": item_id,
                "output_index": 0, "summary_index": 0,
                "part": {"type": "summary_text", "text": reasoning}})
    yield ("response.output_text.done",
           {"type": "response.output_text.done", "item_id": item_id,
            "output_index": 0, "content_index": 0, "text": content})
    yield ("response.content_part.done",
           {"type": "response.content_part.done", "item_id": item_id,
            "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": content}})
    item = {"id": item_id, "type": "message", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": content, "annotations": []}]}
    yield ("response.output_item.done",
           {"type": "response.output_item.done", "output_index": 0, "item": item})
    body = dict(base)
    body["status"] = "incomplete" if trunc else "completed"
    body["output"] = [item]
    body["usage"] = {"input_tokens": len(prompt) // 4, "output_tokens": len(content) // 4,
                     "total_tokens": (len(prompt) + len(content)) // 4}
    try:
        srv.store_response(rid, body)
    except Exception:
        pass
    yield ("response.completed", {"type": "response.completed", "response": body})


def _input_text(inp):
    """把 /v1/responses 的 input (字符串 / 消息 数组) 抽 成 正文."""
    if inp is None:
        return ""
    if isinstance(inp, str):
        return inp.strip()
    if isinstance(inp, list):
        parts = []
        for it in inp:
            if isinstance(it, str):
                parts.append(it)
            elif isinstance(it, dict):
                parts.append(_content_text(it.get("content")))
        return " ".join(p for p in parts if p).strip()
    if isinstance(inp, dict):
        return _content_text(inp.get("content")).strip()
    return ""


def _input_msgs(inp):
    """input -> 消息 数组 (闭源 compact 的 encrypted_content 就是 这个 形状)."""
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}]
    if isinstance(inp, list):
        out = []
        for it in inp:
            if isinstance(it, dict):
                role = it.get("role") or "user"
                ctext = _content_text(it.get("content"))
                out.append({"role": role, "content": ctext})
        return out
    if isinstance(inp, dict):
        role = inp.get("role") or "user"
        ctext = _content_text(inp.get("content"))
        return [{"role": role, "content": ctext}]
    return []


def responses_stream(srv, h, eng, model, msgs, b, max_tokens, tb):
    """POST /v1/responses (stream=true): SSE 版 的 Responses 事件 序列."""
    sse_start(h)
    try:
        for name, data in responses_events(srv, eng, model, msgs, b, max_tokens, tb):
            sse_event(h, name, data)
    except Exception as e:
        sse_event(h, "response.failed",
                  {"type": "response.failed",
                   "response": {"status": "failed",
                                "error": {"code": "server_error", "message": str(e)}}})
    sse_done(h)
    srv.touch("llm")
